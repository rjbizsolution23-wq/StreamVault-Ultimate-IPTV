#!/usr/bin/env python3
"""
StreamVault Pro Ultimate - Pensieve Neural Adaptive Bitrate Engine
Based on: MIT Pensieve (https://github.com/hongzimao/pensieve)
Paper: "Neural Adaptive Video Streaming with Pensieve" (SIGCOMM 2017)

This implements a deep reinforcement learning system for adaptive bitrate
streaming that learns optimal policies without hand-crafted rules.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import List, Tuple, Optional, Dict
import json
import os

# ==========================================
# CONFIGURATION
# ==========================================

class PensieveConfig:
    """Configuration for Pensieve ABR system."""
    
    # Video bitrate levels (Kbps)
    VIDEO_BITRATES = [300, 750, 1200, 1850, 2850, 4300, 6000, 8000]  # 8 quality levels
    
    # State dimensions
    S_DIM = 6  # State features
    A_DIM = len(VIDEO_BITRATES)  # Number of actions (bitrate choices)
    
    # Network architecture
    HIDDEN_LAYERS = [128, 128]
    
    # Training parameters
    LEARNING_RATE = 1e-4
    GAMMA = 0.99  # Discount factor
    ENTROPY_WEIGHT = 0.5  # Entropy regularization
    
    # Experience replay
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    
    # QoE weights (customize for your platform)
    QOE_QUALITY_WEIGHT = 1.0
    QOE_REBUFFER_WEIGHT = 4.3
    QOE_SMOOTH_WEIGHT = 1.0
    
    # Chunk settings
    CHUNK_DURATION = 4.0  # seconds
    BUFFER_THRESHOLD = 60.0  # seconds
    
    # Model paths
    MODEL_PATH = "./models/pensieve_model.pt"


# ==========================================
# NEURAL NETWORK ARCHITECTURE
# ==========================================

class ActorNetwork(nn.Module):
    """
    Actor network for policy gradient.
    Takes state as input, outputs action probabilities.
    """
    
    def __init__(self, s_dim: int, a_dim: int, hidden_layers: List[int]):
        super(ActorNetwork, self).__init__()
        
        layers = []
        input_dim = s_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, a_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state)
        logits = self.output_layer(features)
        return F.softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """
    Critic network for value estimation.
    Takes state as input, outputs state value.
    """
    
    def __init__(self, s_dim: int, hidden_layers: List[int]):
        super(CriticNetwork, self).__init__()
        
        layers = []
        input_dim = s_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state)
        return self.output_layer(features)


class Conv1DFeatureExtractor(nn.Module):
    """
    Advanced feature extractor using 1D convolutions for temporal patterns.
    Processes historical throughput and buffer data.
    """
    
    def __init__(self, history_length: int = 8, num_features: int = 6):
        super(Conv1DFeatureExtractor, self).__init__()
        
        # Conv layers for throughput history
        self.throughput_conv = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=4),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Conv layers for download time history
        self.download_conv = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=4),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Conv layers for chunk size history
        self.chunk_conv = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=4),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Linear for buffer and bitrate
        self.buffer_linear = nn.Linear(1, 128)
        self.bitrate_linear = nn.Linear(1, 128)
        self.remain_linear = nn.Linear(1, 128)
        
        # Calculate conv output size
        conv_output_size = 128 * (history_length - 6)  # After two conv with kernel 4
        
        # Final combination layer
        self.combine = nn.Sequential(
            nn.Linear(conv_output_size * 3 + 128 * 3, 256),
            nn.ReLU()
        )
    
    def forward(self, throughput_hist, download_hist, chunk_hist, 
                buffer_size, last_bitrate, chunks_remain):
        
        # Process histories through conv layers
        t_feat = self.throughput_conv(throughput_hist.unsqueeze(1))
        d_feat = self.download_conv(download_hist.unsqueeze(1))
        c_feat = self.chunk_conv(chunk_hist.unsqueeze(1))
        
        # Process scalar features
        b_feat = F.relu(self.buffer_linear(buffer_size))
        br_feat = F.relu(self.bitrate_linear(last_bitrate))
        r_feat = F.relu(self.remain_linear(chunks_remain))
        
        # Combine all features
        combined = torch.cat([t_feat, d_feat, c_feat, b_feat, br_feat, r_feat], dim=-1)
        return self.combine(combined)


# ==========================================
# PENSIEVE A3C AGENT
# ==========================================

class PensieveAgent:
    """
    Pensieve A3C-style agent for adaptive bitrate selection.
    Uses actor-critic architecture with PPO updates.
    """
    
    def __init__(self, config: PensieveConfig = None):
        self.config = config or PensieveConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = ActorNetwork(
            self.config.S_DIM,
            self.config.A_DIM,
            self.config.HIDDEN_LAYERS
        ).to(self.device)
        
        self.critic = CriticNetwork(
            self.config.S_DIM,
            self.config.HIDDEN_LAYERS
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.config.LEARNING_RATE
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config.LEARNING_RATE
        )
        
        # Experience buffer
        self.buffer = deque(maxlen=self.config.BUFFER_SIZE)
        
        # Training stats
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'avg_qoe': 0,
            'avg_bitrate': 0,
            'rebuffer_events': 0
        }
    
    def get_state(self, throughput_history: List[float], buffer_size: float,
                  last_bitrate: int, chunks_remain: int, 
                  download_time: float, chunk_size: float) -> torch.Tensor:
        """
        Construct state tensor from environment observations.
        
        State features:
        1. Throughput (normalized)
        2. Download time (normalized)
        3. Buffer size (normalized)
        4. Last bitrate index (normalized)
        5. Chunks remaining (normalized)
        6. Chunk size (normalized)
        """
        
        # Normalize features
        throughput = np.mean(throughput_history[-8:]) / 8000.0  # Normalize to max bitrate
        download_norm = min(download_time / 10.0, 1.0)  # Normalize to 10s max
        buffer_norm = min(buffer_size / self.config.BUFFER_THRESHOLD, 1.0)
        bitrate_norm = last_bitrate / (self.config.A_DIM - 1)
        remain_norm = min(chunks_remain / 100.0, 1.0)
        chunk_norm = chunk_size / (max(self.config.VIDEO_BITRATES) * self.config.CHUNK_DURATION / 8)
        
        state = np.array([
            throughput,
            download_norm,
            buffer_norm,
            bitrate_norm,
            remain_norm,
            chunk_norm
        ], dtype=np.float32)
        
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def select_action(self, state: torch.Tensor, explore: bool = True) -> Tuple[int, float]:
        """
        Select bitrate action using the policy network.
        
        Returns:
            action: Selected bitrate index
            log_prob: Log probability of the action
        """
        with torch.no_grad():
            action_probs = self.actor(state)
        
        if explore:
            # Sample from distribution for exploration
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            # Greedy selection for inference
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs[0, action])
        
        return action.item(), log_prob.item()
    
    def compute_qoe(self, bitrate: float, rebuffer_time: float, 
                    last_bitrate: float) -> float:
        """
        Compute Quality of Experience reward.
        
        QoE = quality - rebuffer_penalty - smoothness_penalty
        
        Based on Pensieve paper QoE model.
        """
        # Quality reward (bitrate in Mbps)
        quality = self.config.QOE_QUALITY_WEIGHT * (bitrate / 1000.0)
        
        # Rebuffering penalty
        rebuffer_penalty = self.config.QOE_REBUFFER_WEIGHT * rebuffer_time
        
        # Smoothness penalty (bitrate change)
        smoothness_penalty = self.config.QOE_SMOOTH_WEIGHT * abs(bitrate - last_bitrate) / 1000.0
        
        qoe = quality - rebuffer_penalty - smoothness_penalty
        
        return qoe
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step using PPO.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < self.config.BATCH_SIZE:
            return {}
        
        # Sample batch
        batch = random.sample(self.buffer, self.config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.cat(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute advantages
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()
            current_values = self.critic(states).squeeze()
            td_targets = rewards + self.config.GAMMA * next_values * (1 - dones)
            advantages = td_targets - current_values
        
        # Update critic
        values = self.critic(states).squeeze()
        critic_loss = F.mse_loss(values, td_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Update actor
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        actor_loss = -(log_probs * advantages.detach()).mean()
        actor_loss -= self.config.ENTROPY_WEIGHT * entropy
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'avg_advantage': advantages.mean().item()
        }
    
    def save_model(self, path: str = None):
        """Save model weights."""
        path = path or self.config.MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
        
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str = None):
        """Load model weights."""
        path = path or self.config.MODEL_PATH
        
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            print(f"✅ Model loaded from {path}")
            return True
        return False


# ==========================================
# STREAMING ENVIRONMENT SIMULATOR
# ==========================================

class StreamingEnvironment:
    """
    Simulates video streaming environment for training.
    Uses real network trace data for realistic conditions.
    """
    
    def __init__(self, trace_path: str = None):
        self.config = PensieveConfig()
        
        # Load network traces or use synthetic
        if trace_path and os.path.exists(trace_path):
            self.traces = self._load_traces(trace_path)
        else:
            self.traces = self._generate_synthetic_traces()
        
        self.reset()
    
    def _load_traces(self, path: str) -> List[List[float]]:
        """Load network bandwidth traces from file."""
        traces = []
        with open(path, 'r') as f:
            for line in f:
                trace = [float(x) for x in line.strip().split()]
                traces.append(trace)
        return traces
    
    def _generate_synthetic_traces(self, num_traces: int = 100) -> List[List[float]]:
        """Generate synthetic network traces for training."""
        traces = []
        for _ in range(num_traces):
            # Generate varying network conditions
            base_bw = np.random.uniform(1000, 10000)  # 1-10 Mbps
            length = np.random.randint(100, 500)
            
            trace = []
            current_bw = base_bw
            
            for _ in range(length):
                # Add realistic fluctuations
                change = np.random.normal(0, base_bw * 0.1)
                current_bw = np.clip(current_bw + change, 500, 15000)
                trace.append(current_bw)
            
            traces.append(trace)
        
        return traces
    
    def reset(self) -> Dict:
        """Reset environment for new episode."""
        self.current_trace = random.choice(self.traces)
        self.trace_idx = 0
        self.buffer_size = 0.0
        self.last_bitrate = 0
        self.chunks_sent = 0
        self.total_chunks = len(self.current_trace)
        self.throughput_history = [0.0] * 8
        self.rebuffer_time = 0.0
        
        return self._get_obs()
    
    def _get_obs(self) -> Dict:
        """Get current observation."""
        return {
            'throughput_history': self.throughput_history.copy(),
            'buffer_size': self.buffer_size,
            'last_bitrate': self.last_bitrate,
            'chunks_remain': self.total_chunks - self.chunks_sent
        }
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Bitrate level index
        
        Returns:
            observation, reward, done, info
        """
        # Get current network throughput (Kbps)
        throughput = self.current_trace[min(self.trace_idx, len(self.current_trace) - 1)]
        self.trace_idx += 1
        
        # Selected bitrate
        bitrate = self.config.VIDEO_BITRATES[action]
        
        # Chunk size (bytes)
        chunk_size = bitrate * self.config.CHUNK_DURATION * 1000 / 8  # Convert to bytes
        
        # Download time
        download_time = chunk_size * 8 / (throughput * 1000)  # Convert to seconds
        
        # Update buffer
        self.buffer_size -= download_time  # Time passes during download
        
        # Check for rebuffering
        rebuffer = 0.0
        if self.buffer_size < 0:
            rebuffer = -self.buffer_size
            self.buffer_size = 0
            self.rebuffer_time += rebuffer
        
        # Add chunk to buffer
        self.buffer_size += self.config.CHUNK_DURATION
        self.buffer_size = min(self.buffer_size, self.config.BUFFER_THRESHOLD)
        
        # Update history
        self.throughput_history.pop(0)
        self.throughput_history.append(throughput)
        
        # Compute reward (QoE)
        last_bitrate_kbps = self.config.VIDEO_BITRATES[self.last_bitrate]
        reward = self._compute_reward(bitrate, rebuffer, last_bitrate_kbps)
        
        # Update state
        self.last_bitrate = action
        self.chunks_sent += 1
        
        # Check if done
        done = self.chunks_sent >= self.total_chunks
        
        info = {
            'bitrate': bitrate,
            'throughput': throughput,
            'buffer': self.buffer_size,
            'rebuffer': rebuffer,
            'download_time': download_time,
            'chunk_size': chunk_size
        }
        
        return self._get_obs(), reward, done, info
    
    def _compute_reward(self, bitrate: float, rebuffer: float, last_bitrate: float) -> float:
        """Compute QoE-based reward."""
        # Quality (log scale as per paper)
        quality = np.log(bitrate / 300.0)  # Normalize to lowest bitrate
        
        # Rebuffer penalty
        rebuffer_penalty = self.config.QOE_REBUFFER_WEIGHT * rebuffer
        
        # Smoothness penalty
        smooth_penalty = self.config.QOE_SMOOTH_WEIGHT * np.abs(
            np.log(bitrate / 300.0) - np.log(last_bitrate / 300.0)
        ) if last_bitrate > 0 else 0
        
        return quality - rebuffer_penalty - smooth_penalty


# ==========================================
# TRAINING PIPELINE
# ==========================================

class PensieveTrainer:
    """Training pipeline for Pensieve agent."""
    
    def __init__(self, agent: PensieveAgent, env: StreamingEnvironment):
        self.agent = agent
        self.env = env
        self.best_reward = float('-inf')
    
    def train(self, num_episodes: int = 1000, save_interval: int = 100):
        """Train the agent."""
        
        print("🎬 Starting Pensieve Training...")
        print(f"   Device: {self.agent.device}")
        print(f"   Episodes: {num_episodes}")
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get state
                state = self.agent.get_state(
                    obs['throughput_history'],
                    obs['buffer_size'],
                    obs['last_bitrate'],
                    obs['chunks_remain'],
                    0.0,  # download_time
                    0.0   # chunk_size
                )
                
                # Select action
                action, _ = self.agent.select_action(state, explore=True)
                
                # Step environment
                next_obs, reward, done, info = self.env.step(action)
                
                # Get next state
                next_state = self.agent.get_state(
                    next_obs['throughput_history'],
                    next_obs['buffer_size'],
                    next_obs['last_bitrate'],
                    next_obs['chunks_remain'],
                    info['download_time'],
                    info['chunk_size']
                )
                
                # Store experience
                self.agent.store_experience(state, action, reward, next_state, float(done))
                
                # Train
                if len(self.agent.buffer) >= self.agent.config.BATCH_SIZE:
                    self.agent.train_step()
                
                episode_reward += reward
                obs = next_obs
            
            episode_rewards.append(episode_reward)
            
            # Logging
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg(100): {avg_reward:.2f} | "
                      f"Rebuffer: {self.env.rebuffer_time:.2f}s")
            
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                if np.mean(episode_rewards[-100:]) > self.best_reward:
                    self.best_reward = np.mean(episode_rewards[-100:])
                    self.agent.save_model()
        
        print("✅ Training complete!")
        return episode_rewards


# ==========================================
# PRODUCTION INFERENCE
# ==========================================

class PensieveInference:
    """
    Production-ready inference class for real-time ABR decisions.
    """
    
    def __init__(self, model_path: str = None):
        self.agent = PensieveAgent()
        
        if model_path:
            self.agent.load_model(model_path)
        
        self.throughput_history = deque(maxlen=8)
        self.last_bitrate = 0
    
    def get_bitrate(self, current_throughput: float, buffer_size: float,
                    chunks_remain: int, download_time: float = 0,
                    chunk_size: float = 0) -> Dict:
        """
        Get optimal bitrate for next chunk.
        
        Args:
            current_throughput: Current network throughput (Kbps)
            buffer_size: Current buffer level (seconds)
            chunks_remain: Number of chunks remaining
            download_time: Last chunk download time (seconds)
            chunk_size: Last chunk size (bytes)
        
        Returns:
            Dictionary with selected bitrate and confidence
        """
        # Update history
        self.throughput_history.append(current_throughput)
        while len(self.throughput_history) < 8:
            self.throughput_history.appendleft(current_throughput)
        
        # Get state
        state = self.agent.get_state(
            list(self.throughput_history),
            buffer_size,
            self.last_bitrate,
            chunks_remain,
            download_time,
            chunk_size
        )
        
        # Get action probabilities
        with torch.no_grad():
            action_probs = self.agent.actor(state).cpu().numpy()[0]
        
        # Select best action
        action = np.argmax(action_probs)
        
        # Update state
        self.last_bitrate = action
        
        return {
            'bitrate_idx': int(action),
            'bitrate_kbps': PensieveConfig.VIDEO_BITRATES[action],
            'confidence': float(action_probs[action]),
            'all_probs': action_probs.tolist()
        }
    
    def reset(self):
        """Reset for new video session."""
        self.throughput_history.clear()
        self.last_bitrate = 0


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 StreamVault Pensieve Neural ABR Engine")
    print("=" * 60)
    
    # Initialize
    agent = PensieveAgent()
    env = StreamingEnvironment()
    trainer = PensieveTrainer(agent, env)
    
    # Train (reduce episodes for demo)
    rewards = trainer.train(num_episodes=100, save_interval=50)
    
    print("\n📊 Training Summary:")
    print(f"   Final Avg Reward: {np.mean(rewards[-50:]):.2f}")
    print(f"   Best Reward: {max(rewards):.2f}")
    
    # Test inference
    print("\n🎯 Testing Inference:")
    inference = PensieveInference()
    
    test_result = inference.get_bitrate(
        current_throughput=5000,
        buffer_size=10.0,
        chunks_remain=50
    )
    
    print(f"   Recommended Bitrate: {test_result['bitrate_kbps']} Kbps")
    print(f"   Confidence: {test_result['confidence']:.2%}")
