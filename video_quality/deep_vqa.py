#!/usr/bin/env python3
"""
StreamVault Pro Ultimate - Deep Learning Video Quality Assessment
Based on cutting-edge research papers:
- "Deep Learning for Quality Assessment in Live Video Streaming" (IEEE 2017)
- "QARC: Video Quality Aware Rate Control" (ACM MM 2018)
- "A Brief Survey on Adaptive Video Streaming Quality Assessment" (2022)

Real-time video quality analysis using neural networks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import cv2
from dataclasses import dataclass
from collections import deque
import time
import json
import os

# ==========================================
# CONFIGURATION
# ==========================================

@dataclass
class VQAConfig:
    """Video Quality Assessment configuration."""
    
    # Model settings
    input_size: Tuple[int, int] = (224, 224)
    num_quality_levels: int = 5  # MOS scale 1-5
    
    # Feature extraction
    temporal_window: int = 8  # frames to analyze
    spatial_pooling: str = "adaptive"  # "global", "adaptive", "pyramid"
    
    # Quality metrics weights
    vmaf_weight: float = 0.4
    psnr_weight: float = 0.2
    ssim_weight: float = 0.2
    perceptual_weight: float = 0.2
    
    # Real-time settings
    fps_target: int = 30
    batch_size: int = 4
    
    # Model paths
    model_path: str = "./models/vqa_model.pt"


# ==========================================
# NEURAL NETWORK ARCHITECTURES
# ==========================================

class SpatialFeatureExtractor(nn.Module):
    """
    CNN-based spatial feature extractor.
    Analyzes single frames for quality artifacts.
    """
    
    def __init__(self, pretrained: bool = True):
        super(SpatialFeatureExtractor, self).__init__()
        
        # Use EfficientNet-like architecture
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            # Block 1
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            
            # Global pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(512, 256)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return F.relu(self.fc(features))


class TemporalFeatureExtractor(nn.Module):
    """
    LSTM/Transformer-based temporal feature extractor.
    Analyzes motion and temporal consistency.
    """
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, num_layers: int = 2):
        super(TemporalFeatureExtractor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.fc = nn.Linear(hidden_dim * 2, 128)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        return F.relu(self.fc(context))


class QualityArtifactDetector(nn.Module):
    """
    Detects specific video quality artifacts:
    - Blocking artifacts
    - Blurriness
    - Noise
    - Banding
    - Compression artifacts
    """
    
    def __init__(self):
        super(QualityArtifactDetector, self).__init__()
        
        # Artifact-specific convolutions
        self.block_detector = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=8),  # Block-aligned
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.blur_detector = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=15, padding=7),  # Large kernel for blur
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.noise_detector = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # High-frequency
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(48, 32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block_features = self.block_detector(x).view(x.size(0), -1)
        blur_features = self.blur_detector(x).view(x.size(0), -1)
        noise_features = self.noise_detector(x).view(x.size(0), -1)
        
        combined = torch.cat([block_features, blur_features, noise_features], dim=1)
        return F.relu(self.fc(combined))


class DeepVQAModel(nn.Module):
    """
    Complete Deep Video Quality Assessment model.
    Combines spatial, temporal, and artifact features.
    """
    
    def __init__(self, config: VQAConfig):
        super(DeepVQAModel, self).__init__()
        
        self.config = config
        
        # Feature extractors
        self.spatial_extractor = SpatialFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor(input_dim=256)
        self.artifact_detector = QualityArtifactDetector()
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Quality prediction heads
        self.quality_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # MOS score (1-5)
        )
        
        self.artifact_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 artifact types
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for quality assessment.
        
        Args:
            frames: (batch, temporal_window, C, H, W)
        
        Returns:
            Dictionary with quality score, artifacts, and confidence
        """
        batch_size, seq_len, C, H, W = frames.shape
        
        # Extract spatial features for each frame
        frames_flat = frames.view(batch_size * seq_len, C, H, W)
        spatial_features = self.spatial_extractor(frames_flat)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        # Extract artifact features from middle frame
        middle_frame = frames[:, seq_len // 2]
        artifact_features = self.artifact_detector(middle_frame)
        
        # Extract temporal features
        temporal_features = self.temporal_extractor(spatial_features)
        
        # Global spatial feature (mean pooling over time)
        global_spatial = spatial_features.mean(dim=1)
        
        # Fuse all features
        combined = torch.cat([global_spatial, temporal_features, artifact_features], dim=1)
        fused = self.fusion(combined)
        
        # Predictions
        quality_score = self.quality_head(fused)
        quality_score = torch.clamp(quality_score, 1.0, 5.0)  # MOS range
        
        artifact_scores = torch.sigmoid(self.artifact_head(fused))
        confidence = self.confidence_head(fused)
        
        return {
            'quality_score': quality_score.squeeze(-1),
            'artifact_scores': artifact_scores,
            'confidence': confidence.squeeze(-1)
        }


# ==========================================
# REAL-TIME VIDEO QUALITY ANALYZER
# ==========================================

class RealTimeVQA:
    """
    Real-time video quality analysis for live streaming.
    Provides frame-by-frame and segment-based quality metrics.
    """
    
    def __init__(self, config: VQAConfig = None, device: str = None):
        self.config = config or VQAConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize model
        self.model = DeepVQAModel(self.config).to(self.device)
        self.model.eval()
        
        # Frame buffer for temporal analysis
        self.frame_buffer = deque(maxlen=self.config.temporal_window)
        
        # Statistics
        self.stats = {
            'frames_analyzed': 0,
            'avg_quality': 0.0,
            'min_quality': 5.0,
            'max_quality': 1.0,
            'quality_variance': 0.0,
            'artifact_counts': {
                'blocking': 0,
                'blurriness': 0,
                'noise': 0,
                'banding': 0,
                'compression': 0
            }
        }
        
        self.quality_history = deque(maxlen=1000)
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a frame for model input."""
        # Resize
        frame = cv2.resize(frame, self.config.input_size)
        
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(frame).permute(2, 0, 1)
        
        return tensor
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame and update statistics.
        
        Args:
            frame: BGR frame from video capture
        
        Returns:
            Quality analysis results
        """
        # Preprocess and add to buffer
        tensor = self.preprocess_frame(frame)
        self.frame_buffer.append(tensor)
        
        # Need full temporal window for analysis
        if len(self.frame_buffer) < self.config.temporal_window:
            return {'status': 'buffering', 'frames_needed': self.config.temporal_window - len(self.frame_buffer)}
        
        # Stack frames into batch
        frames = torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            results = self.model(frames)
        
        # Extract results
        quality_score = results['quality_score'].item()
        artifact_scores = results['artifact_scores'].cpu().numpy()[0]
        confidence = results['confidence'].item()
        
        # Update statistics
        self._update_stats(quality_score, artifact_scores)
        
        # Artifact names
        artifact_names = ['blocking', 'blurriness', 'noise', 'banding', 'compression']
        artifacts = {name: float(score) for name, score in zip(artifact_names, artifact_scores)}
        
        return {
            'status': 'analyzed',
            'quality_score': quality_score,
            'quality_label': self._score_to_label(quality_score),
            'confidence': confidence,
            'artifacts': artifacts,
            'dominant_artifact': artifact_names[np.argmax(artifact_scores)],
            'statistics': self._get_current_stats()
        }
    
    def _update_stats(self, quality_score: float, artifact_scores: np.ndarray):
        """Update running statistics."""
        self.stats['frames_analyzed'] += 1
        self.quality_history.append(quality_score)
        
        # Update quality stats
        self.stats['min_quality'] = min(self.stats['min_quality'], quality_score)
        self.stats['max_quality'] = max(self.stats['max_quality'], quality_score)
        
        # Running average
        n = self.stats['frames_analyzed']
        old_avg = self.stats['avg_quality']
        self.stats['avg_quality'] = old_avg + (quality_score - old_avg) / n
        
        # Variance (Welford's algorithm)
        if n > 1:
            self.stats['quality_variance'] = np.var(list(self.quality_history))
        
        # Artifact counts (threshold at 0.5)
        artifact_names = ['blocking', 'blurriness', 'noise', 'banding', 'compression']
        for name, score in zip(artifact_names, artifact_scores):
            if score > 0.5:
                self.stats['artifact_counts'][name] += 1
    
    def _get_current_stats(self) -> Dict:
        """Get current statistics snapshot."""
        return {
            'frames_analyzed': self.stats['frames_analyzed'],
            'avg_quality': round(self.stats['avg_quality'], 2),
            'min_quality': round(self.stats['min_quality'], 2),
            'max_quality': round(self.stats['max_quality'], 2),
            'quality_std': round(np.sqrt(self.stats['quality_variance']), 2),
            'artifact_counts': self.stats['artifact_counts'].copy()
        }
    
    @staticmethod
    def _score_to_label(score: float) -> str:
        """Convert MOS score to quality label."""
        if score >= 4.5:
            return "Excellent"
        elif score >= 3.5:
            return "Good"
        elif score >= 2.5:
            return "Fair"
        elif score >= 1.5:
            return "Poor"
        else:
            return "Bad"
    
    def reset_stats(self):
        """Reset statistics for new session."""
        self.stats = {
            'frames_analyzed': 0,
            'avg_quality': 0.0,
            'min_quality': 5.0,
            'max_quality': 1.0,
            'quality_variance': 0.0,
            'artifact_counts': {
                'blocking': 0,
                'blurriness': 0,
                'noise': 0,
                'banding': 0,
                'compression': 0
            }
        }
        self.quality_history.clear()
        self.frame_buffer.clear()
    
    def save_model(self, path: str = None):
        """Save model weights."""
        path = path or self.config.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"✅ VQA model saved to {path}")
    
    def load_model(self, path: str = None):
        """Load model weights."""
        path = path or self.config.model_path
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            print(f"✅ VQA model loaded from {path}")
            return True
        return False


# ==========================================
# TRADITIONAL QUALITY METRICS
# ==========================================

class TraditionalMetrics:
    """
    Traditional video quality metrics for comparison and validation.
    Includes PSNR, SSIM, and basic artifact detection.
    """
    
    @staticmethod
    def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10(255.0 ** 2 / mse)
    
    @staticmethod
    def calculate_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        # Convert to grayscale if needed
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
        
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        original = original.astype(np.float64)
        compressed = compressed.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(original, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(compressed, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(original ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(compressed ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(original * compressed, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
    
    @staticmethod
    def detect_blocking(frame: np.ndarray, block_size: int = 8) -> float:
        """Detect blocking artifacts from compression."""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame = frame.astype(np.float64)
        h, w = frame.shape
        
        # Calculate horizontal and vertical block boundaries
        h_diff = np.abs(frame[:, block_size-1:-1:block_size] - frame[:, block_size::block_size])
        v_diff = np.abs(frame[block_size-1:-1:block_size, :] - frame[block_size::block_size, :])
        
        # Calculate non-boundary differences
        h_non = np.abs(frame[:, :-1] - frame[:, 1:])
        v_non = np.abs(frame[:-1, :] - frame[1:, :])
        
        # Blocking metric
        if np.mean(h_non) == 0 and np.mean(v_non) == 0:
            return 0.0
        
        blocking = (np.mean(h_diff) + np.mean(v_diff)) / (np.mean(h_non) + np.mean(v_non) + 1e-6)
        return float(np.clip(blocking - 1.0, 0, 1))
    
    @staticmethod
    def detect_blur(frame: np.ndarray) -> float:
        """Detect blurriness using Laplacian variance."""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize (higher variance = sharper)
        # Return blur score (inverted)
        blur_threshold = 500  # Typical threshold
        blur_score = np.clip(1.0 - variance / blur_threshold, 0, 1)
        
        return float(blur_score)
    
    @staticmethod
    def detect_noise(frame: np.ndarray) -> float:
        """Estimate noise level in frame."""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # High-pass filter to isolate noise
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        filtered = cv2.filter2D(frame.astype(np.float64), -1, kernel)
        
        # Estimate noise as MAD (median absolute deviation)
        noise_estimate = np.median(np.abs(filtered - np.median(filtered)))
        
        # Normalize
        noise_score = np.clip(noise_estimate / 30.0, 0, 1)
        
        return float(noise_score)


# ==========================================
# VIDEO STREAM QUALITY MONITOR
# ==========================================

class StreamQualityMonitor:
    """
    Complete stream quality monitoring system.
    Combines deep learning and traditional metrics.
    """
    
    def __init__(self, config: VQAConfig = None):
        self.config = config or VQAConfig()
        self.deep_vqa = RealTimeVQA(self.config)
        self.traditional = TraditionalMetrics()
        
        self.monitoring = False
        self.results_buffer = deque(maxlen=1000)
    
    def analyze_stream(self, frame: np.ndarray, reference_frame: np.ndarray = None) -> Dict:
        """
        Comprehensive stream quality analysis.
        
        Args:
            frame: Current frame to analyze
            reference_frame: Optional reference for full-reference metrics
        
        Returns:
            Complete quality analysis
        """
        results = {}
        
        # Deep learning analysis
        deep_results = self.deep_vqa.analyze_frame(frame)
        results['deep_learning'] = deep_results
        
        # Traditional metrics (no-reference)
        results['traditional'] = {
            'blocking': self.traditional.detect_blocking(frame),
            'blur': self.traditional.detect_blur(frame),
            'noise': self.traditional.detect_noise(frame)
        }
        
        # Full-reference metrics if available
        if reference_frame is not None:
            results['full_reference'] = {
                'psnr': self.traditional.calculate_psnr(reference_frame, frame),
                'ssim': self.traditional.calculate_ssim(reference_frame, frame)
            }
        
        # Combined quality score
        if deep_results.get('status') == 'analyzed':
            results['combined_score'] = self._calculate_combined_score(results)
        
        # Store results
        self.results_buffer.append(results)
        
        return results
    
    def _calculate_combined_score(self, results: Dict) -> float:
        """Calculate weighted combined quality score."""
        score = 0.0
        
        # Deep learning score (MOS scale 1-5, normalize to 0-1)
        dl_score = (results['deep_learning']['quality_score'] - 1) / 4.0
        score += self.config.perceptual_weight * dl_score
        
        # Traditional metrics (inverted since they measure degradation)
        trad = results['traditional']
        trad_score = 1.0 - (trad['blocking'] + trad['blur'] + trad['noise']) / 3.0
        score += (1 - self.config.perceptual_weight) * trad_score
        
        # Full-reference if available
        if 'full_reference' in results:
            fr = results['full_reference']
            # PSNR score (30-50 dB is good range)
            psnr_score = np.clip((fr['psnr'] - 20) / 30, 0, 1)
            ssim_score = fr['ssim']
            
            # Weighted average with FR metrics
            score = 0.4 * score + 0.3 * psnr_score + 0.3 * ssim_score
        
        return float(np.clip(score, 0, 1))
    
    def get_summary_report(self) -> Dict:
        """Generate summary report of analyzed stream."""
        if not self.results_buffer:
            return {'status': 'no_data'}
        
        # Aggregate statistics
        quality_scores = []
        artifacts = {'blocking': [], 'blur': [], 'noise': []}
        
        for result in self.results_buffer:
            if 'combined_score' in result:
                quality_scores.append(result['combined_score'])
            
            if 'traditional' in result:
                for key in artifacts:
                    artifacts[key].append(result['traditional'].get(key, 0))
        
        report = {
            'frames_analyzed': len(self.results_buffer),
            'quality': {
                'mean': np.mean(quality_scores) if quality_scores else 0,
                'std': np.std(quality_scores) if quality_scores else 0,
                'min': np.min(quality_scores) if quality_scores else 0,
                'max': np.max(quality_scores) if quality_scores else 0
            },
            'artifacts': {
                key: {
                    'mean': np.mean(values) if values else 0,
                    'max': np.max(values) if values else 0
                }
                for key, values in artifacts.items()
            },
            'recommendations': self._generate_recommendations(quality_scores, artifacts)
        }
        
        return report
    
    def _generate_recommendations(self, quality_scores: List, artifacts: Dict) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if not quality_scores:
            return recommendations
        
        avg_quality = np.mean(quality_scores)
        
        if avg_quality < 0.5:
            recommendations.append("⚠️ Overall quality is poor. Consider increasing bitrate.")
        
        if artifacts['blocking'] and np.mean(artifacts['blocking']) > 0.3:
            recommendations.append("🔲 High blocking artifacts detected. Increase encoding quality or use deblocking filter.")
        
        if artifacts['blur'] and np.mean(artifacts['blur']) > 0.4:
            recommendations.append("🌫️ Excessive blur detected. Check encoder settings or increase resolution.")
        
        if artifacts['noise'] and np.mean(artifacts['noise']) > 0.3:
            recommendations.append("📺 High noise levels. Consider applying denoise filter or improving source quality.")
        
        if np.std(quality_scores) > 0.2:
            recommendations.append("📊 High quality variance. Consider more consistent encoding settings.")
        
        if not recommendations:
            recommendations.append("✅ Stream quality is good. No immediate improvements needed.")
        
        return recommendations


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎬 StreamVault Deep Video Quality Assessment")
    print("=" * 60)
    
    # Initialize
    config = VQAConfig()
    monitor = StreamQualityMonitor(config)
    
    print(f"\n📊 Configuration:")
    print(f"   Input size: {config.input_size}")
    print(f"   Temporal window: {config.temporal_window}")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Demo with synthetic frames
    print("\n🧪 Testing with synthetic frames...")
    
    for i in range(20):
        # Generate synthetic test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some structure
        cv2.rectangle(frame, (100, 100), (540, 380), (128, 128, 128), -1)
        
        result = monitor.analyze_stream(frame)
        
        if result['deep_learning'].get('status') == 'analyzed':
            print(f"   Frame {i+1}: Quality={result['deep_learning']['quality_score']:.2f} "
                  f"({result['deep_learning']['quality_label']})")
    
    # Generate report
    report = monitor.get_summary_report()
    print(f"\n📈 Summary Report:")
    print(f"   Frames analyzed: {report['frames_analyzed']}")
    print(f"   Quality: {report['quality']['mean']:.2f} ± {report['quality']['std']:.2f}")
    
    print("\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"   {rec}")
    
    print("\n✅ VQA system ready for integration!")
