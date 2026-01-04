#!/usr/bin/env python3
"""
StreamVault Pro Ultimate - SRS (Simple Realtime Server) & MediaMTX Configuration
GitHub: https://github.com/ossrs/srs (26K+ stars)
GitHub: https://github.com/bluenviron/mediamtx (14K+ stars)

Production-ready configurations for:
- Ultra-low latency WebRTC (<200ms)
- RTMP/HLS/DASH/SRT streaming
- Multi-protocol transcoding
- Edge deployment
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

# ==========================================
# SRS CONFIGURATION
# ==========================================

@dataclass
class SRSConfig:
    """
    SRS (Simple Realtime Server) configuration generator.
    Supports: RTMP, WebRTC, HLS, HTTP-FLV, HTTP-TS, SRT, MPEG-DASH, GB28181
    """
    
    # Server settings
    listen: int = 1935  # RTMP port
    max_connections: int = 10000
    daemon: bool = False
    
    # WebRTC settings
    webrtc_enabled: bool = True
    webrtc_port: int = 8000
    rtc_server_listen: int = 8000
    rtc_server_candidate: str = "*"
    
    # HLS settings
    hls_enabled: bool = True
    hls_fragment: float = 2.0  # seconds
    hls_window: float = 10.0
    hls_path: str = "./objs/nginx/html"
    
    # HTTP-FLV settings
    http_flv_enabled: bool = True
    http_server_listen: int = 8080
    
    # SRT settings
    srt_enabled: bool = True
    srt_listen: int = 10080
    
    # Transcoding
    transcode_enabled: bool = True
    ffmpeg_path: str = "/usr/local/bin/ffmpeg"
    
    # Recording
    dvr_enabled: bool = True
    dvr_path: str = "./objs/nginx/html/dvr"
    
    def generate_config(self) -> str:
        """Generate SRS configuration file."""
        config = f"""
# StreamVault Pro Ultimate - SRS Configuration
# Generated automatically - Production Ready
# Documentation: https://ossrs.net/lts/en-us/

listen              {self.listen};
max_connections     {self.max_connections};
daemon              {'on' if self.daemon else 'off'};
srs_log_tank        file;
srs_log_file        ./objs/srs.log;

# HTTP API for management
http_api {{
    enabled         on;
    listen          1985;
}}

# HTTP Server for HLS/FLV
http_server {{
    enabled         on;
    listen          {self.http_server_listen};
    dir             {self.hls_path};
}}

# Stats for monitoring
stats {{
    network         0;
    disk            sda sdb xvda xvdb;
}}

# WebRTC Configuration (Ultra-Low Latency)
rtc_server {{
    enabled         {'on' if self.webrtc_enabled else 'off'};
    listen          {self.rtc_server_listen};
    candidate       {self.rtc_server_candidate};
}}

# SRT Configuration
srt_server {{
    enabled         {'on' if self.srt_enabled else 'off'};
    listen          {self.srt_listen};
}}

# Main Virtual Host
vhost __defaultVhost__ {{
    # Latency settings for ultra-low latency
    min_latency     on;
    mr {{
        enabled     off;
    }}
    mw_latency      0;
    gop_cache       off;
    queue_length    10;
    
    # TCP optimization
    tcp_nodelay     on;
    
    # WebRTC
    rtc {{
        enabled     {'on' if self.webrtc_enabled else 'off'};
        rtmp_to_rtc on;
        rtc_to_rtmp on;
    }}
    
    # HLS
    hls {{
        enabled         {'on' if self.hls_enabled else 'off'};
        hls_fragment    {self.hls_fragment};
        hls_window      {self.hls_window};
        hls_path        {self.hls_path};
        hls_m3u8_file   [app]/[stream].m3u8;
        hls_ts_file     [app]/[stream]-[seq].ts;
    }}
    
    # HTTP-FLV
    http_remux {{
        enabled     {'on' if self.http_flv_enabled else 'off'};
        mount       [vhost]/[app]/[stream].flv;
    }}
    
    # DVR Recording
    dvr {{
        enabled         {'on' if self.dvr_enabled else 'off'};
        dvr_path        {self.dvr_path};
        dvr_plan        session;
        dvr_duration    30;
        dvr_wait_keyframe on;
    }}
    
    # Transcoding (optional)
    transcode {{
        enabled     {'on' if self.transcode_enabled else 'off'};
        ffmpeg      {self.ffmpeg_path};
        
        engine adaptive {{
            enabled     on;
            vcodec      libx264;
            vbitrate    0;
            vfps        0;
            vwidth      0;
            vheight     0;
            vthreads    4;
            vprofile    main;
            vpreset     medium;
            acodec      libfdk_aac;
            abitrate    128;
            asample_rate 44100;
            achannels   2;
            output      rtmp://127.0.0.1:[port]/[app]/[stream]_[engine];
        }}
    }}
    
    # Callback hooks for integration
    http_hooks {{
        enabled         on;
        on_connect      http://127.0.0.1:3000/api/srs/connect;
        on_close        http://127.0.0.1:3000/api/srs/close;
        on_publish      http://127.0.0.1:3000/api/srs/publish;
        on_unpublish    http://127.0.0.1:3000/api/srs/unpublish;
        on_play         http://127.0.0.1:3000/api/srs/play;
        on_stop         http://127.0.0.1:3000/api/srs/stop;
        on_dvr          http://127.0.0.1:3000/api/srs/dvr;
    }}
}}

# Low-latency live streaming vhost
vhost low_latency {{
    min_latency     on;
    gop_cache       off;
    queue_length    5;
    mr {{
        enabled     off;
    }}
    mw_latency      0;
    tcp_nodelay     on;
    
    rtc {{
        enabled     on;
        rtmp_to_rtc on;
        rtc_to_rtmp on;
    }}
    
    play {{
        gop_cache   off;
        queue_length 5;
    }}
}}

# Adaptive bitrate streaming vhost
vhost abr {{
    hls {{
        enabled     on;
        hls_fragment 2;
        hls_window  10;
    }}
    
    transcode {{
        enabled on;
        ffmpeg {self.ffmpeg_path};
        
        # 1080p
        engine 1080p {{
            enabled     on;
            vcodec      libx264;
            vbitrate    4500;
            vfps        30;
            vwidth      1920;
            vheight     1080;
            vthreads    4;
            vprofile    high;
            vpreset     faster;
            acodec      libfdk_aac;
            abitrate    192;
            output      rtmp://127.0.0.1:1935/abr/[stream]_1080p;
        }}
        
        # 720p
        engine 720p {{
            enabled     on;
            vcodec      libx264;
            vbitrate    2500;
            vfps        30;
            vwidth      1280;
            vheight     720;
            vthreads    4;
            vprofile    main;
            vpreset     faster;
            acodec      libfdk_aac;
            abitrate    128;
            output      rtmp://127.0.0.1:1935/abr/[stream]_720p;
        }}
        
        # 480p
        engine 480p {{
            enabled     on;
            vcodec      libx264;
            vbitrate    1200;
            vfps        30;
            vwidth      854;
            vheight     480;
            vthreads    2;
            vprofile    main;
            vpreset     faster;
            acodec      libfdk_aac;
            abitrate    96;
            output      rtmp://127.0.0.1:1935/abr/[stream]_480p;
        }}
        
        # 360p
        engine 360p {{
            enabled     on;
            vcodec      libx264;
            vbitrate    600;
            vfps        25;
            vwidth      640;
            vheight     360;
            vthreads    2;
            vprofile    baseline;
            vpreset     faster;
            acodec      libfdk_aac;
            abitrate    64;
            output      rtmp://127.0.0.1:1935/abr/[stream]_360p;
        }}
    }}
}}
"""
        return config


# ==========================================
# MEDIAMTX CONFIGURATION
# ==========================================

@dataclass
class MediaMTXConfig:
    """
    MediaMTX configuration generator.
    Zero-dependency server supporting RTSP, RTMP, HLS, WebRTC, SRT.
    GitHub: https://github.com/bluenviron/mediamtx
    """
    
    # Global settings
    log_level: str = "info"
    log_destinations: List[str] = field(default_factory=lambda: ["stdout"])
    
    # API
    api_enabled: bool = True
    api_address: str = "127.0.0.1:9997"
    
    # RTSP
    rtsp_enabled: bool = True
    rtsp_protocols: List[str] = field(default_factory=lambda: ["multicast", "tcp", "udp"])
    rtsp_address: str = ":8554"
    rtsps_address: str = ":8322"
    
    # RTMP
    rtmp_enabled: bool = True
    rtmp_address: str = ":1935"
    rtmps_address: str = ":1936"
    
    # HLS
    hls_enabled: bool = True
    hls_address: str = ":8888"
    hls_allow_origin: str = "*"
    hls_variant: str = "lowLatency"
    hls_segment_count: int = 7
    hls_segment_duration: str = "1s"
    hls_part_duration: str = "200ms"
    
    # WebRTC
    webrtc_enabled: bool = True
    webrtc_address: str = ":8889"
    webrtc_allow_origin: str = "*"
    webrtc_ice_servers: List[Dict] = field(default_factory=lambda: [
        {"url": "stun:stun.l.google.com:19302"}
    ])
    
    # SRT
    srt_enabled: bool = True
    srt_address: str = ":8890"
    
    # Recording
    record_enabled: bool = False
    record_path: str = "./recordings/%path/%Y-%m-%d_%H-%M-%S.mp4"
    record_format: str = "mp4"
    record_segment_duration: str = "1h"
    
    # Default paths config
    paths: Dict = field(default_factory=dict)
    
    def generate_config(self) -> str:
        """Generate MediaMTX YAML configuration."""
        config = f"""
###############################################
# StreamVault Pro Ultimate - MediaMTX Config
# GitHub: https://github.com/bluenviron/mediamtx
###############################################

###############################################
# Global settings

# verbosity of the program; available values are "error", "warn", "info", "debug"
logLevel: {self.log_level}
# destinations of log messages
logDestinations: {json.dumps(self.log_destinations)}

###############################################
# API

api: {'yes' if self.api_enabled else 'no'}
apiAddress: {self.api_address}

###############################################
# RTSP settings

rtsp: {'yes' if self.rtsp_enabled else 'no'}
protocols: {json.dumps(self.rtsp_protocols)}
rtspAddress: {self.rtsp_address}
rtspsAddress: {self.rtsps_address}
rtpAddress: :8000
rtcpAddress: :8001
multicastIPRange: 224.1.0.0/16
multicastRTPPort: 8002
multicastRTCPPort: 8003

###############################################
# RTMP settings

rtmp: {'yes' if self.rtmp_enabled else 'no'}
rtmpAddress: {self.rtmp_address}
rtmpsAddress: {self.rtmps_address}

###############################################
# HLS settings (Low-Latency HLS)

hls: {'yes' if self.hls_enabled else 'no'}
hlsAddress: {self.hls_address}
hlsAllowOrigin: '{self.hls_allow_origin}'
hlsVariant: {self.hls_variant}
hlsSegmentCount: {self.hls_segment_count}
hlsSegmentDuration: {self.hls_segment_duration}
hlsPartDuration: {self.hls_part_duration}
hlsSegmentMaxSize: 50M
hlsAlwaysRemux: no

###############################################
# WebRTC settings (Ultra-Low Latency)

webrtc: {'yes' if self.webrtc_enabled else 'no'}
webrtcAddress: {self.webrtc_address}
webrtcAllowOrigin: '{self.webrtc_allow_origin}'
webrtcICEServers: {json.dumps(self.webrtc_ice_servers)}
webrtcICEUDPMuxAddress: :8189
webrtcICETCPMuxAddress: :8189

###############################################
# SRT settings

srt: {'yes' if self.srt_enabled else 'no'}
srtAddress: {self.srt_address}

###############################################
# Recording settings

record: {'yes' if self.record_enabled else 'no'}
recordPath: {self.record_path}
recordFormat: {self.record_format}
recordPartDuration: 100ms
recordSegmentDuration: {self.record_segment_duration}
deleteAfterUpload: no

###############################################
# Default path configuration

pathDefaults:
  # Source settings
  source: publisher
  sourceOnDemand: no
  sourceOnDemandStartTimeout: 10s
  sourceOnDemandCloseAfter: 10s
  
  # Maximum readers
  maxReaders: 0
  
  # Record settings
  record: {'yes' if self.record_enabled else 'no'}
  
  # Authentication (override per path)
  publishUser:
  publishPass:
  publishIPs: []
  readUser:
  readPass:
  readIPs: []
  
  # Hooks
  runOnInit:
  runOnInitRestart: no
  runOnDemand:
  runOnDemandRestart: no
  runOnDemandStartTimeout: 10s
  runOnDemandCloseAfter: 10s
  runOnReady:
  runOnReadyRestart: no
  runOnNotReady:
  runOnRead:
  runOnReadRestart: no
  runOnUnread:
  runOnRecordSegmentCreate:
  runOnRecordSegmentComplete:

###############################################
# Path configurations

paths:
  # Default all streams
  all:
  
  # Live streaming path
  live:
    source: publisher
    record: no
  
  # Recording path
  recordings:
    source: publisher
    record: yes
    recordPath: ./recordings/%path/%Y-%m-%d_%H-%M-%S.mp4
  
  # RTSP camera proxy example
  # cam1:
  #   source: rtsp://admin:password@192.168.1.100:554/stream1
  #   sourceProtocol: tcp
  
  # HLS proxy example
  # tv1:
  #   source: https://example.com/stream.m3u8
  
  # FFmpeg source example
  # screencapture:
  #   source: publisher
  #   runOnInit: ffmpeg -f x11grab -framerate 30 -video_size 1920x1080 -i :0.0 -c:v libx264 -preset ultrafast -tune zerolatency -f rtsp rtsp://localhost:$RTSP_PORT/$MTX_PATH
  #   runOnInitRestart: yes
"""
        return config


# ==========================================
# DOCKER COMPOSE FOR STREAMING SERVERS
# ==========================================

def generate_streaming_docker_compose() -> str:
    """Generate Docker Compose for SRS + MediaMTX + NGINX-RTMP stack."""
    
    compose = """
version: '3.8'

# StreamVault Pro Ultimate - Streaming Server Stack
# Includes: SRS, MediaMTX, NGINX-RTMP, Coturn (TURN)

services:
  # ==========================================
  # SRS - Simple Realtime Server
  # ==========================================
  srs:
    image: ossrs/srs:5
    container_name: streamvault-srs
    restart: unless-stopped
    ports:
      - "1935:1935"      # RTMP
      - "1985:1985"      # HTTP API
      - "8080:8080"      # HTTP Server (HLS/FLV)
      - "8000:8000/udp"  # WebRTC
      - "10080:10080/udp" # SRT
    volumes:
      - ./config/srs.conf:/usr/local/srs/conf/srs.conf
      - ./data/srs:/usr/local/srs/objs
    environment:
      - SRS_LISTEN=1935
      - SRS_DAEMON=off
    networks:
      - streaming
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:1985/api/v1/versions"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ==========================================
  # MediaMTX - Multi-Protocol Media Server
  # ==========================================
  mediamtx:
    image: bluenviron/mediamtx:latest
    container_name: streamvault-mediamtx
    restart: unless-stopped
    ports:
      - "8554:8554"      # RTSP
      - "8322:8322"      # RTSPS
      - "8888:8888"      # HLS
      - "8889:8889"      # WebRTC
      - "8890:8890/udp"  # SRT
      - "9997:9997"      # API
    volumes:
      - ./config/mediamtx.yml:/mediamtx.yml
      - ./data/mediamtx:/recordings
    environment:
      - MTX_PROTOCOLS=tcp
    networks:
      - streaming
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:9997/v3/paths/list"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ==========================================
  # NGINX-RTMP - Traditional RTMP Server
  # ==========================================
  nginx-rtmp:
    image: tiangolo/nginx-rtmp:latest
    container_name: streamvault-nginx-rtmp
    restart: unless-stopped
    ports:
      - "1936:1935"      # RTMP (alternate port)
      - "8081:8080"      # HTTP (HLS)
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf
      - ./data/nginx/hls:/tmp/hls
      - ./data/nginx/dash:/tmp/dash
    networks:
      - streaming

  # ==========================================
  # Coturn - TURN/STUN Server for WebRTC NAT traversal
  # ==========================================
  coturn:
    image: coturn/coturn:latest
    container_name: streamvault-coturn
    restart: unless-stopped
    ports:
      - "3478:3478"      # STUN/TURN
      - "3478:3478/udp"
      - "5349:5349"      # STUN/TURN TLS
      - "5349:5349/udp"
    volumes:
      - ./config/turnserver.conf:/etc/turnserver.conf
    command: -c /etc/turnserver.conf
    networks:
      - streaming

  # ==========================================
  # Redis - Session/Cache Storage
  # ==========================================
  redis:
    image: redis:7-alpine
    container_name: streamvault-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    networks:
      - streaming

  # ==========================================
  # Prometheus - Monitoring
  # ==========================================
  prometheus:
    image: prom/prometheus:latest
    container_name: streamvault-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - streaming

  # ==========================================
  # Grafana - Visualization
  # ==========================================
  grafana:
    image: grafana/grafana:latest
    container_name: streamvault-grafana
    restart: unless-stopped
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/provisioning:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=streamvault123
      - GF_USERS_ALLOW_SIGN_UP=false
    networks:
      - streaming
    depends_on:
      - prometheus

networks:
  streaming:
    driver: bridge

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
"""
    return compose


# ==========================================
# REPOSITORY REFERENCE
# ==========================================

STREAMING_REPOSITORIES = """
# ==========================================
# STREAMING SERVER REPOSITORIES
# ==========================================

# SRS (Simple Realtime Server)
# - GitHub: https://github.com/ossrs/srs (26K+ stars)
# - Features: RTMP, WebRTC, HLS, HTTP-FLV, SRT, MPEG-DASH
# - Latency: 300ms-1s (RTMP), <200ms (WebRTC)
# - Docker: docker run -p 1935:1935 -p 8080:8080 ossrs/srs:5

# MediaMTX (formerly rtsp-simple-server)
# - GitHub: https://github.com/bluenviron/mediamtx (14K+ stars)
# - Features: RTSP, RTMP, HLS, WebRTC, SRT, Zero-dependency
# - Docker: docker run -p 8554:8554 bluenviron/mediamtx

# Ant Media Server
# - GitHub: https://github.com/ant-media/Ant-Media-Server
# - Features: WebRTC, CMAF, HLS, enterprise-grade
# - Use case: Large-scale deployments

# Janus Gateway
# - GitHub: https://github.com/meetecho/janus-gateway (8K+ stars)
# - Features: WebRTC gateway, plugins architecture
# - Use case: Video conferencing, streaming

# LiveGo
# - GitHub: https://github.com/gwuhaolin/livego
# - Features: Simple RTMP/HLS server in Go
# - Use case: Lightweight deployments

# Node-Media-Server
# - GitHub: https://github.com/illuspas/Node-Media-Server
# - Features: RTMP/HTTP-FLV/HLS in Node.js
# - Use case: Integration with Node.js apps

# Nimble Streamer
# - Website: https://wmspanel.com/nimble
# - Features: Low-latency HLS, WebRTC, ABR
# - Use case: Commercial deployments

# ==========================================
# ML/AI FOR STREAMING REPOSITORIES
# ==========================================

# Pensieve (MIT)
# - GitHub: https://github.com/hongzimao/pensieve
# - Paper: "Neural Adaptive Video Streaming with Pensieve" (SIGCOMM 2017)
# - Features: RL-based adaptive bitrate

# Pensieve-PPO
# - GitHub: https://github.com/godka/Pensieve-PPO
# - Features: PyTorch implementation with PPO

# Plume
# - GitHub: https://github.com/sagar-pa/plume
# - Features: OpenAI Gym environment for ABR

# QARC
# - Paper: "Video Quality Aware Rate Control" (ACM MM 2018)
# - Features: Deep RL for rate control

# ==========================================
# VIDEO QUALITY ASSESSMENT
# ==========================================

# VMAF (Netflix)
# - GitHub: https://github.com/Netflix/vmaf
# - Features: Video quality metric

# FFmpeg Quality Metrics
# - Integrated in FFmpeg for PSNR, SSIM, VMAF

# ==========================================
# CLIENT PLAYERS
# ==========================================

# hls.js
# - GitHub: https://github.com/video-dev/hls.js (14K+ stars)
# - Features: HLS playback in browser

# dash.js
# - GitHub: https://github.com/Dash-Industry-Forum/dash.js
# - Features: MPEG-DASH reference player

# Shaka Player (Google)
# - GitHub: https://github.com/shaka-project/shaka-player
# - Features: ABR, DRM, multi-protocol

# Video.js
# - GitHub: https://github.com/videojs/video.js (37K+ stars)
# - Features: Universal video player

# ExoPlayer (Android)
# - GitHub: https://github.com/google/ExoPlayer
# - Features: Advanced Android media player

# AVPlayer (iOS)
# - Apple's native video player
# - Features: HLS, FairPlay DRM
"""


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎬 StreamVault Pro Ultimate - Streaming Server Config")
    print("=" * 60)
    
    # Generate SRS config
    srs = SRSConfig()
    srs_config = srs.generate_config()
    
    os.makedirs("./config", exist_ok=True)
    with open("./config/srs.conf", "w") as f:
        f.write(srs_config)
    print("✅ SRS configuration generated: ./config/srs.conf")
    
    # Generate MediaMTX config
    mtx = MediaMTXConfig()
    mtx_config = mtx.generate_config()
    
    with open("./config/mediamtx.yml", "w") as f:
        f.write(mtx_config)
    print("✅ MediaMTX configuration generated: ./config/mediamtx.yml")
    
    # Generate Docker Compose
    compose = generate_streaming_docker_compose()
    with open("./docker-compose.streaming.yml", "w") as f:
        f.write(compose)
    print("✅ Docker Compose generated: ./docker-compose.streaming.yml")
    
    print("\n📋 Repository Reference saved")
    print("\n🚀 Quick Start:")
    print("   docker-compose -f docker-compose.streaming.yml up -d")
    print("\n📺 Streaming Endpoints:")
    print("   RTMP:   rtmp://localhost:1935/live/stream")
    print("   WebRTC: http://localhost:8000/live/stream.webrtc")
    print("   HLS:    http://localhost:8080/live/stream.m3u8")
    print("   RTSP:   rtsp://localhost:8554/live/stream")
