#!/usr/bin/env python3
"""
StreamVault Pro Ultimate - Advanced VOD (Video on Demand) Engine
Complete VOD system with content management, transcoding, and delivery

Features:
- Multi-resolution transcoding (4K, 1080p, 720p, 480p)
- Adaptive bitrate streaming (HLS/DASH)
- Content metadata management
- Watch history & progress tracking
- Recommendation engine
- Content categorization
- Search & discovery
- DRM integration ready
"""

import os
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# VOD DATA MODELS
# ============================================

class ContentType(Enum):
    MOVIE = "movie"
    TV_SERIES = "tv_series"
    TV_EPISODE = "tv_episode"
    DOCUMENTARY = "documentary"
    SPORTS = "sports"
    PPV_EVENT = "ppv_event"
    LIVE_RECORDING = "live_recording"
    SHORT = "short"


class VideoQuality(Enum):
    SD_480P = "480p"
    HD_720P = "720p"
    FHD_1080P = "1080p"
    UHD_4K = "2160p"
    UHD_8K = "4320p"


@dataclass
class VideoProfile:
    """Encoding profile for different qualities"""
    quality: VideoQuality
    width: int
    height: int
    bitrate_video: str
    bitrate_audio: str
    codec_video: str = "libx264"
    codec_audio: str = "aac"
    preset: str = "medium"
    crf: int = 23


@dataclass
class VODContent:
    """Video on Demand content item"""
    id: str
    title: str
    content_type: ContentType
    description: str = ""
    poster_url: str = ""
    backdrop_url: str = ""
    duration_seconds: int = 0
    release_year: int = 0
    rating: float = 0.0
    genres: List[str] = field(default_factory=list)
    cast: List[str] = field(default_factory=list)
    director: str = ""
    source_file: str = ""
    transcoded_files: Dict[str, str] = field(default_factory=dict)
    hls_manifest: str = ""
    dash_manifest: str = ""
    subtitles: Dict[str, str] = field(default_factory=dict)
    audio_tracks: List[str] = field(default_factory=list)
    is_premium: bool = False
    is_ppv: bool = False
    ppv_price: float = 0.0
    created_at: str = ""
    updated_at: str = ""
    view_count: int = 0
    imdb_id: str = ""
    tmdb_id: str = ""
    
    # TV Series specific
    season_number: int = 0
    episode_number: int = 0
    series_id: str = ""


@dataclass
class VODCategory:
    """Content category/genre"""
    id: str
    name: str
    slug: str
    description: str = ""
    poster_url: str = ""
    content_count: int = 0
    parent_id: str = ""


@dataclass
class UserWatchHistory:
    """User watch progress tracking"""
    user_id: str
    content_id: str
    progress_seconds: int
    duration_seconds: int
    last_watched: str
    completed: bool = False
    rating: float = 0.0


# ============================================
# ENCODING PROFILES
# ============================================

ENCODING_PROFILES = {
    VideoQuality.SD_480P: VideoProfile(
        quality=VideoQuality.SD_480P,
        width=854, height=480,
        bitrate_video="1500k", bitrate_audio="128k",
        crf=23, preset="medium"
    ),
    VideoQuality.HD_720P: VideoProfile(
        quality=VideoQuality.HD_720P,
        width=1280, height=720,
        bitrate_video="3000k", bitrate_audio="192k",
        crf=22, preset="medium"
    ),
    VideoQuality.FHD_1080P: VideoProfile(
        quality=VideoQuality.FHD_1080P,
        width=1920, height=1080,
        bitrate_video="6000k", bitrate_audio="256k",
        crf=21, preset="slow"
    ),
    VideoQuality.UHD_4K: VideoProfile(
        quality=VideoQuality.UHD_4K,
        width=3840, height=2160,
        bitrate_video="15000k", bitrate_audio="384k",
        codec_video="libx265", crf=20, preset="slow"
    ),
}


# ============================================
# VOD TRANSCODER
# ============================================

class VODTranscoder:
    """FFmpeg-based video transcoder for VOD content"""
    
    def __init__(self, output_dir: str = "/var/vod/transcoded"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def get_video_info(self, input_file: str) -> Dict:
        """Get video file information using ffprobe"""
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", input_file
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return {}
    
    def transcode_single(
        self, 
        input_file: str, 
        output_file: str, 
        profile: VideoProfile,
        hardware_accel: str = "auto"
    ) -> bool:
        """Transcode to a single quality profile"""
        
        # Build FFmpeg command
        cmd = ["ffmpeg", "-y", "-i", input_file]
        
        # Hardware acceleration
        if hardware_accel == "nvidia":
            cmd = ["ffmpeg", "-y", "-hwaccel", "cuda", "-i", input_file]
            profile.codec_video = "h264_nvenc"
        elif hardware_accel == "intel":
            cmd = ["ffmpeg", "-y", "-hwaccel", "qsv", "-i", input_file]
            profile.codec_video = "h264_qsv"
        
        # Video encoding
        cmd.extend([
            "-c:v", profile.codec_video,
            "-preset", profile.preset,
            "-crf", str(profile.crf),
            "-b:v", profile.bitrate_video,
            "-maxrate", profile.bitrate_video,
            "-bufsize", f"{int(profile.bitrate_video[:-1]) * 2}k",
            "-vf", f"scale={profile.width}:{profile.height}",
        ])
        
        # Audio encoding
        cmd.extend([
            "-c:a", profile.codec_audio,
            "-b:a", profile.bitrate_audio,
            "-ar", "48000",
        ])
        
        # Output
        cmd.append(output_file)
        
        try:
            logger.info(f"Transcoding to {profile.quality.value}: {output_file}")
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Transcoding failed: {e.stderr.decode()}")
            return False
    
    def create_hls_package(
        self, 
        input_file: str, 
        output_dir: str,
        qualities: List[VideoQuality] = None
    ) -> str:
        """Create HLS adaptive bitrate package"""
        
        if qualities is None:
            qualities = [VideoQuality.SD_480P, VideoQuality.HD_720P, VideoQuality.FHD_1080P]
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Build FFmpeg command for multi-quality HLS
        cmd = ["ffmpeg", "-y", "-i", input_file]
        
        # Add filter complex for multiple outputs
        filter_parts = []
        map_parts = []
        stream_parts = []
        
        for i, quality in enumerate(qualities):
            profile = ENCODING_PROFILES[quality]
            filter_parts.append(f"[0:v]scale={profile.width}:{profile.height}[v{i}]")
            map_parts.extend(["-map", f"[v{i}]", "-map", "0:a"])
            stream_parts.extend([
                f"-c:v:{i}", profile.codec_video,
                f"-b:v:{i}", profile.bitrate_video,
                f"-c:a:{i}", profile.codec_audio,
                f"-b:a:{i}", profile.bitrate_audio,
            ])
        
        cmd.extend(["-filter_complex", ";".join(filter_parts)])
        cmd.extend(map_parts)
        cmd.extend(stream_parts)
        
        # HLS settings
        cmd.extend([
            "-f", "hls",
            "-hls_time", "6",
            "-hls_playlist_type", "vod",
            "-hls_segment_filename", f"{out_path}/%v/segment_%03d.ts",
            "-master_pl_name", "master.m3u8",
            "-var_stream_map", " ".join([f"v:{i},a:{i}" for i in range(len(qualities))]),
            f"{out_path}/%v/playlist.m3u8"
        ])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return str(out_path / "master.m3u8")
        except subprocess.CalledProcessError as e:
            logger.error(f"HLS packaging failed: {e.stderr.decode()}")
            return ""
    
    def create_dash_package(
        self, 
        input_file: str, 
        output_dir: str,
        qualities: List[VideoQuality] = None
    ) -> str:
        """Create DASH adaptive bitrate package"""
        
        if qualities is None:
            qualities = [VideoQuality.SD_480P, VideoQuality.HD_720P, VideoQuality.FHD_1080P]
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # First create fragmented MP4 files
        temp_files = []
        for quality in qualities:
            profile = ENCODING_PROFILES[quality]
            temp_file = out_path / f"temp_{quality.value}.mp4"
            
            cmd = [
                "ffmpeg", "-y", "-i", input_file,
                "-c:v", profile.codec_video,
                "-b:v", profile.bitrate_video,
                "-vf", f"scale={profile.width}:{profile.height}",
                "-c:a", profile.codec_audio,
                "-b:a", profile.bitrate_audio,
                "-movflags", "+faststart",
                str(temp_file)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            temp_files.append(str(temp_file))
        
        # Create DASH manifest using MP4Box or ffmpeg
        manifest_path = out_path / "manifest.mpd"
        cmd = [
            "ffmpeg", "-y",
        ]
        for tf in temp_files:
            cmd.extend(["-i", tf])
        
        cmd.extend([
            "-c", "copy",
            "-f", "dash",
            "-seg_duration", "6",
            "-use_timeline", "1",
            "-use_template", "1",
            "-adaptation_sets", "id=0,streams=v id=1,streams=a",
            str(manifest_path)
        ])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            # Clean up temp files
            for tf in temp_files:
                Path(tf).unlink(missing_ok=True)
            return str(manifest_path)
        except Exception as e:
            logger.error(f"DASH packaging failed: {e}")
            return ""


# ============================================
# VOD DATABASE
# ============================================

class VODDatabase:
    """SQLite database for VOD content management"""
    
    def __init__(self, db_path: str = "/var/vod/vod.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS content (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    description TEXT,
                    poster_url TEXT,
                    backdrop_url TEXT,
                    duration_seconds INTEGER DEFAULT 0,
                    release_year INTEGER,
                    rating REAL DEFAULT 0,
                    genres TEXT,
                    cast_members TEXT,
                    director TEXT,
                    source_file TEXT,
                    transcoded_files TEXT,
                    hls_manifest TEXT,
                    dash_manifest TEXT,
                    subtitles TEXT,
                    is_premium INTEGER DEFAULT 0,
                    is_ppv INTEGER DEFAULT 0,
                    ppv_price REAL DEFAULT 0,
                    view_count INTEGER DEFAULT 0,
                    imdb_id TEXT,
                    tmdb_id TEXT,
                    season_number INTEGER DEFAULT 0,
                    episode_number INTEGER DEFAULT 0,
                    series_id TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                
                CREATE TABLE IF NOT EXISTS categories (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    slug TEXT UNIQUE,
                    description TEXT,
                    poster_url TEXT,
                    parent_id TEXT
                );
                
                CREATE TABLE IF NOT EXISTS content_categories (
                    content_id TEXT,
                    category_id TEXT,
                    PRIMARY KEY (content_id, category_id)
                );
                
                CREATE TABLE IF NOT EXISTS watch_history (
                    user_id TEXT,
                    content_id TEXT,
                    progress_seconds INTEGER DEFAULT 0,
                    duration_seconds INTEGER,
                    last_watched TEXT,
                    completed INTEGER DEFAULT 0,
                    rating REAL DEFAULT 0,
                    PRIMARY KEY (user_id, content_id)
                );
                
                CREATE TABLE IF NOT EXISTS favorites (
                    user_id TEXT,
                    content_id TEXT,
                    added_at TEXT,
                    PRIMARY KEY (user_id, content_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_content_type ON content(content_type);
                CREATE INDEX IF NOT EXISTS idx_content_genre ON content(genres);
                CREATE INDEX IF NOT EXISTS idx_content_year ON content(release_year);
                CREATE INDEX IF NOT EXISTS idx_watch_user ON watch_history(user_id);
            """)
    
    def add_content(self, content: VODContent) -> bool:
        """Add or update VOD content"""
        content.updated_at = datetime.now().isoformat()
        if not content.created_at:
            content.created_at = content.updated_at
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO content 
                (id, title, content_type, description, poster_url, backdrop_url,
                 duration_seconds, release_year, rating, genres, cast_members,
                 director, source_file, transcoded_files, hls_manifest, dash_manifest,
                 subtitles, is_premium, is_ppv, ppv_price, view_count, imdb_id, tmdb_id,
                 season_number, episode_number, series_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content.id, content.title, content.content_type.value,
                content.description, content.poster_url, content.backdrop_url,
                content.duration_seconds, content.release_year, content.rating,
                json.dumps(content.genres), json.dumps(content.cast),
                content.director, content.source_file,
                json.dumps(content.transcoded_files), content.hls_manifest,
                content.dash_manifest, json.dumps(content.subtitles),
                int(content.is_premium), int(content.is_ppv), content.ppv_price,
                content.view_count, content.imdb_id, content.tmdb_id,
                content.season_number, content.episode_number, content.series_id,
                content.created_at, content.updated_at
            ))
            return True
    
    def get_content(self, content_id: str) -> Optional[VODContent]:
        """Get content by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM content WHERE id = ?", (content_id,)
            ).fetchone()
            
            if row:
                return self._row_to_content(dict(row))
        return None
    
    def search_content(
        self, 
        query: str = "",
        content_type: ContentType = None,
        genre: str = None,
        year_from: int = None,
        year_to: int = None,
        is_premium: bool = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[VODContent]:
        """Search and filter content"""
        sql = "SELECT * FROM content WHERE 1=1"
        params = []
        
        if query:
            sql += " AND (title LIKE ? OR description LIKE ?)"
            params.extend([f"%{query}%", f"%{query}%"])
        
        if content_type:
            sql += " AND content_type = ?"
            params.append(content_type.value)
        
        if genre:
            sql += " AND genres LIKE ?"
            params.append(f'%"{genre}"%')
        
        if year_from:
            sql += " AND release_year >= ?"
            params.append(year_from)
        
        if year_to:
            sql += " AND release_year <= ?"
            params.append(year_to)
        
        if is_premium is not None:
            sql += " AND is_premium = ?"
            params.append(int(is_premium))
        
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_content(dict(row)) for row in rows]
    
    def get_trending(self, limit: int = 20) -> List[VODContent]:
        """Get trending content by view count"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM content ORDER BY view_count DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [self._row_to_content(dict(row)) for row in rows]
    
    def get_recently_added(self, limit: int = 20) -> List[VODContent]:
        """Get recently added content"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM content ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [self._row_to_content(dict(row)) for row in rows]
    
    def increment_view(self, content_id: str):
        """Increment view count"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE content SET view_count = view_count + 1 WHERE id = ?",
                (content_id,)
            )
    
    def _row_to_content(self, row: dict) -> VODContent:
        """Convert database row to VODContent"""
        return VODContent(
            id=row["id"],
            title=row["title"],
            content_type=ContentType(row["content_type"]),
            description=row.get("description", ""),
            poster_url=row.get("poster_url", ""),
            backdrop_url=row.get("backdrop_url", ""),
            duration_seconds=row.get("duration_seconds", 0),
            release_year=row.get("release_year", 0),
            rating=row.get("rating", 0),
            genres=json.loads(row.get("genres", "[]")),
            cast=json.loads(row.get("cast_members", "[]")),
            director=row.get("director", ""),
            source_file=row.get("source_file", ""),
            transcoded_files=json.loads(row.get("transcoded_files", "{}")),
            hls_manifest=row.get("hls_manifest", ""),
            dash_manifest=row.get("dash_manifest", ""),
            subtitles=json.loads(row.get("subtitles", "{}")),
            is_premium=bool(row.get("is_premium", 0)),
            is_ppv=bool(row.get("is_ppv", 0)),
            ppv_price=row.get("ppv_price", 0),
            view_count=row.get("view_count", 0),
            imdb_id=row.get("imdb_id", ""),
            tmdb_id=row.get("tmdb_id", ""),
            season_number=row.get("season_number", 0),
            episode_number=row.get("episode_number", 0),
            series_id=row.get("series_id", ""),
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", "")
        )
    
    # Watch History
    def update_watch_progress(
        self, 
        user_id: str, 
        content_id: str, 
        progress_seconds: int,
        duration_seconds: int
    ):
        """Update user watch progress"""
        completed = progress_seconds >= (duration_seconds * 0.9)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO watch_history 
                (user_id, content_id, progress_seconds, duration_seconds, last_watched, completed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id, content_id, progress_seconds, duration_seconds,
                datetime.now().isoformat(), int(completed)
            ))
    
    def get_continue_watching(self, user_id: str, limit: int = 10) -> List[dict]:
        """Get user's continue watching list"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT c.*, w.progress_seconds, w.last_watched 
                FROM content c
                JOIN watch_history w ON c.id = w.content_id
                WHERE w.user_id = ? AND w.completed = 0
                ORDER BY w.last_watched DESC
                LIMIT ?
            """, (user_id, limit)).fetchall()
            
            return [
                {
                    "content": self._row_to_content(dict(row)),
                    "progress_seconds": row["progress_seconds"],
                    "last_watched": row["last_watched"]
                }
                for row in rows
            ]


# ============================================
# VOD API SERVER
# ============================================

class VODAPIServer:
    """FastAPI-based VOD API Server"""
    
    API_CODE = '''
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="StreamVault VOD API",
    description="Video on Demand API for StreamVault Pro",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = VODDatabase()
transcoder = VODTranscoder()

# Pydantic models
class ContentCreate(BaseModel):
    title: str
    content_type: str
    description: str = ""
    poster_url: str = ""
    release_year: int = 0
    genres: List[str] = []
    source_file: str = ""

class WatchProgress(BaseModel):
    progress_seconds: int
    duration_seconds: int

# Endpoints
@app.get("/api/vod/content")
async def list_content(
    query: str = "",
    content_type: str = None,
    genre: str = None,
    year_from: int = None,
    year_to: int = None,
    limit: int = 50,
    offset: int = 0
):
    """List and search VOD content"""
    ct = ContentType(content_type) if content_type else None
    return db.search_content(query, ct, genre, year_from, year_to, limit=limit, offset=offset)

@app.get("/api/vod/content/{content_id}")
async def get_content(content_id: str):
    """Get single content item"""
    content = db.get_content(content_id)
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    db.increment_view(content_id)
    return content

@app.get("/api/vod/trending")
async def get_trending(limit: int = 20):
    """Get trending content"""
    return db.get_trending(limit)

@app.get("/api/vod/recent")
async def get_recent(limit: int = 20):
    """Get recently added content"""
    return db.get_recently_added(limit)

@app.get("/api/vod/movies")
async def get_movies(limit: int = 50, offset: int = 0):
    """Get all movies"""
    return db.search_content(content_type=ContentType.MOVIE, limit=limit, offset=offset)

@app.get("/api/vod/series")
async def get_series(limit: int = 50, offset: int = 0):
    """Get all TV series"""
    return db.search_content(content_type=ContentType.TV_SERIES, limit=limit, offset=offset)

@app.get("/api/vod/sports")
async def get_sports(limit: int = 50, offset: int = 0):
    """Get sports content"""
    return db.search_content(content_type=ContentType.SPORTS, limit=limit, offset=offset)

@app.get("/api/vod/ppv")
async def get_ppv(limit: int = 50, offset: int = 0):
    """Get PPV events"""
    return db.search_content(is_premium=True, limit=limit, offset=offset)

@app.post("/api/vod/content/{content_id}/progress")
async def update_progress(content_id: str, progress: WatchProgress, user_id: str = "default"):
    """Update watch progress"""
    db.update_watch_progress(user_id, content_id, progress.progress_seconds, progress.duration_seconds)
    return {"status": "success"}

@app.get("/api/vod/continue-watching")
async def continue_watching(user_id: str = "default", limit: int = 10):
    """Get continue watching list"""
    return db.get_continue_watching(user_id, limit)

@app.get("/api/vod/categories")
async def list_categories():
    """List all categories"""
    return [
        {"id": "movies", "name": "Movies", "slug": "movies"},
        {"id": "series", "name": "TV Series", "slug": "series"},
        {"id": "sports", "name": "Sports", "slug": "sports"},
        {"id": "documentaries", "name": "Documentaries", "slug": "documentaries"},
        {"id": "ppv", "name": "Pay Per View", "slug": "ppv"},
        {"id": "kids", "name": "Kids", "slug": "kids"},
    ]

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
'''


# ============================================
# CONTENT SCRAPER / METADATA
# ============================================

class ContentMetadataScraper:
    """Scrape metadata from TMDB, IMDB, etc."""
    
    def __init__(self, tmdb_api_key: str = ""):
        self.tmdb_api_key = tmdb_api_key
        self.tmdb_base = "https://api.themoviedb.org/3"
    
    def search_movie(self, title: str, year: int = None) -> Optional[dict]:
        """Search TMDB for movie"""
        import requests
        
        params = {
            "api_key": self.tmdb_api_key,
            "query": title,
        }
        if year:
            params["year"] = year
        
        try:
            response = requests.get(f"{self.tmdb_base}/search/movie", params=params)
            data = response.json()
            if data.get("results"):
                return data["results"][0]
        except Exception as e:
            logger.error(f"TMDB search failed: {e}")
        return None
    
    def get_movie_details(self, tmdb_id: int) -> Optional[dict]:
        """Get full movie details from TMDB"""
        import requests
        
        try:
            response = requests.get(
                f"{self.tmdb_base}/movie/{tmdb_id}",
                params={
                    "api_key": self.tmdb_api_key,
                    "append_to_response": "credits,videos"
                }
            )
            return response.json()
        except Exception as e:
            logger.error(f"TMDB details failed: {e}")
        return None
    
    def enrich_content(self, content: VODContent) -> VODContent:
        """Enrich content with TMDB metadata"""
        if not self.tmdb_api_key:
            return content
        
        movie = self.search_movie(content.title, content.release_year)
        if movie:
            content.tmdb_id = str(movie.get("id", ""))
            content.poster_url = f"https://image.tmdb.org/t/p/w500{movie.get('poster_path', '')}"
            content.backdrop_url = f"https://image.tmdb.org/t/p/original{movie.get('backdrop_path', '')}"
            content.rating = movie.get("vote_average", 0)
            content.description = movie.get("overview", content.description)
            
            # Get full details
            details = self.get_movie_details(movie["id"])
            if details:
                content.genres = [g["name"] for g in details.get("genres", [])]
                credits = details.get("credits", {})
                content.cast = [c["name"] for c in credits.get("cast", [])[:10]]
                directors = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
                content.director = directors[0] if directors else ""
                content.duration_seconds = details.get("runtime", 0) * 60
        
        return content


# ============================================
# MAIN VOD MANAGER
# ============================================

class VODManager:
    """Main VOD system manager"""
    
    def __init__(
        self,
        content_dir: str = "/var/vod/content",
        transcode_dir: str = "/var/vod/transcoded",
        db_path: str = "/var/vod/vod.db",
        tmdb_api_key: str = ""
    ):
        self.content_dir = Path(content_dir)
        self.transcode_dir = Path(transcode_dir)
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.transcode_dir.mkdir(parents=True, exist_ok=True)
        
        self.db = VODDatabase(db_path)
        self.transcoder = VODTranscoder(transcode_dir)
        self.metadata = ContentMetadataScraper(tmdb_api_key)
    
    def ingest_content(
        self,
        source_file: str,
        title: str,
        content_type: ContentType = ContentType.MOVIE,
        auto_transcode: bool = True,
        qualities: List[VideoQuality] = None
    ) -> VODContent:
        """Ingest new content into the VOD system"""
        
        # Generate content ID
        content_id = hashlib.md5(f"{title}{source_file}".encode()).hexdigest()[:12]
        
        # Get video info
        video_info = self.transcoder.get_video_info(source_file)
        duration = int(float(video_info.get("format", {}).get("duration", 0)))
        
        # Create content object
        content = VODContent(
            id=content_id,
            title=title,
            content_type=content_type,
            source_file=source_file,
            duration_seconds=duration
        )
        
        # Enrich with metadata
        content = self.metadata.enrich_content(content)
        
        # Transcode if requested
        if auto_transcode:
            output_dir = self.transcode_dir / content_id
            
            # Create HLS package
            hls_manifest = self.transcoder.create_hls_package(
                source_file, 
                str(output_dir / "hls"),
                qualities or [VideoQuality.SD_480P, VideoQuality.HD_720P, VideoQuality.FHD_1080P]
            )
            content.hls_manifest = hls_manifest
            
            # Create DASH package
            dash_manifest = self.transcoder.create_dash_package(
                source_file,
                str(output_dir / "dash"),
                qualities or [VideoQuality.SD_480P, VideoQuality.HD_720P, VideoQuality.FHD_1080P]
            )
            content.dash_manifest = dash_manifest
        
        # Save to database
        self.db.add_content(content)
        
        logger.info(f"Ingested content: {content.title} (ID: {content.id})")
        return content
    
    def get_playback_url(self, content_id: str, format: str = "hls") -> str:
        """Get playback URL for content"""
        content = self.db.get_content(content_id)
        if not content:
            return ""
        
        if format == "hls":
            return content.hls_manifest
        elif format == "dash":
            return content.dash_manifest
        elif format == "source":
            return content.source_file
        
        return content.hls_manifest


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎬 StreamVault Pro - VOD System")
    print("=" * 60)
    
    # Initialize manager
    manager = VODManager(
        content_dir="./vod_content",
        transcode_dir="./vod_transcoded",
        db_path="./vod.db"
    )
    
    print("\n✅ VOD System initialized!")
    print("\nAvailable features:")
    print("  • Multi-quality transcoding (480p, 720p, 1080p, 4K)")
    print("  • HLS adaptive streaming")
    print("  • DASH adaptive streaming")
    print("  • TMDB metadata enrichment")
    print("  • Watch history tracking")
    print("  • Content categorization")
    print("  • Search & discovery")
    print("\nTo ingest content:")
    print("  manager.ingest_content('movie.mp4', 'Movie Title')")
    print("\nTo start API server:")
    print("  python -c 'exec(VODAPIServer.API_CODE)'")
