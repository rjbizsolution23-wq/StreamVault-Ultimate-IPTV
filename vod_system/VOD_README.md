# 🎬 StreamVault Pro - VOD (Video on Demand) System

## Complete VOD Platform Features

### ✅ What's Included

| Feature | Description |
|---------|-------------|
| **Multi-Quality Transcoding** | 480p, 720p, 1080p, 4K, 8K support |
| **HLS Adaptive Streaming** | Industry standard for iOS/Safari |
| **DASH Adaptive Streaming** | Cross-platform adaptive bitrate |
| **Hardware Acceleration** | NVIDIA NVENC, Intel QSV, AMD VCN |
| **Metadata Enrichment** | Auto-fetch from TMDB/IMDB |
| **Watch Progress Tracking** | Resume where you left off |
| **Continue Watching** | Smart "Continue Watching" lists |
| **Search & Discovery** | Full-text search, filters |
| **Content Categories** | Movies, Series, Sports, PPV |
| **User Favorites** | Personal watchlists |
| **View Analytics** | Trending content tracking |

---

## 🚀 Quick Start

### 1. Initialize VOD System

```python
from vod_engine import VODManager, ContentType

# Initialize
manager = VODManager(
    content_dir="/var/vod/content",
    transcode_dir="/var/vod/transcoded", 
    db_path="/var/vod/vod.db",
    tmdb_api_key="your_tmdb_api_key"  # Optional: for metadata
)
```

### 2. Ingest Content

```python
# Add a movie
content = manager.ingest_content(
    source_file="/path/to/movie.mp4",
    title="Movie Title",
    content_type=ContentType.MOVIE,
    auto_transcode=True
)

print(f"HLS URL: {content.hls_manifest}")
print(f"DASH URL: {content.dash_manifest}")
```

### 3. Start API Server

```bash
cd vod_system
pip install fastapi uvicorn
python -m uvicorn vod_api:app --host 0.0.0.0 --port 8002
```

---

## 📺 VOD API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/vod/content` | GET | List/search content |
| `/api/vod/content/{id}` | GET | Get single content |
| `/api/vod/trending` | GET | Trending content |
| `/api/vod/recent` | GET | Recently added |
| `/api/vod/movies` | GET | All movies |
| `/api/vod/series` | GET | All TV series |
| `/api/vod/sports` | GET | Sports content |
| `/api/vod/ppv` | GET | Pay Per View events |
| `/api/vod/continue-watching` | GET | Continue watching |
| `/api/vod/content/{id}/progress` | POST | Update watch progress |

---

## 🎥 Content Types Supported

- **Movies** - Feature films
- **TV Series** - Episodes organized by season
- **Documentaries** - Non-fiction content
- **Sports** - Sports events & highlights
- **PPV Events** - Pay-per-view content
- **Live Recordings** - Recorded live streams
- **Shorts** - Short-form content

---

## 🔧 Transcoding Profiles

| Quality | Resolution | Video Bitrate | Audio | Codec |
|---------|------------|---------------|-------|-------|
| **SD** | 854×480 | 1.5 Mbps | 128k | H.264 |
| **HD** | 1280×720 | 3 Mbps | 192k | H.264 |
| **FHD** | 1920×1080 | 6 Mbps | 256k | H.264 |
| **4K** | 3840×2160 | 15 Mbps | 384k | H.265 |

---

## 📊 Database Schema

```sql
-- Content table
content (
    id, title, content_type, description,
    poster_url, backdrop_url, duration_seconds,
    release_year, rating, genres, cast_members,
    hls_manifest, dash_manifest, is_premium, is_ppv...
)

-- Watch history
watch_history (
    user_id, content_id, progress_seconds,
    duration_seconds, last_watched, completed
)

-- Favorites
favorites (user_id, content_id, added_at)
```

---

## 🎯 PPV (Pay Per View) System

The VOD engine supports PPV events with:

- **Event Management**: Create/manage PPV events
- **Pricing**: Flexible per-event pricing
- **Access Control**: Token-based access
- **Time Windows**: Limited viewing windows

```python
# Create PPV event
ppv_event = VODContent(
    id="ppv_boxing_2026",
    title="Championship Boxing Match",
    content_type=ContentType.PPV_EVENT,
    is_ppv=True,
    ppv_price=49.99
)
```

---

## 📁 File Structure

```
vod_system/
├── vod_engine.py        # Main VOD engine
├── vod_api.py           # FastAPI server
├── VOD_README.md        # This file
├── content/             # Source content
├── transcoded/          # Transcoded output
│   └── {content_id}/
│       ├── hls/         # HLS segments
│       │   ├── master.m3u8
│       │   ├── 480p/
│       │   ├── 720p/
│       │   └── 1080p/
│       └── dash/        # DASH segments
│           └── manifest.mpd
└── vod.db              # SQLite database
```

---

## 🌐 Integration with StreamVault

The VOD system integrates seamlessly with the main StreamVault platform:

```yaml
# docker-compose.yml
services:
  vod-api:
    build: ./vod_system
    ports:
      - "8002:8002"
    volumes:
      - ./vod_content:/var/vod/content
      - ./vod_transcoded:/var/vod/transcoded
    environment:
      - TMDB_API_KEY=${TMDB_API_KEY}
```

---

## 📌 Legal Notice

This VOD system is designed for **legal content distribution**:

- ✅ Your own created content
- ✅ Licensed content you have rights to
- ✅ Public domain content
- ✅ Creative Commons content

**Do NOT use for pirated content.** Respect copyright laws.

---

*Part of StreamVault Pro Ultimate Platform*
