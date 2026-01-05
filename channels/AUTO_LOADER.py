#!/usr/bin/env python3
"""
StreamVault Pro Ultimate - Automatic Channel Loader
Downloads and merges channels from IPTV-ORG (10,000+ channels)
with our curated LIVE_CHANNELS_MASTER.m3u

Run: python3 AUTO_LOADER.py
"""

import requests
import os
from datetime import datetime
from pathlib import Path

# Configuration
IPTV_ORG_SOURCES = {
    "all_channels": "https://iptv-org.github.io/iptv/index.m3u",
    "by_category": "https://iptv-org.github.io/iptv/index.category.m3u",
    "by_country": "https://iptv-org.github.io/iptv/index.country.m3u",
    "us_only": "https://iptv-org.github.io/iptv/countries/us.m3u",
    "uk_only": "https://iptv-org.github.io/iptv/countries/uk.m3u",
    "canada": "https://iptv-org.github.io/iptv/countries/ca.m3u",
    "germany": "https://iptv-org.github.io/iptv/countries/de.m3u",
    "france": "https://iptv-org.github.io/iptv/countries/fr.m3u",
    "spain": "https://iptv-org.github.io/iptv/countries/es.m3u",
    "italy": "https://iptv-org.github.io/iptv/countries/it.m3u",
    "brazil": "https://iptv-org.github.io/iptv/countries/br.m3u",
    "mexico": "https://iptv-org.github.io/iptv/countries/mx.m3u",
    "india": "https://iptv-org.github.io/iptv/countries/in.m3u",
    "japan": "https://iptv-org.github.io/iptv/countries/jp.m3u",
}

FREE_TV_SOURCES = {
    "free_tv_all": "https://raw.githubusercontent.com/Free-TV/IPTV/master/playlist.m3u8",
}

OUTPUT_DIR = Path(__file__).parent / "downloaded"


def download_playlist(name: str, url: str) -> str:
    """Download a playlist and return content"""
    print(f"📥 Downloading {name}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        lines = content.count('\n')
        channels = content.count('#EXTINF')
        print(f"   ✅ Downloaded: {channels} channels, {lines} lines")
        return content
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return ""


def merge_playlists(playlists: dict) -> str:
    """Merge multiple playlists into one"""
    merged = "#EXTM3U\n"
    merged += f"# StreamVault Pro Ultimate - Auto-Merged Playlist\n"
    merged += f"# Generated: {datetime.now().isoformat()}\n\n"
    
    for name, content in playlists.items():
        if content:
            merged += f"\n# === SOURCE: {name} ===\n"
            # Remove #EXTM3U header from subsequent playlists
            content = content.replace("#EXTM3U", "").strip()
            merged += content + "\n"
    
    return merged


def count_channels(content: str) -> int:
    """Count channels in M3U content"""
    return content.count('#EXTINF')


def save_playlist(content: str, filename: str):
    """Save playlist to file"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    channels = count_channels(content)
    print(f"💾 Saved {filename}: {channels} channels ({filepath.stat().st_size / 1024:.1f} KB)")
    return filepath


def validate_stream(url: str, timeout: int = 5) -> bool:
    """Quick validation of a stream URL"""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except:
        return False


def main():
    print("=" * 60)
    print("🎬 StreamVault Pro Ultimate - Auto Channel Loader")
    print("=" * 60)
    print()
    
    # Download all IPTV-ORG playlists
    print("📡 Fetching IPTV-ORG playlists...")
    iptv_org_content = {}
    for name, url in IPTV_ORG_SOURCES.items():
        content = download_playlist(name, url)
        if content:
            iptv_org_content[name] = content
            save_playlist(content, f"iptv_org_{name}.m3u")
    
    print()
    print("📡 Fetching Free-TV playlists...")
    free_tv_content = {}
    for name, url in FREE_TV_SOURCES.items():
        content = download_playlist(name, url)
        if content:
            free_tv_content[name] = content
            save_playlist(content, f"free_tv_{name}.m3u")
    
    # Create merged master playlist
    print()
    print("🔀 Creating merged master playlist...")
    
    # Read our curated master list
    master_path = Path(__file__).parent / "LIVE_CHANNELS_MASTER.m3u"
    curated_content = ""
    if master_path.exists():
        with open(master_path, 'r', encoding='utf-8') as f:
            curated_content = f.read()
        print(f"   ✅ Loaded curated master: {count_channels(curated_content)} channels")
    
    all_playlists = {
        "StreamVault_Curated": curated_content,
        **iptv_org_content,
        **free_tv_content
    }
    
    merged = merge_playlists(all_playlists)
    merged_path = save_playlist(merged, "COMPLETE_MERGED_MASTER.m3u")
    
    # Summary
    print()
    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    total_channels = count_channels(merged)
    print(f"   Total channels available: {total_channels:,}")
    print(f"   Master playlist: {merged_path}")
    print()
    print("📁 Individual playlists saved to:")
    print(f"   {OUTPUT_DIR}")
    print()
    print("🎯 Quick Start URLs:")
    print(f"   • All channels: {IPTV_ORG_SOURCES['all_channels']}")
    print(f"   • By category: {IPTV_ORG_SOURCES['by_category']}")
    print(f"   • US only: {IPTV_ORG_SOURCES['us_only']}")
    print(f"   • UK only: {IPTV_ORG_SOURCES['uk_only']}")
    print()
    print("✅ Channel loading complete!")


if __name__ == "__main__":
    main()
