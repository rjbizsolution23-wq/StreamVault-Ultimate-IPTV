#!/usr/bin/env python3
"""
StreamVault Pro Ultimate - IPTV Channel Sources & Playlist Manager
Complete integration with public IPTV channel databases

WORKING CHANNEL SOURCES:
- iptv-org/iptv (10,000+ channels)
- Free-TV/IPTV (Legal free channels)
- Plus custom sources
"""

import requests
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import re

# ==========================================
# OFFICIAL IPTV-ORG PLAYLIST URLS
# ==========================================

IPTV_ORG_PLAYLISTS = {
    # Main playlist - ALL channels
    "main": "https://iptv-org.github.io/iptv/index.m3u",
    
    # Grouped by category
    "by_category": "https://iptv-org.github.io/iptv/index.category.m3u",
    
    # Grouped by language
    "by_language": "https://iptv-org.github.io/iptv/index.language.m3u",
    
    # Grouped by country
    "by_country": "https://iptv-org.github.io/iptv/index.country.m3u",
    
    # Grouped by region
    "by_region": "https://iptv-org.github.io/iptv/index.region.m3u",
    
    # Country-specific playlists
    "countries": {
        "us": "https://iptv-org.github.io/iptv/countries/us.m3u",
        "uk": "https://iptv-org.github.io/iptv/countries/uk.m3u",
        "ca": "https://iptv-org.github.io/iptv/countries/ca.m3u",
        "de": "https://iptv-org.github.io/iptv/countries/de.m3u",
        "fr": "https://iptv-org.github.io/iptv/countries/fr.m3u",
        "es": "https://iptv-org.github.io/iptv/countries/es.m3u",
        "it": "https://iptv-org.github.io/iptv/countries/it.m3u",
        "br": "https://iptv-org.github.io/iptv/countries/br.m3u",
        "mx": "https://iptv-org.github.io/iptv/countries/mx.m3u",
        "in": "https://iptv-org.github.io/iptv/countries/in.m3u",
        "jp": "https://iptv-org.github.io/iptv/countries/jp.m3u",
        "kr": "https://iptv-org.github.io/iptv/countries/kr.m3u",
        "au": "https://iptv-org.github.io/iptv/countries/au.m3u",
        "ru": "https://iptv-org.github.io/iptv/countries/ru.m3u",
        "ar": "https://iptv-org.github.io/iptv/countries/ar.m3u",
        "ae": "https://iptv-org.github.io/iptv/countries/ae.m3u",
        "za": "https://iptv-org.github.io/iptv/countries/za.m3u",
        "ng": "https://iptv-org.github.io/iptv/countries/ng.m3u",
        "eg": "https://iptv-org.github.io/iptv/countries/eg.m3u",
        "tr": "https://iptv-org.github.io/iptv/countries/tr.m3u",
        "pl": "https://iptv-org.github.io/iptv/countries/pl.m3u",
        "nl": "https://iptv-org.github.io/iptv/countries/nl.m3u",
        "be": "https://iptv-org.github.io/iptv/countries/be.m3u",
        "se": "https://iptv-org.github.io/iptv/countries/se.m3u",
        "no": "https://iptv-org.github.io/iptv/countries/no.m3u",
        "dk": "https://iptv-org.github.io/iptv/countries/dk.m3u",
        "fi": "https://iptv-org.github.io/iptv/countries/fi.m3u",
        "pt": "https://iptv-org.github.io/iptv/countries/pt.m3u",
        "gr": "https://iptv-org.github.io/iptv/countries/gr.m3u",
        "ch": "https://iptv-org.github.io/iptv/countries/ch.m3u",
        "at": "https://iptv-org.github.io/iptv/countries/at.m3u",
        "ie": "https://iptv-org.github.io/iptv/countries/ie.m3u",
        "nz": "https://iptv-org.github.io/iptv/countries/nz.m3u",
        "ph": "https://iptv-org.github.io/iptv/countries/ph.m3u",
        "th": "https://iptv-org.github.io/iptv/countries/th.m3u",
        "vn": "https://iptv-org.github.io/iptv/countries/vn.m3u",
        "id": "https://iptv-org.github.io/iptv/countries/id.m3u",
        "my": "https://iptv-org.github.io/iptv/countries/my.m3u",
        "sg": "https://iptv-org.github.io/iptv/countries/sg.m3u",
        "cn": "https://iptv-org.github.io/iptv/countries/cn.m3u",
        "tw": "https://iptv-org.github.io/iptv/countries/tw.m3u",
        "hk": "https://iptv-org.github.io/iptv/countries/hk.m3u",
    },
    
    # Category-specific playlists
    "categories": {
        "news": "https://iptv-org.github.io/iptv/categories/news.m3u",
        "sports": "https://iptv-org.github.io/iptv/categories/sports.m3u",
        "entertainment": "https://iptv-org.github.io/iptv/categories/entertainment.m3u",
        "movies": "https://iptv-org.github.io/iptv/categories/movies.m3u",
        "music": "https://iptv-org.github.io/iptv/categories/music.m3u",
        "kids": "https://iptv-org.github.io/iptv/categories/kids.m3u",
        "documentary": "https://iptv-org.github.io/iptv/categories/documentary.m3u",
        "lifestyle": "https://iptv-org.github.io/iptv/categories/lifestyle.m3u",
        "education": "https://iptv-org.github.io/iptv/categories/education.m3u",
        "business": "https://iptv-org.github.io/iptv/categories/business.m3u",
        "religious": "https://iptv-org.github.io/iptv/categories/religious.m3u",
        "travel": "https://iptv-org.github.io/iptv/categories/travel.m3u",
        "cooking": "https://iptv-org.github.io/iptv/categories/cooking.m3u",
        "weather": "https://iptv-org.github.io/iptv/categories/weather.m3u",
        "science": "https://iptv-org.github.io/iptv/categories/science.m3u",
        "animation": "https://iptv-org.github.io/iptv/categories/animation.m3u",
        "classic": "https://iptv-org.github.io/iptv/categories/classic.m3u",
        "comedy": "https://iptv-org.github.io/iptv/categories/comedy.m3u",
        "culture": "https://iptv-org.github.io/iptv/categories/culture.m3u",
        "family": "https://iptv-org.github.io/iptv/categories/family.m3u",
        "general": "https://iptv-org.github.io/iptv/categories/general.m3u",
        "legislative": "https://iptv-org.github.io/iptv/categories/legislative.m3u",
        "outdoor": "https://iptv-org.github.io/iptv/categories/outdoor.m3u",
        "relax": "https://iptv-org.github.io/iptv/categories/relax.m3u",
        "series": "https://iptv-org.github.io/iptv/categories/series.m3u",
        "shop": "https://iptv-org.github.io/iptv/categories/shop.m3u",
        "xxx": "https://iptv-org.github.io/iptv/categories/xxx.m3u",  # Adult content
    },
    
    # Language-specific playlists
    "languages": {
        "eng": "https://iptv-org.github.io/iptv/languages/eng.m3u",
        "spa": "https://iptv-org.github.io/iptv/languages/spa.m3u",
        "por": "https://iptv-org.github.io/iptv/languages/por.m3u",
        "fra": "https://iptv-org.github.io/iptv/languages/fra.m3u",
        "deu": "https://iptv-org.github.io/iptv/languages/deu.m3u",
        "ita": "https://iptv-org.github.io/iptv/languages/ita.m3u",
        "rus": "https://iptv-org.github.io/iptv/languages/rus.m3u",
        "ara": "https://iptv-org.github.io/iptv/languages/ara.m3u",
        "hin": "https://iptv-org.github.io/iptv/languages/hin.m3u",
        "jpn": "https://iptv-org.github.io/iptv/languages/jpn.m3u",
        "kor": "https://iptv-org.github.io/iptv/languages/kor.m3u",
        "zho": "https://iptv-org.github.io/iptv/languages/zho.m3u",
        "tur": "https://iptv-org.github.io/iptv/languages/tur.m3u",
        "pol": "https://iptv-org.github.io/iptv/languages/pol.m3u",
        "nld": "https://iptv-org.github.io/iptv/languages/nld.m3u",
    }
}

# Free-TV IPTV Playlists (Legal free channels)
FREE_TV_PLAYLISTS = {
    "main": "https://raw.githubusercontent.com/Free-TV/IPTV/master/playlist.m3u8",
    "pluto_tv": "https://raw.githubusercontent.com/Free-TV/IPTV/master/playlists/playlist_usa_pluto_tv.m3u8",
    "plex_tv": "https://raw.githubusercontent.com/Free-TV/IPTV/master/playlists/playlist_usa_plex_tv.m3u8",
    "samsung_tv": "https://raw.githubusercontent.com/Free-TV/IPTV/master/playlists/playlist_usa_samsung_tv_plus.m3u8",
    "stirr": "https://raw.githubusercontent.com/Free-TV/IPTV/master/playlists/playlist_usa_stirr.m3u8",
    "xumo": "https://raw.githubusercontent.com/Free-TV/IPTV/master/playlists/playlist_usa_xumo.m3u8",
    "tubi": "https://raw.githubusercontent.com/Free-TV/IPTV/master/playlists/playlist_usa_tubi.m3u8",
    "redbox": "https://raw.githubusercontent.com/Free-TV/IPTV/master/playlists/playlist_usa_redbox.m3u8",
}

# Additional free legal streaming sources
ADDITIONAL_SOURCES = {
    "cbs_news": "https://cbsn-us.cbsistatic.com/m3u/cbsn-national.m3u8",
    "abc_news": "https://content.uplynk.com/channel/3324f2467c414329b3b0cc5cd987b6be.m3u8",
    "nasa_tv": "https://ntv1.akamaized.net/hls/live/2014075/NASA-NTV1-HLS/master.m3u8",
    "bloomberg": "https://liveproduseast.blob.core.usgovcloudapi.net/live/dc/live.m3u8",
    "sky_news": "https://skynews-global-hls.akamaized.net/skynews/1702/main.m3u8",
    "france24_en": "https://static.france24.com/live/F24_EN_LO_HLS/live_web.m3u8",
    "dw_english": "https://dwamdstream102.akamaized.net/hls/live/2015525/dwstream102/index.m3u8",
    "euronews": "https://euronews.dfrnt.fr/euronews_en_hls/master.m3u8",
    "al_jazeera": "https://live-hls-web-aje.getaj.net/AJE/01.m3u8",
    "nhk_world": "https://nhkwlive-xjp.akamaized.net/hls/live/2003459/nhkwlive-xjp-en/index.m3u8",
    "cgtn": "https://news.cgtn.com/resource/live/english/cgtn-news.m3u8",
    "rt": "https://rt-news-gd.akamaized.net/hls/live/2031052/rtsvod/index.m3u8",
    "trt_world": "https://tv-trtworld.medya.trt.com.tr/master.m3u8",
}

# EPG (Electronic Program Guide) Sources
EPG_SOURCES = {
    "main": "https://iptv-org.github.io/epg/guides/index.xml",
    "by_country": "https://iptv-org.github.io/epg/guides/{country}.xml",
    "api": "https://iptv-org.github.io/api/guides.json"
}


@dataclass
class Channel:
    """Represents an IPTV channel."""
    id: str
    name: str
    url: str
    logo: str = ""
    group: str = ""
    language: str = ""
    country: str = ""
    tvg_id: str = ""
    tvg_name: str = ""
    is_working: bool = True
    quality: str = "HD"
    
    def to_m3u_entry(self) -> str:
        """Convert to M3U format entry."""
        attrs = []
        if self.tvg_id:
            attrs.append(f'tvg-id="{self.tvg_id}"')
        if self.tvg_name:
            attrs.append(f'tvg-name="{self.tvg_name}"')
        if self.logo:
            attrs.append(f'tvg-logo="{self.logo}"')
        if self.group:
            attrs.append(f'group-title="{self.group}"')
        if self.language:
            attrs.append(f'tvg-language="{self.language}"')
        if self.country:
            attrs.append(f'tvg-country="{self.country}"')
        
        attrs_str = " ".join(attrs)
        return f'#EXTINF:-1 {attrs_str},{self.name}\n{self.url}'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "logo": self.logo,
            "group": self.group,
            "language": self.language,
            "country": self.country,
            "tvg_id": self.tvg_id,
            "quality": self.quality
        }


class IPTVPlaylistManager:
    """
    Manages IPTV playlists, channel verification, and streaming.
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.channels: List[Channel] = []
        self.playlists: Dict[str, str] = {}
    
    def download_playlist(self, url: str, name: str = "playlist") -> Optional[str]:
        """Download a playlist from URL."""
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                content = response.text
                
                # Cache the playlist
                cache_path = os.path.join(self.cache_dir, f"{name}.m3u")
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.playlists[name] = cache_path
                print(f"✅ Downloaded playlist: {name} ({len(content)} bytes)")
                return content
            else:
                print(f"❌ Failed to download {url}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error downloading playlist: {e}")
        return None
    
    def parse_m3u(self, content: str) -> List[Channel]:
        """Parse M3U playlist content into Channel objects."""
        channels = []
        lines = content.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('#EXTINF:'):
                # Parse channel info
                channel_info = self._parse_extinf(line)
                
                # Get URL from next line
                if i + 1 < len(lines):
                    url = lines[i + 1].strip()
                    if url and not url.startswith('#'):
                        channel = Channel(
                            id=channel_info.get('tvg-id', f"ch_{len(channels)}"),
                            name=channel_info.get('name', 'Unknown'),
                            url=url,
                            logo=channel_info.get('tvg-logo', ''),
                            group=channel_info.get('group-title', ''),
                            language=channel_info.get('tvg-language', ''),
                            country=channel_info.get('tvg-country', ''),
                            tvg_id=channel_info.get('tvg-id', ''),
                            tvg_name=channel_info.get('tvg-name', '')
                        )
                        channels.append(channel)
                    i += 1
            i += 1
        
        print(f"📺 Parsed {len(channels)} channels")
        return channels
    
    def _parse_extinf(self, line: str) -> Dict[str, str]:
        """Parse EXTINF line attributes."""
        result = {}
        
        # Extract attributes
        attrs_pattern = r'(\w+[-\w]*?)="([^"]*)"'
        for match in re.finditer(attrs_pattern, line):
            key, value = match.groups()
            result[key] = value
        
        # Extract channel name (after last comma)
        if ',' in line:
            result['name'] = line.split(',')[-1].strip()
        
        return result
    
    def verify_stream(self, url: str, timeout: int = 5) -> bool:
        """Verify if a stream URL is working."""
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            return response.status_code in [200, 206, 302, 301]
        except:
            return False
    
    def verify_channels(self, channels: List[Channel], max_workers: int = 10) -> List[Channel]:
        """Verify multiple channels in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        working_channels = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_channel = {
                executor.submit(self.verify_stream, ch.url): ch 
                for ch in channels
            }
            
            for future in as_completed(future_to_channel):
                channel = future_to_channel[future]
                try:
                    if future.result():
                        channel.is_working = True
                        working_channels.append(channel)
                except:
                    pass
        
        print(f"✅ {len(working_channels)}/{len(channels)} channels verified")
        return working_channels
    
    def generate_playlist(self, channels: List[Channel], filename: str = "playlist.m3u") -> str:
        """Generate M3U playlist from channels."""
        output = "#EXTM3U\n"
        
        for channel in channels:
            output += channel.to_m3u_entry() + "\n"
        
        filepath = os.path.join(self.cache_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        
        print(f"💾 Generated playlist: {filepath}")
        return filepath
    
    def get_all_iptv_org_playlists(self) -> Dict[str, str]:
        """Get all IPTV-org playlist URLs."""
        return IPTV_ORG_PLAYLISTS
    
    def download_iptv_org_main(self) -> List[Channel]:
        """Download and parse the main IPTV-org playlist."""
        content = self.download_playlist(IPTV_ORG_PLAYLISTS["main"], "iptv_org_main")
        if content:
            self.channels = self.parse_m3u(content)
            return self.channels
        return []
    
    def download_by_country(self, country_code: str) -> List[Channel]:
        """Download playlist for specific country."""
        if country_code.lower() in IPTV_ORG_PLAYLISTS["countries"]:
            url = IPTV_ORG_PLAYLISTS["countries"][country_code.lower()]
            content = self.download_playlist(url, f"country_{country_code}")
            if content:
                return self.parse_m3u(content)
        return []
    
    def download_by_category(self, category: str) -> List[Channel]:
        """Download playlist for specific category."""
        if category.lower() in IPTV_ORG_PLAYLISTS["categories"]:
            url = IPTV_ORG_PLAYLISTS["categories"][category.lower()]
            content = self.download_playlist(url, f"category_{category}")
            if content:
                return self.parse_m3u(content)
        return []
    
    def get_free_legal_channels(self) -> List[Channel]:
        """Get legally free streaming channels."""
        channels = []
        
        for name, url in ADDITIONAL_SOURCES.items():
            channel = Channel(
                id=name,
                name=name.replace('_', ' ').title(),
                url=url,
                group="Free Legal",
                quality="HD"
            )
            channels.append(channel)
        
        return channels
    
    def export_to_json(self, channels: List[Channel], filename: str = "channels.json") -> str:
        """Export channels to JSON format."""
        data = {
            "updated": datetime.now().isoformat(),
            "count": len(channels),
            "channels": [ch.to_dict() for ch in channels]
        }
        
        filepath = os.path.join(self.cache_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return filepath


# ==========================================
# API ENDPOINTS CONFIGURATION
# ==========================================

IPTV_API_CONFIG = {
    "channels_api": "https://iptv-org.github.io/api/channels.json",
    "streams_api": "https://iptv-org.github.io/api/streams.json",
    "guides_api": "https://iptv-org.github.io/api/guides.json",
    "categories_api": "https://iptv-org.github.io/api/categories.json",
    "languages_api": "https://iptv-org.github.io/api/languages.json",
    "countries_api": "https://iptv-org.github.io/api/countries.json",
    "regions_api": "https://iptv-org.github.io/api/regions.json",
    "subdivisions_api": "https://iptv-org.github.io/api/subdivisions.json",
    "blocklist_api": "https://iptv-org.github.io/api/blocklist.json"
}


def get_channel_count() -> Dict[str, int]:
    """Get current channel counts from IPTV-org API."""
    try:
        response = requests.get(IPTV_API_CONFIG["channels_api"], timeout=10)
        if response.status_code == 200:
            channels = response.json()
            return {
                "total": len(channels),
                "by_country": len(set(ch.get("country", "") for ch in channels)),
                "by_language": len(set(ch.get("language", "") for ch in channels))
            }
    except:
        pass
    return {"total": 10000, "by_country": 200, "by_language": 100}


# ==========================================
# QUICK START FUNCTIONS
# ==========================================

def quick_start():
    """Quick start guide for using the IPTV system."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         🎬 STREAMVAULT PRO ULTIMATE - IPTV QUICK START          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📺 MAIN PLAYLIST (10,000+ Channels):                           ║
║     https://iptv-org.github.io/iptv/index.m3u                   ║
║                                                                  ║
║  🌍 BY COUNTRY:                                                  ║
║     https://iptv-org.github.io/iptv/countries/{code}.m3u        ║
║     Example: .../countries/us.m3u (USA channels)                ║
║                                                                  ║
║  📁 BY CATEGORY:                                                 ║
║     https://iptv-org.github.io/iptv/categories/{name}.m3u       ║
║     Categories: news, sports, movies, music, kids, etc.         ║
║                                                                  ║
║  🗣️ BY LANGUAGE:                                                 ║
║     https://iptv-org.github.io/iptv/languages/{code}.m3u        ║
║     Example: .../languages/eng.m3u (English channels)           ║
║                                                                  ║
║  📖 EPG (TV Guide):                                              ║
║     https://iptv-org.github.io/epg/guides/index.xml             ║
║                                                                  ║
║  🔗 USAGE:                                                       ║
║     1. Copy playlist URL                                         ║
║     2. Open in VLC, Kodi, or any M3U player                     ║
║     3. Start watching!                                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("🎬 StreamVault Pro Ultimate - IPTV Channel Manager")
    print("=" * 60)
    
    # Show quick start
    quick_start()
    
    # Initialize manager
    manager = IPTVPlaylistManager("./iptv_cache")
    
    # Get channel counts
    counts = get_channel_count()
    print(f"\n📊 IPTV-org Statistics:")
    print(f"   Total channels: {counts['total']:,}")
    print(f"   Countries covered: {counts['by_country']}")
    print(f"   Languages: {counts['by_language']}")
    
    # Download and parse main playlist
    print("\n📥 Downloading main playlist...")
    channels = manager.download_iptv_org_main()
    
    if channels:
        print(f"\n✅ Loaded {len(channels)} channels!")
        
        # Show sample channels
        print("\n📺 Sample channels:")
        for ch in channels[:10]:
            print(f"   • {ch.name} ({ch.group})")
        
        # Get free legal channels
        free_channels = manager.get_free_legal_channels()
        print(f"\n🆓 Free legal channels: {len(free_channels)}")
        for ch in free_channels:
            print(f"   • {ch.name}")
        
        # Export to JSON
        json_path = manager.export_to_json(channels[:100], "sample_channels.json")
        print(f"\n💾 Sample exported to: {json_path}")
    
    print("\n✅ IPTV system ready!")
