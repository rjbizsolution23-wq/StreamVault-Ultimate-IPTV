#!/usr/bin/env python3
"""
StreamVault Pro Ultimate - Universal Academic Research Scraper
Integrates with ALL major academic databases for cutting-edge IPTV/Streaming research
Based on: PART 1-10 Academic Resources Catalog
"""

import requests
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse

class UniversalAcademicScraper:
    """
    Comprehensive academic research scraper for IPTV, streaming, ML, and AI research.
    Integrates with:
    - arXiv (2.4M+ papers)
    - Semantic Scholar (200M+ papers) 
    - OpenAlex (250M+ papers)
    - Papers With Code
    - PubMed Central (35M+ papers)
    - DBLP
    - Hugging Face Papers
    """
    
    def __init__(self, output_dir: str = "./research_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # API Endpoints
        self.endpoints = {
            'arxiv': 'http://export.arxiv.org/api/query',
            'semantic_scholar': 'https://api.semanticscholar.org/graph/v1',
            'openalex': 'https://api.openalex.org',
            'papers_with_code': 'https://paperswithcode.com/api/v1',
            'pubmed': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils',
            'dblp': 'https://dblp.org/search/publ/api',
            'huggingface': 'https://huggingface.co/api'
        }
        
        # IPTV/Streaming Research Topics
        self.iptv_topics = [
            'IPTV streaming machine learning',
            'adaptive bitrate optimization neural network',
            'WebRTC ultra low latency streaming',
            'video quality assessment deep learning',
            'network digital twin video streaming',
            'reinforcement learning adaptive streaming',
            '5G edge computing video delivery',
            'DASH HLS optimization machine learning',
            'video codec neural network',
            'content delivery network optimization AI'
        ]
        
        self.headers = {
            'User-Agent': 'StreamVault-Research-Bot/1.0 (Academic Research)',
            'Accept': 'application/json'
        }
    
    def scrape_arxiv(self, query: str, max_results: int = 100) -> List[Dict]:
        """Scrape arXiv for IPTV/streaming research papers."""
        papers = []
        
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(
                self.endpoints['arxiv'],
                params=params,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                # Parse XML response
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
                
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                
                for entry in root.findall('atom:entry', ns):
                    paper = {
                        'source': 'arxiv',
                        'id': entry.find('atom:id', ns).text if entry.find('atom:id', ns) is not None else '',
                        'title': entry.find('atom:title', ns).text.strip() if entry.find('atom:title', ns) is not None else '',
                        'abstract': entry.find('atom:summary', ns).text.strip() if entry.find('atom:summary', ns) is not None else '',
                        'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                        'published': entry.find('atom:published', ns).text if entry.find('atom:published', ns) is not None else '',
                        'pdf_url': None,
                        'categories': []
                    }
                    
                    for link in entry.findall('atom:link', ns):
                        if link.get('title') == 'pdf':
                            paper['pdf_url'] = link.get('href')
                    
                    for category in entry.findall('atom:category', ns):
                        paper['categories'].append(category.get('term'))
                    
                    papers.append(paper)
                    
        except Exception as e:
            print(f"arXiv scraping error: {e}")
        
        return papers
    
    def scrape_semantic_scholar(self, query: str, limit: int = 100) -> List[Dict]:
        """Scrape Semantic Scholar for research papers."""
        papers = []
        
        params = {
            'query': query,
            'limit': limit,
            'fields': 'paperId,title,abstract,authors,year,citationCount,venue,openAccessPdf,url'
        }
        
        try:
            response = requests.get(
                f"{self.endpoints['semantic_scholar']}/paper/search",
                params=params,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                for paper_data in data.get('data', []):
                    paper = {
                        'source': 'semantic_scholar',
                        'id': paper_data.get('paperId', ''),
                        'title': paper_data.get('title', ''),
                        'abstract': paper_data.get('abstract', ''),
                        'authors': [a.get('name', '') for a in paper_data.get('authors', [])],
                        'year': paper_data.get('year'),
                        'citations': paper_data.get('citationCount', 0),
                        'venue': paper_data.get('venue', ''),
                        'pdf_url': paper_data.get('openAccessPdf', {}).get('url') if paper_data.get('openAccessPdf') else None,
                        'url': paper_data.get('url', '')
                    }
                    papers.append(paper)
                    
        except Exception as e:
            print(f"Semantic Scholar scraping error: {e}")
        
        return papers
    
    def scrape_openalex(self, query: str, per_page: int = 100) -> List[Dict]:
        """Scrape OpenAlex for research papers (250M+ papers, no rate limits)."""
        papers = []
        
        params = {
            'search': query,
            'per-page': per_page,
            'sort': 'cited_by_count:desc'
        }
        
        try:
            response = requests.get(
                f"{self.endpoints['openalex']}/works",
                params=params,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                for work in data.get('results', []):
                    paper = {
                        'source': 'openalex',
                        'id': work.get('id', ''),
                        'title': work.get('title', ''),
                        'abstract': work.get('abstract_inverted_index', {}),
                        'authors': [a.get('author', {}).get('display_name', '') for a in work.get('authorships', [])],
                        'year': work.get('publication_year'),
                        'citations': work.get('cited_by_count', 0),
                        'doi': work.get('doi', ''),
                        'pdf_url': work.get('open_access', {}).get('oa_url'),
                        'concepts': [c.get('display_name', '') for c in work.get('concepts', [])[:5]]
                    }
                    papers.append(paper)
                    
        except Exception as e:
            print(f"OpenAlex scraping error: {e}")
        
        return papers
    
    def scrape_papers_with_code(self, query: str, items_per_page: int = 50) -> List[Dict]:
        """Scrape Papers With Code for ML papers with implementations."""
        papers = []
        
        try:
            response = requests.get(
                f"{self.endpoints['papers_with_code']}/papers/",
                params={'q': query, 'items_per_page': items_per_page},
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                for paper_data in data.get('results', []):
                    paper = {
                        'source': 'papers_with_code',
                        'id': paper_data.get('id', ''),
                        'title': paper_data.get('title', ''),
                        'abstract': paper_data.get('abstract', ''),
                        'authors': paper_data.get('authors', []),
                        'arxiv_id': paper_data.get('arxiv_id', ''),
                        'url_pdf': paper_data.get('url_pdf', ''),
                        'url_abs': paper_data.get('url_abs', ''),
                        'proceeding': paper_data.get('proceeding', ''),
                        'conference': paper_data.get('conference', ''),
                        'repository_url': None  # Fetch separately if needed
                    }
                    papers.append(paper)
                    
        except Exception as e:
            print(f"Papers With Code scraping error: {e}")
        
        return papers
    
    def scrape_huggingface_papers(self, limit: int = 50) -> List[Dict]:
        """Scrape Hugging Face daily papers."""
        papers = []
        
        try:
            response = requests.get(
                f"{self.endpoints['huggingface']}/daily_papers",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                for paper_data in data[:limit]:
                    paper = {
                        'source': 'huggingface',
                        'id': paper_data.get('paper', {}).get('id', ''),
                        'title': paper_data.get('paper', {}).get('title', ''),
                        'abstract': paper_data.get('paper', {}).get('summary', ''),
                        'authors': paper_data.get('paper', {}).get('authors', []),
                        'published_at': paper_data.get('publishedAt', ''),
                        'upvotes': paper_data.get('paper', {}).get('upvotes', 0)
                    }
                    papers.append(paper)
                    
        except Exception as e:
            print(f"Hugging Face scraping error: {e}")
        
        return papers
    
    def scrape_dblp(self, query: str, max_results: int = 100) -> List[Dict]:
        """Scrape DBLP for computer science publications."""
        papers = []
        
        params = {
            'q': query,
            'format': 'json',
            'h': max_results
        }
        
        try:
            response = requests.get(
                self.endpoints['dblp'],
                params=params,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                hits = data.get('result', {}).get('hits', {}).get('hit', [])
                
                for hit in hits:
                    info = hit.get('info', {})
                    paper = {
                        'source': 'dblp',
                        'id': hit.get('@id', ''),
                        'title': info.get('title', ''),
                        'authors': info.get('authors', {}).get('author', []),
                        'venue': info.get('venue', ''),
                        'year': info.get('year', ''),
                        'type': info.get('type', ''),
                        'url': info.get('url', ''),
                        'doi': info.get('doi', '')
                    }
                    papers.append(paper)
                    
        except Exception as e:
            print(f"DBLP scraping error: {e}")
        
        return papers
    
    def scrape_all_iptv_research(self) -> Dict[str, List[Dict]]:
        """Scrape all sources for IPTV/streaming research."""
        all_results = {}
        
        print("🔬 Starting comprehensive IPTV research scrape...")
        
        for topic in self.iptv_topics:
            print(f"  📚 Researching: {topic}")
            
            topic_results = {
                'arxiv': self.scrape_arxiv(topic, 20),
                'semantic_scholar': self.scrape_semantic_scholar(topic, 20),
                'openalex': self.scrape_openalex(topic, 20),
                'papers_with_code': self.scrape_papers_with_code(topic, 10)
            }
            
            all_results[topic] = topic_results
            time.sleep(1)  # Respect rate limits
        
        # Add Hugging Face latest
        all_results['huggingface_daily'] = self.scrape_huggingface_papers(50)
        
        return all_results
    
    def download_pdf(self, url: str, filename: str) -> bool:
        """Download a PDF paper."""
        try:
            response = requests.get(url, headers=self.headers, timeout=60, stream=True)
            if response.status_code == 200:
                filepath = os.path.join(self.output_dir, 'pdfs', filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        except Exception as e:
            print(f"PDF download error: {e}")
        return False
    
    def bulk_download_pdfs(self, papers: List[Dict], max_workers: int = 5) -> int:
        """Bulk download PDFs from papers with threading."""
        downloaded = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for paper in papers:
                pdf_url = paper.get('pdf_url') or paper.get('url_pdf')
                if pdf_url:
                    paper_id = paper.get('id', '').replace('/', '_').replace(':', '_')
                    filename = f"{paper['source']}_{paper_id}.pdf"
                    futures.append(executor.submit(self.download_pdf, pdf_url, filename))
            
            for future in as_completed(futures):
                if future.result():
                    downloaded += 1
        
        return downloaded
    
    def export_results(self, results: Dict, filename: str = "iptv_research.json"):
        """Export results to JSON."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Results exported to: {filepath}")
        return filepath


class IPTVResearchAnalyzer:
    """Analyze and categorize IPTV research for platform integration."""
    
    def __init__(self, research_data: Dict):
        self.data = research_data
        self.categories = {
            'adaptive_streaming': [],
            'video_quality': [],
            'low_latency': [],
            'machine_learning': [],
            'cdn_optimization': [],
            '5g_edge': [],
            'security': [],
            'codec_transcoding': []
        }
    
    def categorize_paper(self, paper: Dict) -> List[str]:
        """Categorize a paper by topic."""
        categories = []
        title = (paper.get('title', '') + ' ' + str(paper.get('abstract', ''))).lower()
        
        if any(kw in title for kw in ['adaptive', 'bitrate', 'abr', 'dash', 'hls']):
            categories.append('adaptive_streaming')
        if any(kw in title for kw in ['quality', 'qoe', 'qos', 'assessment', 'vmaf']):
            categories.append('video_quality')
        if any(kw in title for kw in ['latency', 'real-time', 'webrtc', 'low-latency']):
            categories.append('low_latency')
        if any(kw in title for kw in ['machine learning', 'neural', 'deep learning', 'reinforcement']):
            categories.append('machine_learning')
        if any(kw in title for kw in ['cdn', 'edge', 'caching', 'delivery']):
            categories.append('cdn_optimization')
        if any(kw in title for kw in ['5g', 'mec', 'mobile edge']):
            categories.append('5g_edge')
        if any(kw in title for kw in ['security', 'drm', 'encryption', 'protection']):
            categories.append('security')
        if any(kw in title for kw in ['codec', 'encoding', 'transcoding', 'av1', 'hevc', 'h.265']):
            categories.append('codec_transcoding')
        
        return categories if categories else ['general']
    
    def generate_insights(self) -> Dict:
        """Generate research insights for platform enhancement."""
        insights = {
            'total_papers': 0,
            'by_source': {},
            'by_category': {},
            'top_cited': [],
            'recent_breakthroughs': [],
            'implementation_ready': []
        }
        
        all_papers = []
        
        for topic, sources in self.data.items():
            if isinstance(sources, dict):
                for source, papers in sources.items():
                    for paper in papers:
                        paper['topic'] = topic
                        all_papers.append(paper)
                        insights['total_papers'] += 1
                        insights['by_source'][source] = insights['by_source'].get(source, 0) + 1
            elif isinstance(sources, list):
                for paper in sources:
                    paper['topic'] = topic
                    all_papers.append(paper)
                    insights['total_papers'] += 1
        
        # Categorize all papers
        for paper in all_papers:
            cats = self.categorize_paper(paper)
            for cat in cats:
                if cat not in insights['by_category']:
                    insights['by_category'][cat] = []
                insights['by_category'][cat].append(paper)
        
        # Top cited
        cited_papers = [p for p in all_papers if p.get('citations', 0) > 0]
        insights['top_cited'] = sorted(cited_papers, key=lambda x: x.get('citations', 0), reverse=True)[:20]
        
        # Implementation ready (from Papers With Code)
        insights['implementation_ready'] = [p for p in all_papers if p.get('source') == 'papers_with_code'][:20]
        
        return insights


# Bulk Download Instructions for Large-Scale Research
BULK_DOWNLOAD_COMMANDS = """
# ==========================================
# BULK DOWNLOAD INSTRUCTIONS FOR RESEARCH
# ==========================================

# 1. arXiv Bulk Download (S3)
aws s3 sync s3://arxiv/pdf/ ./arxiv_pdfs/ --no-sign-request

# 2. Semantic Scholar Bulk Download
wget https://api.semanticscholar.org/datasets/v1/release/latest/dataset/papers
wget https://api.semanticscholar.org/datasets/v1/release/latest/dataset/abstracts

# 3. OpenAlex Snapshot Download
wget https://openalex.org/snapshot
# OR via S3:
aws s3 sync s3://openalex ./openalex_data/ --no-sign-request

# 4. Common Crawl (for NLP datasets)
aws s3 sync s3://commoncrawl/cc-news/ ./common_crawl_news/ --no-sign-request

# 5. The Pile (825GB NLP dataset)
# Download from: https://pile.eleuther.ai/

# 6. Kaggle Datasets
pip install kaggle
kaggle datasets download -d Cornell-University/arxiv
"""


if __name__ == "__main__":
    print("=" * 60)
    print("🎬 StreamVault Pro Ultimate - Academic Research Integration")
    print("=" * 60)
    
    scraper = UniversalAcademicScraper(output_dir="./iptv_research")
    
    # Scrape all IPTV-related research
    results = scraper.scrape_all_iptv_research()
    
    # Export results
    scraper.export_results(results)
    
    # Analyze and generate insights
    analyzer = IPTVResearchAnalyzer(results)
    insights = analyzer.generate_insights()
    
    print("\n📊 Research Insights:")
    print(f"  Total papers found: {insights['total_papers']}")
    print(f"  Sources: {insights['by_source']}")
    print(f"  Categories: {list(insights['by_category'].keys())}")
    print(f"  Implementation-ready papers: {len(insights['implementation_ready'])}")
    
    # Export insights
    with open('./iptv_research/insights.json', 'w') as f:
        json.dump(insights, f, indent=2, default=str)
    
    print("\n✅ Research integration complete!")
