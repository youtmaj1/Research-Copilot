"""
ArXiv API Client

A wrapper around the ArXiv API to fetch research papers and metadata.
Provides structured access to ArXiv's database with proper error handling
and rate limiting.
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Generator
import time
import logging
from urllib.parse import urlencode
import hashlib

logger = logging.getLogger(__name__)


class ArxivClient:
    """Client for interacting with the ArXiv API."""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    RATE_LIMIT_DELAY = 3  # seconds between requests
    MAX_RESULTS_PER_REQUEST = 1000
    
    def __init__(self):
        self.session = requests.Session()
        self.last_request_time = 0
        
    def _rate_limit(self):
        """Ensure we don't exceed ArXiv's rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, params: Dict) -> str:
        """Make a request to the ArXiv API with rate limiting."""
        self._rate_limit()
        
        url = f"{self.BASE_URL}?{urlencode(params)}"
        logger.info(f"Making ArXiv API request: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"ArXiv API request failed: {e}")
            raise
    
    def _parse_entry(self, entry) -> Dict:
        """Parse a single ArXiv entry from XML to structured metadata."""
        # Define namespaces
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        # Extract basic fields
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        
        # Extract ArXiv ID from the id field
        arxiv_url = entry.find('atom:id', ns).text
        arxiv_id = arxiv_url.split('/')[-1]
        
        # Extract authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns).text
            authors.append(name)
        
        # Extract published and updated dates
        published = entry.find('atom:published', ns).text
        updated = entry.find('atom:updated', ns).text
        
        # Extract categories
        categories = []
        for category in entry.findall('atom:category', ns):
            categories.append(category.get('term'))
        
        # Extract DOI if available
        doi = None
        doi_elem = entry.find('arxiv:doi', ns)
        if doi_elem is not None:
            doi = doi_elem.text
        
        # Extract PDF link
        pdf_url = None
        for link in entry.findall('atom:link', ns):
            if link.get('type') == 'application/pdf':
                pdf_url = link.get('href')
                break
        
        # Create a hash for deduplication
        content_hash = hashlib.md5(f"{title}_{authors[0] if authors else ''}".encode()).hexdigest()
        
        return {
            'id': arxiv_id,
            'title': title,
            'authors': authors,
            'abstract': summary,
            'doi': doi,
            'published_date': datetime.fromisoformat(published.replace('Z', '+00:00')),
            'updated_date': datetime.fromisoformat(updated.replace('Z', '+00:00')),
            'categories': categories,
            'source': 'arxiv',
            'pdf_url': pdf_url,
            'arxiv_url': arxiv_url,
            'hash': content_hash
        }
    
    def _parse_response(self, xml_content: str) -> List[Dict]:
        """Parse ArXiv API XML response into structured metadata."""
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
            }
            
            # Check for errors
            total_results = int(root.find('opensearch:totalResults', ns).text)
            if total_results == 0:
                logger.warning("No results found in ArXiv response")
                return []
            
            # Parse entries
            papers = []
            for entry in root.findall('atom:entry', ns):
                try:
                    paper = self._parse_entry(entry)
                    papers.append(paper)
                except Exception as e:
                    logger.error(f"Failed to parse ArXiv entry: {e}")
                    continue
            
            logger.info(f"Successfully parsed {len(papers)} papers from ArXiv")
            return papers
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse ArXiv XML response: {e}")
            raise
    
    def search(self, query: str, max_results: int = 100, start: int = 0) -> List[Dict]:
        """
        Search ArXiv for papers matching the query.
        
        Args:
            query: Search query (supports ArXiv query syntax)
            max_results: Maximum number of results to return
            start: Starting index for pagination
            
        Returns:
            List of paper metadata dictionaries
        """
        params = {
            'search_query': query,
            'start': start,
            'max_results': min(max_results, self.MAX_RESULTS_PER_REQUEST),
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        xml_response = self._make_request(params)
        return self._parse_response(xml_response)
    
    def get_recent_papers(self, category: str, days_back: int = 7, max_results: int = 100) -> List[Dict]:
        """
        Get recent papers from a specific ArXiv category.
        
        Args:
            category: ArXiv category (e.g., 'cs.AI', 'cs.LG', 'stat.ML')
            days_back: Number of days to look back
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Format dates for ArXiv query
        start_str = start_date.strftime('%Y%m%d%H%M')
        end_str = end_date.strftime('%Y%m%d%H%M')
        
        # Build query for category and date range
        query = f"cat:{category} AND submittedDate:[{start_str} TO {end_str}]"
        
        return self.search(query, max_results)
    
    def get_by_id(self, arxiv_id: str) -> Optional[Dict]:
        """
        Get a specific paper by ArXiv ID.
        
        Args:
            arxiv_id: ArXiv paper ID (e.g., '2301.07041')
            
        Returns:
            Paper metadata dictionary or None if not found
        """
        query = f"id:{arxiv_id}"
        results = self.search(query, max_results=1)
        return results[0] if results else None
    
    def search_by_author(self, author: str, max_results: int = 100) -> List[Dict]:
        """
        Search for papers by a specific author.
        
        Args:
            author: Author name
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        query = f"au:{author}"
        return self.search(query, max_results)
    
    def search_advanced(self, 
                       title: Optional[str] = None,
                       author: Optional[str] = None,
                       abstract: Optional[str] = None,
                       category: Optional[str] = None,
                       max_results: int = 100) -> List[Dict]:
        """
        Advanced search with multiple criteria.
        
        Args:
            title: Title keywords
            author: Author name
            abstract: Abstract keywords
            category: ArXiv category
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        query_parts = []
        
        if title:
            query_parts.append(f"ti:{title}")
        if author:
            query_parts.append(f"au:{author}")
        if abstract:
            query_parts.append(f"abs:{abstract}")
        if category:
            query_parts.append(f"cat:{category}")
        
        if not query_parts:
            raise ValueError("At least one search criterion must be provided")
        
        query = " AND ".join(query_parts)
        return self.search(query, max_results)


def get_popular_categories():
    """Return list of popular ArXiv categories for AI/ML research."""
    return {
        'cs.AI': 'Artificial Intelligence',
        'cs.LG': 'Machine Learning',
        'cs.CL': 'Computation and Language',
        'cs.CV': 'Computer Vision and Pattern Recognition',
        'cs.NE': 'Neural and Evolutionary Computing',
        'stat.ML': 'Machine Learning (Statistics)',
        'cs.IR': 'Information Retrieval',
        'cs.RO': 'Robotics',
        'cs.HC': 'Human-Computer Interaction',
        'cs.CR': 'Cryptography and Security'
    }


if __name__ == "__main__":
    # Example usage
    client = ArxivClient()
    
    # Search for papers
    papers = client.search("large language models", max_results=5)
    for paper in papers:
        print(f"Title: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"ArXiv ID: {paper['id']}")
        print("---")
