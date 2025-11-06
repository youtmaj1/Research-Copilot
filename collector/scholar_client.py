"""
Google Scholar Client

A fallback scraper for Google Scholar using the scholarly library.
Provides access to Google Scholar's database with careful request handling
to avoid being blocked.
"""

import time
import logging
from typing import List, Dict, Optional
import hashlib
from datetime import datetime
import random

try:
    from scholarly import scholarly, ProxyGenerator
    SCHOLARLY_AVAILABLE = True
except ImportError:
    SCHOLARLY_AVAILABLE = False
    logging.warning("scholarly library not available. Install with: pip install scholarly")

logger = logging.getLogger(__name__)


class ScholarClient:
    """Client for scraping Google Scholar using the scholarly library."""
    
    def __init__(self, use_proxy: bool = False, delay_range: tuple = (5, 10)):
        """
        Initialize the Scholar client.
        
        Args:
            use_proxy: Whether to use proxy rotation (recommended for production)
            delay_range: Range of seconds to wait between requests (min, max)
        """
        if not SCHOLARLY_AVAILABLE:
            raise ImportError("scholarly library is required. Install with: pip install scholarly")
        
        self.delay_range = delay_range
        self.last_request_time = 0
        
        # Setup proxy rotation if requested
        if use_proxy:
            try:
                pg = ProxyGenerator()
                pg.FreeProxies()
                scholarly.use_proxy(pg)
                logger.info("Proxy rotation enabled for Scholar client")
            except Exception as e:
                logger.warning(f"Failed to setup proxy rotation: {e}")
    
    def _rate_limit(self):
        """Rate limiting to avoid being blocked by Google Scholar."""
        elapsed = time.time() - self.last_request_time
        min_delay = random.uniform(*self.delay_range)
        
        if elapsed < min_delay:
            sleep_time = min_delay - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _normalize_paper(self, scholar_paper) -> Dict:
        """Convert scholarly paper object to our standard format."""
        try:
            # Extract basic information
            title = scholar_paper.get('title', '').strip()
            authors = scholar_paper.get('author', [])
            if isinstance(authors, str):
                authors = [authors]
            
            # Extract publication info
            venue = scholar_paper.get('venue', '')
            year = scholar_paper.get('year', '')
            
            # Try to extract abstract (may not always be available)
            abstract = scholar_paper.get('abstract', '')
            
            # Extract citation count
            citations = scholar_paper.get('num_citations', 0)
            
            # Extract URL
            url = scholar_paper.get('url', '')
            
            # Extract PDF URL if available
            pdf_url = None
            if 'eprint_url' in scholar_paper:
                pdf_url = scholar_paper['eprint_url']
            
            # Create publication date (may be incomplete)
            published_date = None
            if year:
                try:
                    published_date = datetime(int(year), 1, 1)
                except (ValueError, TypeError):
                    pass
            
            # Create hash for deduplication
            content_hash = hashlib.md5(f"{title}_{authors[0] if authors else ''}".encode()).hexdigest()
            
            # Generate an ID based on title hash (Scholar doesn't provide consistent IDs)
            paper_id = f"scholar_{hashlib.md5(title.encode()).hexdigest()[:12]}"
            
            return {
                'id': paper_id,
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'doi': None,  # Scholar doesn't reliably provide DOI
                'published_date': published_date,
                'venue': venue,
                'year': year,
                'citations': citations,
                'source': 'scholar',
                'url': url,
                'pdf_url': pdf_url,
                'hash': content_hash
            }
            
        except Exception as e:
            logger.error(f"Failed to normalize Scholar paper: {e}")
            return None
    
    def search(self, query: str, max_results: int = 100, start_year: Optional[int] = None) -> List[Dict]:
        """
        Search Google Scholar for papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            start_year: Optional year to filter results from
            
        Returns:
            List of paper metadata dictionaries
        """
        if not SCHOLARLY_AVAILABLE:
            logger.error("scholarly library not available")
            return []
        
        papers = []
        try:
            logger.info(f"Searching Google Scholar for: {query}")
            
            # Build search query
            search_query = scholarly.search_pubs(query)
            
            count = 0
            for paper in search_query:
                if count >= max_results:
                    break
                
                # Rate limiting
                self._rate_limit()
                
                try:
                    # Filter by year if specified
                    if start_year and paper.get('year'):
                        try:
                            paper_year = int(paper['year'])
                            if paper_year < start_year:
                                continue
                        except (ValueError, TypeError):
                            pass
                    
                    # Normalize the paper
                    normalized_paper = self._normalize_paper(paper)
                    if normalized_paper:
                        papers.append(normalized_paper)
                        count += 1
                        logger.debug(f"Found paper: {normalized_paper['title'][:50]}...")
                
                except Exception as e:
                    logger.error(f"Failed to process Scholar paper: {e}")
                    continue
            
            logger.info(f"Successfully retrieved {len(papers)} papers from Google Scholar")
            return papers
            
        except Exception as e:
            logger.error(f"Google Scholar search failed: {e}")
            return []
    
    def search_author(self, author_name: str, max_results: int = 50) -> List[Dict]:
        """
        Search for papers by a specific author.
        
        Args:
            author_name: Name of the author
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        if not SCHOLARLY_AVAILABLE:
            logger.error("scholarly library not available")
            return []
        
        papers = []
        try:
            logger.info(f"Searching Google Scholar for author: {author_name}")
            
            # Search for the author first
            search_query = scholarly.search_author(author_name)
            
            # Get the first author result
            try:
                author = next(search_query)
                # Fill in author details
                author = scholarly.fill(author)
                
                # Get publications
                count = 0
                for pub in author['publications']:
                    if count >= max_results:
                        break
                    
                    # Rate limiting
                    self._rate_limit()
                    
                    try:
                        # Fill in publication details
                        filled_pub = scholarly.fill(pub)
                        
                        # Normalize the paper
                        normalized_paper = self._normalize_paper(filled_pub)
                        if normalized_paper:
                            papers.append(normalized_paper)
                            count += 1
                            logger.debug(f"Found paper: {normalized_paper['title'][:50]}...")
                    
                    except Exception as e:
                        logger.error(f"Failed to process author publication: {e}")
                        continue
                
                logger.info(f"Successfully retrieved {len(papers)} papers for author {author_name}")
                return papers
                
            except StopIteration:
                logger.warning(f"No author found for: {author_name}")
                return []
            
        except Exception as e:
            logger.error(f"Google Scholar author search failed: {e}")
            return []
    
    def get_paper_details(self, paper_title: str) -> Optional[Dict]:
        """
        Get detailed information for a specific paper by title.
        
        Args:
            paper_title: Title of the paper
            
        Returns:
            Paper metadata dictionary or None if not found
        """
        if not SCHOLARLY_AVAILABLE:
            logger.error("scholarly library not available")
            return None
        
        try:
            # Search for the specific paper
            search_query = scholarly.search_pubs(paper_title)
            
            # Rate limiting
            self._rate_limit()
            
            # Get the first result
            paper = next(search_query)
            
            # Fill in details
            filled_paper = scholarly.fill(paper)
            
            # Normalize and return
            return self._normalize_paper(filled_paper)
            
        except StopIteration:
            logger.warning(f"No paper found with title: {paper_title}")
            return None
        except Exception as e:
            logger.error(f"Failed to get paper details: {e}")
            return None
    
    def search_recent(self, query: str, max_results: int = 50, start_year: int = None) -> List[Dict]:
        """
        Search for recent papers (useful for getting latest research).
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            start_year: Start year for filtering (defaults to current year - 1)
            
        Returns:
            List of paper metadata dictionaries
        """
        if start_year is None:
            start_year = datetime.now().year - 1
        
        return self.search(query, max_results, start_year)


def is_available() -> bool:
    """Check if the scholarly library is available."""
    return SCHOLARLY_AVAILABLE


if __name__ == "__main__":
    # Example usage
    if SCHOLARLY_AVAILABLE:
        client = ScholarClient()
        
        # Search for papers
        papers = client.search("large language models", max_results=3)
        for paper in papers:
            print(f"Title: {paper['title']}")
            print(f"Authors: {', '.join(paper['authors'])}")
            print(f"Year: {paper['year']}")
            print(f"Citations: {paper['citations']}")
            print("---")
    else:
        print("scholarly library not available. Install with: pip install scholarly")
