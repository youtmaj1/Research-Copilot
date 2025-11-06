"""
Paper Collector Orchestrator

Main orchestrator that coordinates ArXiv and Scholar clients to collect research papers,
download PDFs, and manage metadata with deduplication and error handling.
"""

import os
import logging
import time
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
from urllib.parse import urlparse
import random

from .arxiv_client import ArxivClient
from .scholar_client import ScholarClient, is_available as scholar_available
from .database import PaperDatabase

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff and jitter."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        # Add jitter to avoid thundering herd
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter


class PaperCollector:
    """Main orchestrator for collecting research papers from multiple sources."""
    
    def __init__(self, 
                 data_dir: str = "data",
                 db_path: str = "papers.db",
                 use_scholar_proxy: bool = False,
                 retry_config: Optional[RetryConfig] = None):
        """
        Initialize the Paper Collector.
        
        Args:
            data_dir: Base directory for storing data
            db_path: Path to SQLite database
            use_scholar_proxy: Whether to use proxy for Scholar scraping
            retry_config: Configuration for retry logic
        """
        self.data_dir = Path(data_dir)
        self.papers_dir = self.data_dir / "raw" / "papers"
        self.metadata_dir = self.data_dir / "raw" / "metadata"
        
        # Create directories
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize clients
        self.arxiv_client = ArxivClient()
        self.scholar_client = ScholarClient(use_proxy=use_scholar_proxy) if scholar_available() else None
        self.database = PaperDatabase(db_path)
        
        # Retry configuration
        self.retry_config = retry_config or RetryConfig()
        
        # Session for downloading PDFs
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        logger.info("Paper Collector initialized")
        if not self.scholar_client:
            logger.warning("Scholar client not available - install 'scholarly' package for Scholar support")
    
    def _retry_operation(self, operation, *args, **kwargs):
        """Execute an operation with retry logic and exponential backoff."""
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.retry_config.max_retries + 1} attempts failed")
        
        raise last_exception
    
    def _download_pdf(self, url: str, paper_id: str) -> Optional[str]:
        """
        Download a PDF file from a URL.
        
        Args:
            url: PDF URL
            paper_id: Paper ID for filename
            
        Returns:
            Local path to downloaded PDF or None if failed
        """
        try:
            # Generate filename
            filename = f"{paper_id}.pdf"
            file_path = self.papers_dir / filename
            
            # Skip if already exists
            if file_path.exists():
                logger.debug(f"PDF already exists: {file_path}")
                return str(file_path)
            
            logger.info(f"Downloading PDF: {url}")
            
            # Download with retry logic
            def _download():
                response = self.session.get(url, timeout=30, stream=True)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'application/pdf' not in content_type and 'pdf' not in content_type:
                    logger.warning(f"Unexpected content type: {content_type}")
                
                # Write to file
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return str(file_path)
            
            return self._retry_operation(_download)
            
        except Exception as e:
            logger.error(f"Failed to download PDF from {url}: {e}")
            return None
    
    def _save_metadata(self, paper: Dict) -> bool:
        """
        Save paper metadata to JSON file.
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            import json
            
            filename = f"{paper['id']}_metadata.json"
            file_path = self.metadata_dir / filename
            
            # Convert datetime objects to strings for JSON serialization
            paper_copy = paper.copy()
            for key, value in paper_copy.items():
                if isinstance(value, datetime):
                    paper_copy[key] = value.isoformat()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(paper_copy, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved metadata: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save metadata for {paper['id']}: {e}")
            return False
    
    def _process_papers(self, papers: List[Dict], download_pdfs: bool = True) -> Tuple[int, int]:
        """
        Process a list of papers: save to database, download PDFs, save metadata.
        
        Args:
            papers: List of paper dictionaries
            download_pdfs: Whether to download PDF files
            
        Returns:
            Tuple of (papers_added, papers_skipped)
        """
        papers_added = 0
        papers_skipped = 0
        
        for paper in papers:
            try:
                # Check if paper already exists
                if self.database.paper_exists(
                    paper_id=paper.get('id'),
                    doi=paper.get('doi'),
                    hash_value=paper.get('hash')
                ):
                    logger.debug(f"Skipping duplicate paper: {paper.get('title', 'Unknown')}")
                    papers_skipped += 1
                    continue
                
                # Download PDF if available and requested
                pdf_path = None
                if download_pdfs and paper.get('pdf_url'):
                    pdf_path = self._download_pdf(paper['pdf_url'], paper['id'])
                    if pdf_path:
                        paper['pdf_path'] = pdf_path
                
                # Save to database
                if self.database.add_paper(paper):
                    # Save metadata file
                    self._save_metadata(paper)
                    papers_added += 1
                    logger.info(f"Added paper: {paper.get('title', 'Unknown')}")
                else:
                    papers_skipped += 1
                    logger.warning(f"Failed to add paper to database: {paper.get('title', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"Failed to process paper {paper.get('id', 'unknown')}: {e}")
                papers_skipped += 1
                continue
        
        return papers_added, papers_skipped
    
    def search(self, 
               query: str, 
               max_results: int = 100,
               download_pdfs: bool = True,
               use_scholar_fallback: bool = True) -> Dict:
        """
        Search for papers using ArXiv API with Scholar fallback.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            download_pdfs: Whether to download PDF files
            use_scholar_fallback: Whether to use Scholar as fallback
            
        Returns:
            Dictionary with search results and statistics
        """
        start_time = datetime.now()
        results = {
            'query': query,
            'timestamp': start_time,
            'arxiv_papers': 0,
            'scholar_papers': 0,
            'total_found': 0,
            'papers_added': 0,
            'papers_skipped': 0,
            'pdfs_downloaded': 0,
            'errors': []
        }
        
        logger.info(f"Starting search for: {query}")
        
        try:
            # First, try ArXiv
            arxiv_papers = []
            try:
                arxiv_papers = self._retry_operation(
                    self.arxiv_client.search, 
                    query, 
                    max_results
                )
                results['arxiv_papers'] = len(arxiv_papers)
                logger.info(f"Found {len(arxiv_papers)} papers from ArXiv")
                
            except Exception as e:
                error_msg = f"ArXiv search failed: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
            
            # Use Scholar as fallback if needed and available
            scholar_papers = []
            if (use_scholar_fallback and 
                self.scholar_client and 
                len(arxiv_papers) < max_results):
                
                try:
                    remaining_results = max_results - len(arxiv_papers)
                    scholar_papers = self._retry_operation(
                        self.scholar_client.search,
                        query,
                        remaining_results
                    )
                    results['scholar_papers'] = len(scholar_papers)
                    logger.info(f"Found {len(scholar_papers)} papers from Scholar")
                    
                except Exception as e:
                    error_msg = f"Scholar search failed: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            # Combine and process papers
            all_papers = arxiv_papers + scholar_papers
            results['total_found'] = len(all_papers)
            
            if all_papers:
                # Process papers (deduplication, PDF download, database storage)
                papers_added, papers_skipped = self._process_papers(all_papers, download_pdfs)
                results['papers_added'] = papers_added
                results['papers_skipped'] = papers_skipped
                
                # Count PDFs downloaded
                if download_pdfs:
                    results['pdfs_downloaded'] = sum(1 for p in all_papers if p.get('pdf_path'))
            
            # Log ingestion event
            self.database.log_ingestion(
                query=query,
                source='arxiv+scholar',
                papers_found=results['total_found'],
                papers_added=results['papers_added'],
                papers_skipped=results['papers_skipped'],
                success=len(results['errors']) == 0,
                error_message='; '.join(results['errors']) if results['errors'] else None
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Search completed in {duration:.2f}s: "
                       f"{results['papers_added']} added, {results['papers_skipped']} skipped")
            
            return results
            
        except Exception as e:
            error_msg = f"Search operation failed: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            
            # Log failed ingestion
            self.database.log_ingestion(
                query=query,
                source='arxiv+scholar',
                papers_found=0,
                papers_added=0,
                papers_skipped=0,
                success=False,
                error_message=error_msg
            )
            
            return results
    
    def update_recent(self, 
                     category: str = "cs.AI", 
                     days_back: int = 7,
                     download_pdfs: bool = True) -> Dict:
        """
        Fetch latest papers from ArXiv category.
        
        Args:
            category: ArXiv category (e.g., 'cs.AI', 'cs.LG')
            days_back: Number of days to look back
            download_pdfs: Whether to download PDF files
            
        Returns:
            Dictionary with update results and statistics
        """
        start_time = datetime.now()
        results = {
            'category': category,
            'days_back': days_back,
            'timestamp': start_time,
            'total_found': 0,
            'papers_added': 0,
            'papers_skipped': 0,
            'pdfs_downloaded': 0,
            'errors': []
        }
        
        logger.info(f"Updating recent papers for category {category} ({days_back} days back)")
        
        try:
            # Get recent papers from ArXiv
            papers = self._retry_operation(
                self.arxiv_client.get_recent_papers,
                category,
                days_back,
                max_results=1000  # Get all recent papers
            )
            
            results['total_found'] = len(papers)
            logger.info(f"Found {len(papers)} recent papers in {category}")
            
            if papers:
                # Process papers
                papers_added, papers_skipped = self._process_papers(papers, download_pdfs)
                results['papers_added'] = papers_added
                results['papers_skipped'] = papers_skipped
                
                # Count PDFs downloaded
                if download_pdfs:
                    results['pdfs_downloaded'] = sum(1 for p in papers if p.get('pdf_path'))
            
            # Log ingestion event
            self.database.log_ingestion(
                query=f"recent:{category}:{days_back}d",
                source='arxiv',
                papers_found=results['total_found'],
                papers_added=results['papers_added'],
                papers_skipped=results['papers_skipped'],
                success=True,
                error_message=None
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Update completed in {duration:.2f}s: "
                       f"{results['papers_added']} added, {results['papers_skipped']} skipped")
            
            return results
            
        except Exception as e:
            error_msg = f"Recent update failed: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            
            # Log failed ingestion
            self.database.log_ingestion(
                query=f"recent:{category}:{days_back}d",
                source='arxiv',
                papers_found=0,
                papers_added=0,
                papers_skipped=0,
                success=False,
                error_message=error_msg
            )
            
            return results
    
    def get_stats(self) -> Dict:
        """Get collector statistics."""
        db_stats = self.database.get_stats()
        
        # Add directory stats
        try:
            pdf_count = len(list(self.papers_dir.glob("*.pdf")))
            metadata_count = len(list(self.metadata_dir.glob("*_metadata.json")))
            
            stats = {
                **db_stats,
                'pdf_files': pdf_count,
                'metadata_files': metadata_count,
                'data_directory': str(self.data_dir),
                'arxiv_available': True,
                'scholar_available': self.scholar_client is not None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return db_stats
    
    def search_local(self, 
                    query: str, 
                    source: str = None,
                    limit: int = 100) -> List[Dict]:
        """
        Search locally stored papers.
        
        Args:
            query: Search query
            source: Filter by source ('arxiv' or 'scholar')
            limit: Maximum number of results
            
        Returns:
            List of matching papers
        """
        return self.database.search_papers(query, source, limit)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize collector
    collector = PaperCollector()
    
    # Search for papers
    results = collector.search("large language models", max_results=5)
    print(f"Search results: {results}")
    
    # Get recent papers
    recent_results = collector.update_recent("cs.AI", days_back=3)
    print(f"Recent papers: {recent_results}")
    
    # Get stats
    stats = collector.get_stats()
    print(f"Collector stats: {stats}")
