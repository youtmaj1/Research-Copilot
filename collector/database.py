"""
Database schema and operations for the Paper Collector.

Handles SQLite database operations for storing paper metadata,
deduplication, and indexing.
"""

import sqlite3
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class PaperDatabase:
    """Database handler for paper metadata and deduplication."""
    
    def __init__(self, db_path: str = "papers.db"):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create papers table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS papers (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        authors TEXT NOT NULL,  -- JSON array of authors
                        abstract TEXT,
                        doi TEXT,
                        published_date DATE,
                        updated_date DATE,
                        venue TEXT,
                        year INTEGER,
                        categories TEXT,  -- JSON array of categories
                        source TEXT NOT NULL,  -- 'arxiv' or 'scholar'
                        url TEXT,
                        pdf_url TEXT,
                        pdf_path TEXT,  -- Local path to downloaded PDF
                        citations INTEGER DEFAULT 0,
                        hash TEXT NOT NULL,  -- For deduplication
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indices for efficient querying
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_hash ON papers(hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_published_date ON papers(published_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_papers_created_at ON papers(created_at)")
                
                # Create ingestion log table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ingestion_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT,
                        source TEXT,
                        papers_found INTEGER,
                        papers_added INTEGER,
                        papers_skipped INTEGER,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        success BOOLEAN DEFAULT TRUE,
                        error_message TEXT
                    )
                """)
                
                # Create search history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS search_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL,
                        source TEXT NOT NULL,
                        results_count INTEGER,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
                
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def paper_exists(self, paper_id: str = None, doi: str = None, hash_value: str = None) -> bool:
        """
        Check if a paper already exists in the database.
        
        Args:
            paper_id: Paper ID
            doi: DOI
            hash_value: Content hash
            
        Returns:
            True if paper exists, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                conditions = []
                params = []
                
                if paper_id:
                    conditions.append("id = ?")
                    params.append(paper_id)
                
                if doi:
                    conditions.append("doi = ?")
                    params.append(doi)
                
                if hash_value:
                    conditions.append("hash = ?")
                    params.append(hash_value)
                
                if not conditions:
                    return False
                
                query = f"SELECT COUNT(*) FROM papers WHERE {' OR '.join(conditions)}"
                cursor.execute(query, params)
                
                count = cursor.fetchone()[0]
                return count > 0
                
        except sqlite3.Error as e:
            logger.error(f"Failed to check if paper exists: {e}")
            return False
    
    def add_paper(self, paper: Dict) -> bool:
        """
        Add a paper to the database.
        
        Args:
            paper: Paper metadata dictionary
            
        Returns:
            True if paper was added, False if it already exists or failed
        """
        try:
            # Check for duplicates
            if self.paper_exists(
                paper_id=paper.get('id'),
                doi=paper.get('doi'),
                hash_value=paper.get('hash')
            ):
                logger.debug(f"Paper already exists: {paper.get('title', 'Unknown')}")
                return False
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Prepare data
                authors_json = json.dumps(paper.get('authors', []))
                categories_json = json.dumps(paper.get('categories', []))
                
                # Handle date conversion
                published_date = paper.get('published_date')
                if isinstance(published_date, datetime):
                    published_date = published_date.isoformat()
                
                updated_date = paper.get('updated_date')
                if isinstance(updated_date, datetime):
                    updated_date = updated_date.isoformat()
                
                cursor.execute("""
                    INSERT INTO papers (
                        id, title, authors, abstract, doi, published_date, updated_date,
                        venue, year, categories, source, url, pdf_url, pdf_path,
                        citations, hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper.get('id'),
                    paper.get('title'),
                    authors_json,
                    paper.get('abstract'),
                    paper.get('doi'),
                    published_date,
                    updated_date,
                    paper.get('venue'),
                    paper.get('year'),
                    categories_json,
                    paper.get('source'),
                    paper.get('url'),
                    paper.get('pdf_url'),
                    paper.get('pdf_path'),
                    paper.get('citations', 0),
                    paper.get('hash')
                ))
                
                conn.commit()
                logger.debug(f"Added paper: {paper.get('title', 'Unknown')}")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Failed to add paper: {e}")
            return False
    
    def update_pdf_path(self, paper_id: str, pdf_path: str) -> bool:
        """
        Update the local PDF path for a paper.
        
        Args:
            paper_id: Paper ID
            pdf_path: Local path to the PDF file
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE papers 
                    SET pdf_path = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (pdf_path, paper_id))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.debug(f"Updated PDF path for paper {paper_id}")
                    return True
                else:
                    logger.warning(f"No paper found with ID {paper_id}")
                    return False
                
        except sqlite3.Error as e:
            logger.error(f"Failed to update PDF path: {e}")
            return False
    
    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """
        Get a paper by its ID.
        
        Args:
            paper_id: Paper ID
            
        Returns:
            Paper dictionary or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
                row = cursor.fetchone()
                
                if row:
                    paper = dict(row)
                    # Parse JSON fields
                    paper['authors'] = json.loads(paper['authors'])
                    paper['categories'] = json.loads(paper['categories']) if paper['categories'] else []
                    return paper
                
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get paper: {e}")
            return None
    
    def search_papers(self, 
                     query: str = None,
                     source: str = None,
                     limit: int = 100,
                     offset: int = 0) -> List[Dict]:
        """
        Search papers in the database.
        
        Args:
            query: Search query (searches title and abstract)
            source: Filter by source ('arxiv' or 'scholar')
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of paper dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                conditions = []
                params = []
                
                if query:
                    conditions.append("(title LIKE ? OR abstract LIKE ?)")
                    params.extend([f"%{query}%", f"%{query}%"])
                
                if source:
                    conditions.append("source = ?")
                    params.append(source)
                
                where_clause = ""
                if conditions:
                    where_clause = f"WHERE {' AND '.join(conditions)}"
                
                sql = f"""
                    SELECT * FROM papers 
                    {where_clause}
                    ORDER BY published_date DESC, created_at DESC
                    LIMIT ? OFFSET ?
                """
                params.extend([limit, offset])
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                papers = []
                for row in rows:
                    paper = dict(row)
                    # Parse JSON fields
                    paper['authors'] = json.loads(paper['authors'])
                    paper['categories'] = json.loads(paper['categories']) if paper['categories'] else []
                    papers.append(paper)
                
                return papers
                
        except sqlite3.Error as e:
            logger.error(f"Failed to search papers: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Total papers
                cursor.execute("SELECT COUNT(*) FROM papers")
                stats['total_papers'] = cursor.fetchone()[0]
                
                # Papers by source
                cursor.execute("SELECT source, COUNT(*) FROM papers GROUP BY source")
                stats['by_source'] = dict(cursor.fetchall())
                
                # Papers with PDFs
                cursor.execute("SELECT COUNT(*) FROM papers WHERE pdf_path IS NOT NULL")
                stats['papers_with_pdfs'] = cursor.fetchone()[0]
                
                # Recent papers (last 30 days)
                cursor.execute("""
                    SELECT COUNT(*) FROM papers 
                    WHERE created_at >= datetime('now', '-30 days')
                """)
                stats['recent_papers'] = cursor.fetchone()[0]
                
                return stats
                
        except sqlite3.Error as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def log_ingestion(self, 
                     query: str,
                     source: str,
                     papers_found: int,
                     papers_added: int,
                     papers_skipped: int,
                     success: bool = True,
                     error_message: str = None):
        """
        Log an ingestion event.
        
        Args:
            query: Search query used
            source: Data source
            papers_found: Number of papers found
            papers_added: Number of papers added
            papers_skipped: Number of papers skipped (duplicates)
            success: Whether the ingestion was successful
            error_message: Error message if failed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO ingestion_log (
                        query, source, papers_found, papers_added, papers_skipped,
                        success, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (query, source, papers_found, papers_added, papers_skipped,
                      success, error_message))
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Failed to log ingestion: {e}")
    
    def close(self):
        """Close database connection (not needed with context managers, but provided for completeness)."""
        pass


if __name__ == "__main__":
    # Example usage
    db = PaperDatabase("test_papers.db")
    
    # Add a test paper
    test_paper = {
        'id': 'test_001',
        'title': 'Test Paper',
        'authors': ['John Doe', 'Jane Smith'],
        'abstract': 'This is a test paper abstract.',
        'source': 'arxiv',
        'hash': 'test_hash_123'
    }
    
    success = db.add_paper(test_paper)
    print(f"Paper added: {success}")
    
    # Get stats
    stats = db.get_stats()
    print(f"Database stats: {stats}")
    
    # Clean up test database
    os.remove("test_papers.db")
