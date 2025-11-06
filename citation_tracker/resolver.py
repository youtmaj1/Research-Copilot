"""
Citation Resolver Module

This module resolves extracted citations by matching them to papers in the database.
It uses various matching strategies including exact DOI/arXiv matching, fuzzy title matching,
and author-based matching to identify papers that have been cited.

Key Features:
- Exact matching via DOI and arXiv ID
- Fuzzy title matching using string similarity
- Author and year-based matching
- Confidence scoring for matches
- Support for ambiguous citation resolution
- Integration with papers.db database

Classes:
    CitationMatch: Data class representing a citation match
    CitationResolver: Main class for resolving citations to database papers
"""

import logging
import sqlite3
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
import re
try:
    from fuzzywuzzy import fuzz, process
    HAS_FUZZYWUZZY = True
except ImportError:
    HAS_FUZZYWUZZY = False
    print("Warning: fuzzywuzzy not available. Fuzzy matching will be disabled.")

try:
    from .extractor import ExtractedCitation
except ImportError:
    from extractor import ExtractedCitation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CitationMatch:
    """
    Represents a match between an extracted citation and a paper in the database.
    
    Attributes:
        paper_id (str): ID of the matched paper in the database
        citation (ExtractedCitation): The original extracted citation
        match_type (str): Type of match (doi, arxiv, title, author, etc.)
        confidence (float): Confidence score for the match (0.0-1.0)
        paper_title (str): Title of the matched paper
        paper_authors (str): Authors of the matched paper
        paper_year (Optional[int]): Year of the matched paper
        paper_doi (Optional[str]): DOI of the matched paper
        paper_arxiv_id (Optional[str]): arXiv ID of the matched paper
        ambiguous (bool): Whether this match is ambiguous (multiple possible matches)
        alternative_matches (List[str]): Alternative paper IDs for ambiguous matches
    """
    paper_id: str
    citation: ExtractedCitation
    match_type: str
    confidence: float
    paper_title: str = ""
    paper_authors: str = ""
    paper_year: Optional[int] = None
    paper_doi: Optional[str] = None
    paper_arxiv_id: Optional[str] = None
    ambiguous: bool = False
    alternative_matches: List[str] = None
    
    def __post_init__(self):
        if self.alternative_matches is None:
            self.alternative_matches = []

class CitationResolver:
    """
    Resolves extracted citations by matching them to papers in the research database.
    
    This class implements various matching strategies to identify which papers
    in the database correspond to extracted citations, including:
    - Exact DOI matching
    - Exact arXiv ID matching
    - Fuzzy title matching
    - Author and year matching
    - Combined heuristic matching
    """
    
    def __init__(self, db_path: str = "papers.db"):
        """
        Initialize the CitationResolver with database connection.
        
        Args:
            db_path (str): Path to the papers database
        """
        self.db_path = db_path
        self.connection = None
        self.title_similarity_threshold = 0.8
        self.author_similarity_threshold = 0.7
        self.fuzzy_threshold = 85  # For fuzzywuzzy matching
        
        # Initialize database connection
        self._connect_to_database()
        
        logger.info(f"CitationResolver initialized with database: {db_path}")
    
    def _connect_to_database(self):
        """Establish connection to the papers database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            
            # Verify database structure
            self._verify_database_structure()
            
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            self.connection = None
    
    def _verify_database_structure(self):
        """Verify that the database has the expected structure."""
        try:
            cursor = self.connection.cursor()
            
            # Check if papers table exists with required columns
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='papers'
            """)
            
            if not cursor.fetchone():
                logger.warning("Papers table not found in database")
                return
            
            # Check table schema
            cursor.execute("PRAGMA table_info(papers)")
            columns = {row[1] for row in cursor.fetchall()}
            
            required_columns = {'id', 'title', 'authors', 'doi', 'arxiv_id', 'year'}
            missing_columns = required_columns - columns
            
            if missing_columns:
                logger.warning(f"Missing required columns in papers table: {missing_columns}")
            else:
                logger.info("Database structure verified successfully")
                
        except sqlite3.Error as e:
            logger.error(f"Database structure verification error: {e}")
    
    def resolve_citations(self, citations: List[ExtractedCitation]) -> List[CitationMatch]:
        """
        Resolve a list of extracted citations to database papers.
        
        Args:
            citations (List[ExtractedCitation]): Citations to resolve
            
        Returns:
            List[CitationMatch]: List of resolved citation matches
        """
        if not self.connection:
            logger.error("No database connection available")
            return []
        
        logger.info(f"Resolving {len(citations)} citations")
        matches = []
        
        for citation in citations:
            match = self.resolve_single_citation(citation)
            if match:
                matches.append(match)
        
        logger.info(f"Successfully resolved {len(matches)} citations")
        return matches
    
    def resolve_single_citation(self, citation: ExtractedCitation) -> Optional[CitationMatch]:
        """
        Resolve a single citation to a database paper.
        
        Args:
            citation (ExtractedCitation): Citation to resolve
            
        Returns:
            Optional[CitationMatch]: Citation match or None if not found
        """
        if not self.connection:
            return None
        
        # Try exact matching strategies first (highest confidence)
        match = self._try_doi_match(citation)
        if match:
            return match
        
        match = self._try_arxiv_match(citation)
        if match:
            return match
        
        # Try fuzzy matching strategies
        match = self._try_title_match(citation)
        if match:
            return match
        
        match = self._try_author_year_match(citation)
        if match:
            return match
        
        # Try combined heuristic matching
        match = self._try_combined_match(citation)
        if match:
            return match
        
        logger.debug(f"Could not resolve citation: {citation.raw_text[:100]}...")
        return None
    
    def _try_doi_match(self, citation: ExtractedCitation) -> Optional[CitationMatch]:
        """Try to match citation by DOI."""
        if not citation.doi:
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM papers 
                WHERE LOWER(doi) = LOWER(?) AND doi IS NOT NULL
            """, (citation.doi,))
            
            row = cursor.fetchone()
            if row:
                match = CitationMatch(
                    paper_id=row['id'],
                    citation=citation,
                    match_type='doi',
                    confidence=0.95,
                    paper_title=row['title'] or '',
                    paper_authors=row['authors'] or '',
                    paper_year=row['year'],
                    paper_doi=row['doi'],
                    paper_arxiv_id=row['arxiv_id']
                )
                logger.debug(f"DOI match found: {citation.doi} -> {row['id']}")
                return match
                
        except sqlite3.Error as e:
            logger.error(f"DOI matching error: {e}")
        
        return None
    
    def _try_arxiv_match(self, citation: ExtractedCitation) -> Optional[CitationMatch]:
        """Try to match citation by arXiv ID."""
        if not citation.arxiv_id:
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT * FROM papers 
                WHERE LOWER(arxiv_id) = LOWER(?) AND arxiv_id IS NOT NULL
            """, (citation.arxiv_id,))
            
            row = cursor.fetchone()
            if row:
                match = CitationMatch(
                    paper_id=row['id'],
                    citation=citation,
                    match_type='arxiv',
                    confidence=0.95,
                    paper_title=row['title'] or '',
                    paper_authors=row['authors'] or '',
                    paper_year=row['year'],
                    paper_doi=row['doi'],
                    paper_arxiv_id=row['arxiv_id']
                )
                logger.debug(f"arXiv match found: {citation.arxiv_id} -> {row['id']}")
                return match
                
        except sqlite3.Error as e:
            logger.error(f"arXiv matching error: {e}")
        
        return None
    
    def _try_title_match(self, citation: ExtractedCitation) -> Optional[CitationMatch]:
        """Try to match citation by title using fuzzy matching."""
        if not citation.title or len(citation.title) < 10:
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT id, title, authors, year, doi, arxiv_id 
                FROM papers 
                WHERE title IS NOT NULL AND LENGTH(title) > 5
            """)
            
            rows = cursor.fetchall()
            best_match = None
            best_score = 0
            alternatives = []
            
            for row in rows:
                if not row['title']:
                    continue
                
                # Calculate similarity score
                similarity = self._calculate_title_similarity(citation.title, row['title'])
                
                if similarity > self.title_similarity_threshold:
                    if similarity > best_score:
                        if best_match:
                            alternatives.append(best_match[0])
                        best_match = (row, similarity)
                        best_score = similarity
                    elif similarity > 0.75:  # Also track high-scoring alternatives
                        alternatives.append(row['id'])
            
            if best_match:
                row, similarity = best_match
                match = CitationMatch(
                    paper_id=row['id'],
                    citation=citation,
                    match_type='title',
                    confidence=similarity * 0.9,  # Slightly reduce confidence for fuzzy matches
                    paper_title=row['title'] or '',
                    paper_authors=row['authors'] or '',
                    paper_year=row['year'],
                    paper_doi=row['doi'],
                    paper_arxiv_id=row['arxiv_id'],
                    ambiguous=len(alternatives) > 0,
                    alternative_matches=alternatives
                )
                logger.debug(f"Title match found: {citation.title[:50]}... -> {row['id']} (score: {similarity:.3f})")
                return match
                
        except sqlite3.Error as e:
            logger.error(f"Title matching error: {e}")
        
        return None
    
    def _try_author_year_match(self, citation: ExtractedCitation) -> Optional[CitationMatch]:
        """Try to match citation by authors and year."""
        if not citation.authors or not citation.year:
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT id, title, authors, year, doi, arxiv_id 
                FROM papers 
                WHERE year = ? AND authors IS NOT NULL
            """, (citation.year,))
            
            rows = cursor.fetchall()
            best_match = None
            best_score = 0
            
            for row in rows:
                if not row['authors']:
                    continue
                
                # Calculate author similarity
                author_similarity = self._calculate_author_similarity(citation.authors, row['authors'])
                
                if author_similarity > self.author_similarity_threshold:
                    # Boost score if title also matches
                    title_boost = 0
                    if citation.title and row['title']:
                        title_sim = self._calculate_title_similarity(citation.title, row['title'])
                        if title_sim > 0.6:
                            title_boost = 0.2
                    
                    total_score = author_similarity + title_boost
                    
                    if total_score > best_score:
                        best_match = (row, total_score)
                        best_score = total_score
            
            if best_match:
                row, score = best_match
                match = CitationMatch(
                    paper_id=row['id'],
                    citation=citation,
                    match_type='author_year',
                    confidence=min(score * 0.8, 0.9),  # Cap confidence for heuristic matches
                    paper_title=row['title'] or '',
                    paper_authors=row['authors'] or '',
                    paper_year=row['year'],
                    paper_doi=row['doi'],
                    paper_arxiv_id=row['arxiv_id']
                )
                logger.debug(f"Author-year match found: {citation.authors[:30]}... -> {row['id']} (score: {score:.3f})")
                return match
                
        except sqlite3.Error as e:
            logger.error(f"Author-year matching error: {e}")
        
        return None
    
    def _try_combined_match(self, citation: ExtractedCitation) -> Optional[CitationMatch]:
        """Try combined heuristic matching using multiple factors."""
        try:
            cursor = self.connection.cursor()
            
            # Build query conditions based on available citation data
            conditions = []
            params = []
            
            if citation.year:
                conditions.append("(year = ? OR year BETWEEN ? AND ?)")
                params.extend([citation.year, citation.year - 1, citation.year + 1])
            
            if citation.venue:
                conditions.append("(LOWER(venue) LIKE ? OR LOWER(title) LIKE ?)")
                venue_pattern = f"%{citation.venue.lower()}%"
                params.extend([venue_pattern, venue_pattern])
            
            if not conditions:
                return None
            
            query = f"""
                SELECT id, title, authors, year, doi, arxiv_id, venue
                FROM papers 
                WHERE {' AND '.join(conditions)} AND title IS NOT NULL
            """
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return None
            
            # Score each potential match
            best_match = None
            best_score = 0
            
            for row in rows:
                score = self._calculate_combined_score(citation, row)
                
                if score > 0.6 and score > best_score:  # Minimum threshold for combined matching
                    best_match = (row, score)
                    best_score = score
            
            if best_match:
                row, score = best_match
                match = CitationMatch(
                    paper_id=row['id'],
                    citation=citation,
                    match_type='combined',
                    confidence=min(score * 0.7, 0.8),  # Conservative confidence for heuristic matches
                    paper_title=row['title'] or '',
                    paper_authors=row['authors'] or '',
                    paper_year=row['year'],
                    paper_doi=row['doi'],
                    paper_arxiv_id=row['arxiv_id']
                )
                logger.debug(f"Combined match found: -> {row['id']} (score: {score:.3f})")
                return match
                
        except sqlite3.Error as e:
            logger.error(f"Combined matching error: {e}")
        
        return None
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles."""
        if not title1 or not title2:
            return 0.0
        
        # Normalize titles
        t1 = self._normalize_title(title1)
        t2 = self._normalize_title(title2)
        
        # Use multiple similarity metrics
        sequence_similarity = SequenceMatcher(None, t1, t2).ratio()
        
        if HAS_FUZZYWUZZY:
            fuzzy_similarity = fuzz.ratio(t1, t2) / 100.0
            token_similarity = fuzz.token_sort_ratio(t1, t2) / 100.0
            # Return weighted average
            return (sequence_similarity * 0.3 + fuzzy_similarity * 0.4 + token_similarity * 0.3)
        else:
            # Fall back to sequence matcher only
            return sequence_similarity
    
    def _calculate_author_similarity(self, authors1: str, authors2: str) -> float:
        """Calculate similarity between author strings."""
        if not authors1 or not authors2:
            return 0.0
        
        # Normalize author strings
        a1 = self._normalize_authors(authors1)
        a2 = self._normalize_authors(authors2)
        
        # Extract individual author names
        authors1_list = self._extract_author_names(a1)
        authors2_list = self._extract_author_names(a2)
        
        if not authors1_list or not authors2_list:
            if HAS_FUZZYWUZZY:
                return fuzz.ratio(a1, a2) / 100.0
            else:
                return SequenceMatcher(None, a1, a2).ratio()
        
        # Calculate overlap in author names
        matches = 0
        for author1 in authors1_list:
            for author2 in authors2_list:
                if HAS_FUZZYWUZZY:
                    similarity = fuzz.ratio(author1, author2)
                else:
                    similarity = SequenceMatcher(None, author1, author2).ratio() * 100
                
                if similarity > 80:  # High threshold for name matching
                    matches += 1
                    break
        
        # Return proportion of matched authors
        return matches / max(len(authors1_list), len(authors2_list))
    
    def _calculate_combined_score(self, citation: ExtractedCitation, paper_row) -> float:
        """Calculate combined similarity score using multiple factors."""
        score = 0.0
        factors = 0
        
        # Title similarity
        if citation.title and paper_row['title']:
            title_sim = self._calculate_title_similarity(citation.title, paper_row['title'])
            score += title_sim * 0.4
            factors += 0.4
        
        # Author similarity
        if citation.authors and paper_row['authors']:
            author_sim = self._calculate_author_similarity(citation.authors, paper_row['authors'])
            score += author_sim * 0.3
            factors += 0.3
        
        # Year match
        if citation.year and paper_row['year']:
            if citation.year == paper_row['year']:
                score += 0.2
            elif abs(citation.year - paper_row['year']) <= 1:
                score += 0.1
            factors += 0.2
        
        # Venue match
        if citation.venue and paper_row.get('venue'):
            if HAS_FUZZYWUZZY:
                venue_sim = fuzz.ratio(citation.venue.lower(), paper_row['venue'].lower()) / 100.0
            else:
                venue_sim = SequenceMatcher(None, citation.venue.lower(), paper_row['venue'].lower()).ratio()
            if venue_sim > 0.7:
                score += venue_sim * 0.1
            factors += 0.1
        
        # Normalize by the total weight of factors considered
        if factors > 0:
            score = score / factors
        
        return score
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison."""
        # Convert to lowercase
        normalized = title.lower()
        
        # Remove common prefixes/suffixes
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
        normalized = re.sub(r'\s*\([^)]*\)$', '', normalized)  # Remove trailing parentheses
        
        # Remove punctuation except spaces and hyphens
        normalized = re.sub(r'[^\w\s\-]', ' ', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _normalize_authors(self, authors: str) -> str:
        """Normalize author string for comparison."""
        # Convert to lowercase
        normalized = authors.lower()
        
        # Remove common prefixes
        normalized = re.sub(r'\bet\s+al\.?', 'et al', normalized)
        
        # Normalize punctuation
        normalized = re.sub(r'[,\.]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _extract_author_names(self, authors: str) -> List[str]:
        """Extract individual author names from author string."""
        # Split by common separators
        names = re.split(r'\s+and\s+|,\s*', authors)
        
        # Clean and filter names
        cleaned_names = []
        for name in names:
            name = name.strip()
            if len(name) > 2 and name not in ['et al', 'others']:
                # Normalize initials (A. B. Smith -> a b smith)
                name = re.sub(r'([A-Z])\.\s*', r'\1 ', name).strip()
                cleaned_names.append(name)
        
        return cleaned_names
    
    def get_unresolved_citations(self, citations: List[ExtractedCitation]) -> List[ExtractedCitation]:
        """
        Get citations that could not be resolved to database papers.
        
        Args:
            citations (List[ExtractedCitation]): Original citations
            
        Returns:
            List[ExtractedCitation]: Unresolved citations
        """
        matches = self.resolve_citations(citations)
        resolved_citations = {match.citation for match in matches}
        
        unresolved = [citation for citation in citations if citation not in resolved_citations]
        logger.info(f"Found {len(unresolved)} unresolved citations out of {len(citations)}")
        
        return unresolved
    
    def get_resolution_statistics(self, citations: List[ExtractedCitation]) -> Dict[str, any]:
        """
        Get statistics about citation resolution performance.
        
        Args:
            citations (List[ExtractedCitation]): Citations to analyze
            
        Returns:
            Dict[str, any]: Resolution statistics
        """
        if not citations:
            return {"error": "No citations provided"}
        
        matches = self.resolve_citations(citations)
        
        # Count matches by type
        match_types = {}
        confidence_scores = []
        ambiguous_count = 0
        
        for match in matches:
            match_types[match.match_type] = match_types.get(match.match_type, 0) + 1
            confidence_scores.append(match.confidence)
            if match.ambiguous:
                ambiguous_count += 1
        
        return {
            "total_citations": len(citations),
            "resolved_citations": len(matches),
            "resolution_rate": len(matches) / len(citations),
            "match_types": match_types,
            "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "ambiguous_matches": ambiguous_count,
            "unresolved_citations": len(citations) - len(matches)
        }
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

def resolve_citations_from_file(citations_file: str, db_path: str = "papers.db") -> List[CitationMatch]:
    """
    Utility function to resolve citations from a file.
    
    Args:
        citations_file (str): Path to file containing citation data
        db_path (str): Path to papers database
        
    Returns:
        List[CitationMatch]: Resolved citation matches
    """
    # This would need implementation based on file format
    # For now, just return empty list
    logger.warning("File-based citation resolution not implemented yet")
    return []
