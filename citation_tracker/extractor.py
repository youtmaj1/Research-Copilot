"""
Citation Extractor Module

This module extracts citations from research papers' reference sections using
various parsing techniques including regex patterns, DOI detection, and arXiv ID extraction.

Key Features:
- Parse reference sections from PDF text or structured text
- Extract DOIs, arXiv IDs, and paper titles
- Normalize extracted strings for better matching
- Handle various citation formats and styles
- Support for both structured and unstructured reference data

Classes:
    ExtractedCitation: Data class representing an extracted citation
    CitationExtractor: Main class for extracting citations from papers
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not available. PDF extraction will be disabled.")
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedCitation:
    """
    Represents an extracted citation from a paper's reference section.
    
    Attributes:
        raw_text (str): Original citation text as found in the paper
        doi (Optional[str]): Extracted DOI if found
        arxiv_id (Optional[str]): Extracted arXiv ID if found
        title (Optional[str]): Extracted paper title if found
        authors (Optional[str]): Extracted author names if found
        year (Optional[int]): Publication year if found
        venue (Optional[str]): Publication venue if found
        normalized_text (str): Normalized version of raw_text for matching
        confidence (float): Confidence score for the extraction (0.0-1.0)
        source_paper_id (str): ID of the paper this citation was found in
    """
    raw_text: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    normalized_text: str = ""
    confidence: float = 0.0
    source_paper_id: str = ""

class CitationExtractor:
    """
    Extracts citations from research papers using multiple parsing strategies.
    
    This class implements various techniques to identify and extract citation
    information from reference sections of research papers, including:
    - DOI pattern matching
    - arXiv ID detection
    - Title and author extraction
    - Year and venue identification
    - Text normalization for better matching
    """
    
    def __init__(self):
        """Initialize the CitationExtractor with regex patterns."""
        self.doi_pattern = re.compile(
            r'(?:doi:?\s*|DOI:?\s*|https?://(?:dx\.)?doi\.org/)'
            r'(10\.\d{4,}[^\s\]]+)',
            re.IGNORECASE
        )
        
        self.arxiv_pattern = re.compile(
            r'(?:arXiv:?\s*|arxiv:?\s*)'
            r'(\d{4}\.\d{4,5}(?:v\d+)?)',
            re.IGNORECASE
        )
        
        # Pattern for old arXiv format
        self.arxiv_old_pattern = re.compile(
            r'(?:arXiv:?\s*|arxiv:?\s*)'
            r'([a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)',
            re.IGNORECASE
        )
        
        # Year pattern (4 digits, likely between 1900-2030)
        self.year_pattern = re.compile(r'\b(19[5-9]\d|20[0-3]\d)\b')
        
        # Common venue patterns
        self.venue_patterns = [
            re.compile(r'\b(ICML|NIPS|NeurIPS|ICLR|AAAI|IJCAI|CVPR|ICCV|ECCV)\b', re.IGNORECASE),
            re.compile(r'\b(Nature|Science|Cell|PNAS)\b', re.IGNORECASE),
            re.compile(r'\b(IEEE|ACM|Springer|Elsevier)\b', re.IGNORECASE),
        ]
        
        # Patterns to identify reference sections
        self.reference_section_patterns = [
            re.compile(r'^References\s*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^Bibliography\s*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^Works Cited\s*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\d+\.\s*References\s*$', re.MULTILINE | re.IGNORECASE),
        ]
        
        logger.info("CitationExtractor initialized with regex patterns")
    
    def extract_citations_from_text(self, text: str, source_paper_id: str = "") -> List[ExtractedCitation]:
        """
        Extract citations from raw text, typically from a reference section.
        
        Args:
            text (str): Raw text containing citations
            source_paper_id (str): ID of the source paper
            
        Returns:
            List[ExtractedCitation]: List of extracted citations
        """
        logger.info(f"Extracting citations from text for paper {source_paper_id}")
        
        # First, try to identify and extract the reference section
        reference_text = self._extract_reference_section(text)
        if not reference_text:
            reference_text = text
            logger.warning("Could not identify reference section, using full text")
        
        # Split into individual citation entries
        citation_entries = self._split_into_citations(reference_text)
        logger.info(f"Found {len(citation_entries)} potential citation entries")
        
        # Extract information from each citation
        extracted_citations = []
        for i, entry in enumerate(citation_entries):
            citation = self._parse_single_citation(entry, source_paper_id)
            if citation and self._is_valid_citation(citation):
                extracted_citations.append(citation)
        
        logger.info(f"Successfully extracted {len(extracted_citations)} citations")
        return extracted_citations
    
    def extract_citations_from_pdf(self, pdf_path: str, source_paper_id: str = "") -> List[ExtractedCitation]:
        """
        Extract citations from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            source_paper_id (str): ID of the source paper
            
        Returns:
            List[ExtractedCitation]: List of extracted citations
        """
        logger.info(f"Extracting citations from PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            text = self._extract_text_from_pdf(pdf_path)
            if not text:
                logger.error(f"Could not extract text from PDF: {pdf_path}")
                return []
            
            # Use paper filename as ID if not provided
            if not source_paper_id:
                source_paper_id = Path(pdf_path).stem
            
            return self.extract_citations_from_text(text, source_paper_id)
            
        except Exception as e:
            logger.error(f"Error extracting citations from PDF {pdf_path}: {e}")
            return []
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        if not HAS_PYMUPDF:
            logger.error("PyMuPDF not available. Cannot extract text from PDF.")
            return ""
            
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text()
            
            doc.close()
            return text
            
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def _extract_reference_section(self, text: str) -> str:
        """
        Extract the reference section from the full paper text.
        
        Args:
            text (str): Full paper text
            
        Returns:
            str: Reference section text, or empty string if not found
        """
        for pattern in self.reference_section_patterns:
            match = pattern.search(text)
            if match:
                # Extract everything after the reference section header
                ref_start = match.end()
                
                # Try to find the end of references (acknowledgments, appendix, etc.)
                end_patterns = [
                    re.compile(r'\n\s*(?:Acknowledgments?|Appendix|Index)\s*\n', re.IGNORECASE),
                    re.compile(r'\n\s*\d+\.\s*(?:Acknowledgments?|Appendix)\s*\n', re.IGNORECASE)
                ]
                
                ref_end = len(text)
                for end_pattern in end_patterns:
                    end_match = end_pattern.search(text, ref_start)
                    if end_match:
                        ref_end = end_match.start()
                        break
                
                return text[ref_start:ref_end].strip()
        
        return ""
    
    def _split_into_citations(self, reference_text: str) -> List[str]:
        """
        Split reference section into individual citation entries.
        
        Args:
            reference_text (str): Reference section text
            
        Returns:
            List[str]: Individual citation entries
        """
        # Common patterns for citation numbering/bullets
        patterns = [
            r'\n\s*\[\d+\]',  # [1], [2], etc.
            r'\n\s*\(\d+\)',  # (1), (2), etc.
            r'\n\s*\d+\.',    # 1., 2., etc.
        ]
        
        # Try each pattern
        for pattern in patterns:
            splits = re.split(pattern, reference_text)
            if len(splits) > 3:  # If we get reasonable number of splits
                # Remove empty entries and clean up
                citations = [cite.strip() for cite in splits if cite.strip()]
                return citations
        
        # Fallback: split by double newlines
        citations = [cite.strip() for cite in reference_text.split('\n\n') if cite.strip()]
        
        # If still not good, split by single newlines and try to group
        if len(citations) < 2:
            lines = [line.strip() for line in reference_text.split('\n') if line.strip()]
            citations = self._group_citation_lines(lines)
        
        return citations
    
    def _group_citation_lines(self, lines: List[str]) -> List[str]:
        """
        Group lines that belong to the same citation.
        
        Args:
            lines (List[str]): Individual lines from reference section
            
        Returns:
            List[str]: Grouped citation entries
        """
        citations = []
        current_citation = []
        
        for line in lines:
            # Check if this line starts a new citation
            if (re.match(r'^\[\d+\]', line) or 
                re.match(r'^\(\d+\)', line) or 
                re.match(r'^\d+\.', line)):
                
                # Save previous citation if exists
                if current_citation:
                    citations.append(' '.join(current_citation))
                    current_citation = []
            
            current_citation.append(line)
        
        # Add last citation
        if current_citation:
            citations.append(' '.join(current_citation))
        
        return citations
    
    def _parse_single_citation(self, citation_text: str, source_paper_id: str) -> Optional[ExtractedCitation]:
        """
        Parse a single citation entry to extract structured information.
        
        Args:
            citation_text (str): Raw citation text
            source_paper_id (str): ID of the source paper
            
        Returns:
            Optional[ExtractedCitation]: Parsed citation or None if invalid
        """
        if not citation_text or len(citation_text.strip()) < 10:
            return None
        
        citation = ExtractedCitation(
            raw_text=citation_text,
            source_paper_id=source_paper_id
        )
        
        # Extract DOI
        doi_match = self.doi_pattern.search(citation_text)
        if doi_match:
            citation.doi = doi_match.group(1)
            citation.confidence += 0.3
        
        # Extract arXiv ID (new format)
        arxiv_match = self.arxiv_pattern.search(citation_text)
        if arxiv_match:
            citation.arxiv_id = arxiv_match.group(1)
            citation.confidence += 0.3
        
        # Extract arXiv ID (old format)
        if not citation.arxiv_id:
            arxiv_old_match = self.arxiv_old_pattern.search(citation_text)
            if arxiv_old_match:
                citation.arxiv_id = arxiv_old_match.group(1)
                citation.confidence += 0.3
        
        # Extract year
        year_match = self.year_pattern.search(citation_text)
        if year_match:
            citation.year = int(year_match.group(1))
            citation.confidence += 0.1
        
        # Extract venue
        for venue_pattern in self.venue_patterns:
            venue_match = venue_pattern.search(citation_text)
            if venue_match:
                citation.venue = venue_match.group(1)
                citation.confidence += 0.1
                break
        
        # Extract title (heuristic approach)
        citation.title = self._extract_title(citation_text)
        if citation.title:
            citation.confidence += 0.2
        
        # Extract authors (heuristic approach)
        citation.authors = self._extract_authors(citation_text)
        if citation.authors:
            citation.confidence += 0.1
        
        # Normalize text
        citation.normalized_text = self._normalize_text(citation_text)
        
        return citation
    
    def _extract_title(self, citation_text: str) -> Optional[str]:
        """
        Extract paper title from citation text using heuristics.
        
        Args:
            citation_text (str): Raw citation text
            
        Returns:
            Optional[str]: Extracted title or None
        """
        # Remove citation numbering
        text = re.sub(r'^\s*[\[\(]?\d+[\]\)]?\s*', '', citation_text)
        
        # Common patterns for titles (often in quotes or between specific punctuation)
        title_patterns = [
            r'"([^"]+)"',  # Title in double quotes
            r"'([^']+)'",  # Title in single quotes
            r'[^.]+?\.\s*([^.]+?)\.\s*(?:In|Proceedings|Journal|arXiv)',  # Title between dots
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) > 10 and len(title) < 200:  # Reasonable title length
                    return self._clean_title(title)
        
        # Fallback: try to extract first reasonable sentence
        sentences = text.split('.')
        for sentence in sentences[:3]:  # Check first 3 sentences
            cleaned = sentence.strip()
            if (len(cleaned) > 15 and len(cleaned) < 150 and 
                not re.match(r'^[A-Z]\.\s*[A-Z]\.', cleaned)):  # Not author initials
                return self._clean_title(cleaned)
        
        return None
    
    def _clean_title(self, title: str) -> str:
        """Clean extracted title text."""
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove common prefixes/suffixes
        title = re.sub(r'^(Title:|Paper:|Article:)\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*\([^)]*\)$', '', title)  # Remove trailing parentheses
        
        return title
    
    def _extract_authors(self, citation_text: str) -> Optional[str]:
        """
        Extract author names from citation text using heuristics.
        
        Args:
            citation_text (str): Raw citation text
            
        Returns:
            Optional[str]: Extracted authors or None
        """
        # Remove citation numbering
        text = re.sub(r'^\s*[\[\(]?\d+[\]\)]?\s*', '', citation_text)
        
        # Pattern for author names (usually at the beginning)
        # Look for patterns like "A. Smith, B. Jones, and C. Brown" or "Smith, A., Jones, B."
        author_patterns = [
            r'^([A-Z][a-z]*(?:\s+[A-Z]\.)?\s*,\s*[A-Z]\.(?:\s*[A-Z]\.)*(?:\s*,\s*[^.]+?)*?)(?:\s*[\.\(])',
            r'^([A-Z]\.(?:\s*[A-Z]\.)*\s+[A-Z][a-z]+(?:\s*,\s*[^.]+?)*?)(?:\s*[\.\(])',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, text)
            if match:
                authors = match.group(1).strip()
                if len(authors) > 3 and len(authors) < 100:
                    return self._clean_authors(authors)
        
        return None
    
    def _clean_authors(self, authors: str) -> str:
        """Clean extracted author text."""
        # Remove extra whitespace
        authors = re.sub(r'\s+', ' ', authors).strip()
        
        # Remove trailing punctuation
        authors = re.sub(r'[,\.\s]+$', '', authors)
        
        return authors
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize citation text for better matching.
        
        Args:
            text (str): Raw citation text
            
        Returns:
            str: Normalized text
        """
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove punctuation except essential ones
        normalized = re.sub(r'[^\w\s\-\.\:]', ' ', normalized)
        
        # Remove common citation artifacts
        normalized = re.sub(r'^\s*[\[\(]?\d+[\]\)]?\s*', '', normalized)
        
        # Remove URLs
        normalized = re.sub(r'https?://[^\s]+', '', normalized)
        
        # Normalize DOI format
        normalized = re.sub(r'doi:?\s*', 'doi:', normalized)
        
        # Remove extra spaces again
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _is_valid_citation(self, citation: ExtractedCitation) -> bool:
        """
        Check if the extracted citation is valid and worth keeping.
        
        Args:
            citation (ExtractedCitation): Citation to validate
            
        Returns:
            bool: True if citation is valid
        """
        # Must have some identifying information
        if not (citation.doi or citation.arxiv_id or citation.title):
            return False
        
        # Raw text must be reasonable length
        if len(citation.raw_text.strip()) < 20:
            return False
        
        # Confidence threshold
        if citation.confidence < 0.1:
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the extractor's performance.
        
        Returns:
            Dict[str, any]: Statistics dictionary
        """
        return {
            "patterns_loaded": {
                "doi_pattern": bool(self.doi_pattern),
                "arxiv_pattern": bool(self.arxiv_pattern),
                "year_pattern": bool(self.year_pattern),
                "venue_patterns": len(self.venue_patterns),
                "reference_patterns": len(self.reference_section_patterns)
            },
            "version": "1.0.0"
        }

def normalize_citation_text(text: str) -> str:
    """
    Utility function to normalize citation text.
    
    Args:
        text (str): Raw citation text
        
    Returns:
        str: Normalized text
    """
    extractor = CitationExtractor()
    return extractor._normalize_text(text)

def extract_dois_from_text(text: str) -> List[str]:
    """
    Utility function to extract DOIs from text.
    
    Args:
        text (str): Text containing potential DOIs
        
    Returns:
        List[str]: List of extracted DOIs
    """
    extractor = CitationExtractor()
    dois = []
    for match in extractor.doi_pattern.finditer(text):
        dois.append(match.group(1))
    return dois

def extract_arxiv_ids_from_text(text: str) -> List[str]:
    """
    Utility function to extract arXiv IDs from text.
    
    Args:
        text (str): Text containing potential arXiv IDs
        
    Returns:
        List[str]: List of extracted arXiv IDs
    """
    extractor = CitationExtractor()
    arxiv_ids = []
    
    # New format
    for match in extractor.arxiv_pattern.finditer(text):
        arxiv_ids.append(match.group(1))
    
    # Old format
    for match in extractor.arxiv_old_pattern.finditer(text):
        arxiv_ids.append(match.group(1))
    
    return arxiv_ids
