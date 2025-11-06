"""
Citation Extractor

Parses reference sections from research papers and extracts citation information.
Matches DOIs, arXiv IDs, and titles to identify relationships between papers.
"""

import re
import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from collections import defaultdict

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a single citation extracted from a paper."""
    raw_text: str  # Original reference text
    title: Optional[str] = None
    authors: List[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    confidence: float = 0.0  # Confidence in extraction accuracy
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []


@dataclass
class CitationMatch:
    """Represents a matched citation between papers."""
    source_paper_id: str  # ID of paper containing the citation
    cited_paper_id: str   # ID of paper being cited
    citation: Citation    # Citation details
    match_type: str       # 'doi', 'arxiv', 'title', 'fuzzy'
    confidence: float     # Match confidence score


class CitationExtractor:
    """
    Extracts citations from research papers.
    
    Supports multiple extraction methods:
    - DOI-based matching
    - arXiv ID matching  
    - Title-based matching with fuzzy matching
    """
    
    def __init__(self):
        self.doi_pattern = re.compile(
            r'10\.\d{4,}/[^\s\]]+',
            re.IGNORECASE
        )
        
        self.arxiv_pattern = re.compile(
            r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)',
            re.IGNORECASE
        )
        
        # Patterns for common reference formats
        self.reference_patterns = [
            # IEEE format: [1] Author, "Title," Journal, year.
            re.compile(r'\[(\d+)\]\s*([^"]+)"([^"]+)"\s*([^,]+),?\s*(\d{4})', re.IGNORECASE),
            
            # APA format: Author (year). Title. Journal.
            re.compile(r'([^.]+)\s*\((\d{4})\)\.\s*([^.]+)\.\s*([^.]+)', re.IGNORECASE),
            
            # Basic format: Author, Title, Journal, year
            re.compile(r'([^,]+),\s*([^,]+),\s*([^,]+),\s*(\d{4})', re.IGNORECASE)
        ]
        
        # Common section headers for references
        self.reference_headers = [
            'references',
            'bibliography', 
            'works cited',
            'cited works',
            'literature cited'
        ]
        
        logger.info("Citation extractor initialized")
    
    def extract_from_pdf(self, pdf_path: str) -> List[Citation]:
        """
        Extract citations from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted citations
        """
        if fitz is None:
            logger.error("PyMuPDF not available for PDF extraction")
            return []
        
        try:
            doc = fitz.open(pdf_path)
            
            # Extract all text
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            
            doc.close()
            
            # Find reference section
            ref_text = self._extract_reference_section(full_text)
            
            if not ref_text:
                logger.warning(f"No reference section found in {pdf_path}")
                return []
            
            # Extract individual citations
            citations = self._parse_citations(ref_text)
            
            logger.info(f"Extracted {len(citations)} citations from {pdf_path}")
            return citations
            
        except Exception as e:
            logger.error(f"Failed to extract citations from {pdf_path}: {e}")
            return []
    
    def extract_from_text(self, text: str) -> List[Citation]:
        """
        Extract citations from raw text.
        
        Args:
            text: Raw text containing references
            
        Returns:
            List of extracted citations
        """
        ref_text = self._extract_reference_section(text)
        if not ref_text:
            # If no explicit reference section, try parsing the whole text
            ref_text = text
        
        return self._parse_citations(ref_text)
    
    def _extract_reference_section(self, text: str) -> str:
        """Extract the references section from text."""
        text_lower = text.lower()
        
        # Find reference section start
        ref_start = -1
        for header in self.reference_headers:
            pattern = rf'\b{re.escape(header)}\b'
            match = re.search(pattern, text_lower)
            if match:
                ref_start = match.start()
                break
        
        if ref_start == -1:
            return ""
        
        # Find section end (next major section or end of text)
        end_patterns = [
            r'\bappendix\b',
            r'\backnowledgments?\b',
            r'\bbiograph(y|ies)\b'
        ]
        
        ref_end = len(text)
        for pattern in end_patterns:
            match = re.search(pattern, text_lower[ref_start + 50:])  # Skip 50 chars to avoid matching header
            if match:
                ref_end = ref_start + 50 + match.start()
                break
        
        return text[ref_start:ref_end]
    
    def _parse_citation(self, citation_text: str) -> Citation:
        """
        Parse a single citation text into a Citation object.
        
        Args:
            citation_text: Single citation text
            
        Returns:
            Parsed Citation object
        """
        return self._parse_single_citation(citation_text)
    
    def _parse_citations(self, ref_text: str) -> List[Citation]:
        """Parse individual citations from reference text."""
        citations = []
        
        # Split into individual references
        # Try multiple splitting strategies
        ref_lines = self._split_references(ref_text)
        
        for line in ref_lines:
            line = line.strip()
            if len(line) < 20:  # Skip very short lines
                continue
                
            citation = self._parse_single_citation(line)
            if citation:
                citations.append(citation)
        
        return citations
    
    def _split_references(self, ref_text: str) -> List[str]:
        """Split reference text into individual citations."""
        # Method 1: Split by numbered references [1], [2], etc.
        numbered_refs = re.split(r'\[\d+\]', ref_text)
        if len(numbered_refs) > 2:  # Found numbered references
            return [ref.strip() for ref in numbered_refs[1:] if ref.strip()]
        
        # Method 2: Split by double newlines
        double_newline_refs = re.split(r'\n\s*\n', ref_text)
        if len(double_newline_refs) > 1:
            return [ref.strip() for ref in double_newline_refs if ref.strip()]
        
        # Method 3: Split by single newlines and group
        lines = ref_text.split('\n')
        grouped_refs = []
        current_ref = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # If line starts with author pattern or number, it's likely a new reference
            if (re.match(r'^[A-Z][a-z]+,?\s+[A-Z]', line) or 
                re.match(r'^\[\d+\]', line) or
                re.match(r'^\d+\.', line)):
                
                if current_ref:
                    grouped_refs.append(current_ref)
                current_ref = line
            else:
                current_ref += " " + line
        
        if current_ref:
            grouped_refs.append(current_ref)
        
        return grouped_refs if grouped_refs else [ref_text]
    
    def _parse_single_citation(self, citation_text: str) -> Optional[Citation]:
        """Parse a single citation string."""
        citation = Citation(raw_text=citation_text)
        
        # Extract DOI
        doi_match = self.doi_pattern.search(citation_text)
        if doi_match:
            citation.doi = doi_match.group(0)
            citation.confidence += 0.4
        
        # Extract arXiv ID
        arxiv_match = self.arxiv_pattern.search(citation_text)
        if arxiv_match:
            citation.arxiv_id = arxiv_match.group(1)
            citation.confidence += 0.4
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', citation_text)
        if year_match:
            citation.year = int(year_match.group(0))
            citation.confidence += 0.1
        
        # Try structured parsing with regex patterns
        for pattern in self.reference_patterns:
            match = pattern.search(citation_text)
            if match:
                groups = match.groups()
                if len(groups) >= 3:
                    citation.authors = [groups[0].strip()]
                    citation.title = groups[2].strip() if len(groups) > 2 else groups[1].strip()
                    if len(groups) > 3:
                        citation.journal = groups[3].strip()
                    citation.confidence += 0.3
                break
        
        # Fallback: extract title from quotes
        if not citation.title:
            title_match = re.search(r'"([^"]+)"', citation_text)
            if title_match:
                citation.title = title_match.group(1).strip()
                citation.confidence += 0.2
        
        # Extract URLs
        url_match = re.search(r'https?://[^\s\]]+', citation_text)
        if url_match:
            citation.url = url_match.group(0)
            citation.confidence += 0.1
        
        # Only return citation if we extracted some useful information
        if citation.doi or citation.arxiv_id or citation.title or citation.url:
            return citation
        
        return None
    
    def match_citations_to_papers(
        self, 
        citations: List[Citation], 
        source_paper_id: str,
        known_papers: Dict[str, Dict]
    ) -> List[CitationMatch]:
        """
        Match extracted citations to known papers.
        
        Args:
            citations: List of extracted citations
            source_paper_id: ID of paper containing citations
            known_papers: Dict of paper_id -> paper metadata
            
        Returns:
            List of citation matches
        """
        matches = []
        
        for citation in citations:
            match = self._find_best_match(citation, known_papers)
            if match:
                citation_match = CitationMatch(
                    source_paper_id=source_paper_id,
                    cited_paper_id=match['paper_id'],
                    citation=citation,
                    match_type=match['match_type'],
                    confidence=match['confidence']
                )
                matches.append(citation_match)
        
        logger.info(f"Matched {len(matches)} out of {len(citations)} citations for paper {source_paper_id}")
        return matches
    
    def _find_best_match(
        self, 
        citation: Citation, 
        known_papers: Dict[str, Dict]
    ) -> Optional[Dict]:
        """Find the best matching paper for a citation."""
        best_match = None
        best_score = 0.0
        
        for paper_id, paper_info in known_papers.items():
            score, match_type = self._calculate_match_score(citation, paper_info)
            
            if score > best_score and score > 0.5:  # Minimum confidence threshold
                best_score = score
                best_match = {
                    'paper_id': paper_id,
                    'confidence': score,
                    'match_type': match_type
                }
        
        return best_match
    
    def _calculate_match_score(
        self, 
        citation: Citation, 
        paper_info: Dict
    ) -> Tuple[float, str]:
        """Calculate match score between citation and paper."""
        # DOI match (highest confidence)
        if citation.doi and paper_info.get('doi'):
            if citation.doi.lower() == paper_info['doi'].lower():
                return 1.0, 'doi'
        
        # arXiv ID match (high confidence)
        if citation.arxiv_id and paper_info.get('arxiv_id'):
            if citation.arxiv_id == paper_info['arxiv_id']:
                return 0.95, 'arxiv'
        
        # Title match (medium confidence)
        if citation.title and paper_info.get('title'):
            title_similarity = self._calculate_title_similarity(
                citation.title, 
                paper_info['title']
            )
            if title_similarity > 0.8:
                return title_similarity * 0.8, 'title'
        
        # Author + year match (lower confidence) 
        if (citation.authors and citation.year and 
            paper_info.get('authors') and paper_info.get('published_date')):
            
            author_match = self._check_author_match(citation.authors, paper_info['authors'])
            year_match = self._check_year_match(citation.year, paper_info['published_date'])
            
            if author_match and year_match:
                return 0.6, 'author_year'
        
        return 0.0, 'none'
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles."""
        # Simple token-based similarity
        t1_tokens = set(re.findall(r'\w+', title1.lower()))
        t2_tokens = set(re.findall(r'\w+', title2.lower()))
        
        if not t1_tokens or not t2_tokens:
            return 0.0
        
        intersection = len(t1_tokens & t2_tokens)
        union = len(t1_tokens | t2_tokens)
        
        return intersection / union if union > 0 else 0.0
    
    def _check_author_match(self, citation_authors: List[str], paper_authors: List[str]) -> bool:
        """Check if authors match between citation and paper."""
        if not citation_authors or not paper_authors:
            return False
        
        # Extract last names
        citation_last_names = set()
        for author in citation_authors:
            parts = author.split()
            if parts:
                citation_last_names.add(parts[-1].lower())
        
        paper_last_names = set()
        for author in paper_authors:
            parts = author.split()
            if parts:
                paper_last_names.add(parts[-1].lower())
        
        # Check for overlap
        return len(citation_last_names & paper_last_names) > 0
    
    def _check_year_match(self, citation_year: int, published_date: str) -> bool:
        """Check if years match between citation and paper."""
        try:
            paper_year = int(published_date[:4])  # Assume YYYY-MM-DD format
            return abs(citation_year - paper_year) <= 1  # Allow 1 year difference
        except (ValueError, TypeError):
            return False
    
    def extract_citations_batch(
        self, 
        pdf_paths: List[str],
        paper_ids: List[str]
    ) -> Dict[str, List[Citation]]:
        """
        Extract citations from multiple PDFs.
        
        Args:
            pdf_paths: List of PDF file paths
            paper_ids: Corresponding paper IDs
            
        Returns:
            Dict mapping paper_id to list of citations
        """
        results = {}
        
        for pdf_path, paper_id in zip(pdf_paths, paper_ids):
            try:
                citations = self.extract_from_pdf(pdf_path)
                results[paper_id] = citations
            except Exception as e:
                logger.error(f"Failed to extract citations from {pdf_path}: {e}")
                results[paper_id] = []
        
        return results
    
    def save_citations(self, citations: Dict[str, List[Citation]], output_path: str):
        """Save extracted citations to JSON file."""
        # Convert citations to serializable format
        serializable_citations = {}
        
        for paper_id, citation_list in citations.items():
            serializable_citations[paper_id] = [
                {
                    'raw_text': c.raw_text,
                    'title': c.title,
                    'authors': c.authors,
                    'journal': c.journal,
                    'year': c.year,
                    'doi': c.doi,
                    'arxiv_id': c.arxiv_id,
                    'url': c.url,
                    'confidence': c.confidence
                }
                for c in citation_list
            ]
        
        with open(output_path, 'w') as f:
            json.dump(serializable_citations, f, indent=2)
        
        logger.info(f"Saved citations to {output_path}")
    
    def load_citations(self, input_path: str) -> Dict[str, List[Citation]]:
        """Load citations from JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        citations = {}
        for paper_id, citation_list in data.items():
            citations[paper_id] = [
                Citation(
                    raw_text=c['raw_text'],
                    title=c.get('title'),
                    authors=c.get('authors', []),
                    journal=c.get('journal'),
                    year=c.get('year'),
                    doi=c.get('doi'),
                    arxiv_id=c.get('arxiv_id'),
                    url=c.get('url'),
                    confidence=c.get('confidence', 0.0)
                )
                for c in citation_list
            ]
        
        return citations


def extract_citations_from_directory(
    pdf_directory: str,
    output_file: str,
    paper_metadata: Optional[Dict[str, Dict]] = None
) -> Dict[str, List[Citation]]:
    """
    Convenience function to extract citations from all PDFs in a directory.
    
    Args:
        pdf_directory: Directory containing PDF files
        output_file: Output JSON file for citations
        paper_metadata: Optional metadata for matching
        
    Returns:
        Dict mapping filename to citations
    """
    extractor = CitationExtractor()
    pdf_dir = Path(pdf_directory)
    
    all_citations = {}
    
    for pdf_file in pdf_dir.glob("*.pdf"):
        paper_id = pdf_file.stem
        citations = extractor.extract_from_pdf(str(pdf_file))
        all_citations[paper_id] = citations
    
    # Save results
    extractor.save_citations(all_citations, output_file)
    
    return all_citations


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python citation_extractor.py <pdf_file_or_directory>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if Path(input_path).is_file():
        # Single file
        extractor = CitationExtractor()
        citations = extractor.extract_from_pdf(input_path)
        
        print(f"Extracted {len(citations)} citations:")
        for i, citation in enumerate(citations, 1):
            print(f"\n{i}. {citation.raw_text[:100]}...")
            if citation.title:
                print(f"   Title: {citation.title}")
            if citation.doi:
                print(f"   DOI: {citation.doi}")
            if citation.arxiv_id:
                print(f"   arXiv: {citation.arxiv_id}")
            print(f"   Confidence: {citation.confidence:.2f}")
    
    else:
        # Directory
        output_file = "extracted_citations.json"
        citations = extract_citations_from_directory(input_path, output_file)
        print(f"Extracted citations from {len(citations)} papers, saved to {output_file}")
