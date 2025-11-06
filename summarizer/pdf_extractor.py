"""
PDF Text Extractor

Extracts structured text from PDF research papers using PyMuPDF.
Handles section detection, metadata extraction, and OCR fallback.
"""

import re
import logging
import io
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from dataclasses import dataclass, asdict

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a section of a research paper."""
    title: str
    content: str
    page_start: int
    page_end: int
    level: int  # Header level (1 for main sections, 2 for subsections)
    bbox: Optional[Tuple[float, float, float, float]] = None  # Bounding box


@dataclass
class PaperStructure:
    """Complete structure of an extracted research paper."""
    title: str
    authors: List[str]
    abstract: str
    sections: List[Section]
    references: str
    metadata: Dict
    total_pages: int
    extraction_method: str  # 'text' or 'ocr'


class PDFExtractor:
    """Extracts structured text from PDF research papers."""
    
    # Common section headers in research papers
    SECTION_PATTERNS = [
        # Main sections
        r'^\s*(?:abstract|introduction|background|related\s+work|methodology?|methods?|approach|implementation|'
        r'results?|experiments?|evaluation|discussion|conclusions?|future\s+work|acknowledgments?|references?)\s*$',
        
        # Numbered sections
        r'^\s*(?:\d+\.?\s+)(?:abstract|introduction|background|related\s+work|methodology?|methods?|approach|'
        r'implementation|results?|experiments?|evaluation|discussion|conclusions?|future\s+work|acknowledgments?|references?)',
        
        # Roman numerals
        r'^\s*(?:[ivxlc]+\.?\s+)(?:abstract|introduction|background|related\s+work|methodology?|methods?|approach|'
        r'implementation|results?|experiments?|evaluation|discussion|conclusions?|future\s+work|acknowledgments?|references?)',
        
        # Lettered sections
        r'^\s*(?:[a-z]\.?\s+)(?:abstract|introduction|background|related\s+work|methodology?|methods?|approach|'
        r'implementation|results?|experiments?|evaluation|discussion|conclusions?|future\s+work|acknowledgments?|references?)',
    ]
    
    def __init__(self, use_ocr: bool = True, ocr_fallback_threshold: float = 0.1):
        """
        Initialize PDF extractor.
        
        Args:
            use_ocr: Whether to use OCR as fallback for scanned PDFs
            ocr_fallback_threshold: Minimum text ratio to avoid OCR (0.1 = 10% of expected text)
        """
        self.use_ocr = use_ocr
        self.ocr_fallback_threshold = ocr_fallback_threshold
        self.section_regex = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                             for pattern in self.SECTION_PATTERNS]
    
    def extract_from_file(self, pdf_path: str) -> PaperStructure:
        """
        Extract structured text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PaperStructure with extracted content
        """
        if fitz is None:
            raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting text from: {pdf_path}")
        
        try:
            with fitz.open(str(pdf_path)) as doc:
                # Try text extraction first
                paper_structure = self._extract_text(doc)
                
                # Check if OCR fallback is needed
                if self._needs_ocr_fallback(paper_structure):
                    logger.info("Text extraction insufficient, trying OCR fallback")
                    paper_structure = self._extract_with_ocr(doc)
                
                # Add metadata
                paper_structure.metadata.update({
                    'file_path': str(pdf_path),
                    'file_size': pdf_path.stat().st_size,
                    'extraction_timestamp': None  # Will be set by caller
                })
                
                logger.info(f"Successfully extracted {len(paper_structure.sections)} sections "
                          f"from {paper_structure.total_pages} pages")
                
                return paper_structure
                
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise
    
    def _extract_text(self, doc: Any) -> PaperStructure:
        """Extract text using PyMuPDF's built-in text extraction."""
        pages_text = []
        total_pages = len(doc)
        
        # Extract text from each page
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            pages_text.append({
                'page_num': page_num,
                'text': text,
                'blocks': page.get_text("dict")['blocks']
            })
        
        # Combine all text
        full_text = '\n'.join([page['text'] for page in pages_text])
        
        # Extract paper metadata
        title, authors = self._extract_title_authors(pages_text[0]['text'] if pages_text else "")
        abstract = self._extract_abstract(full_text)
        references = self._extract_references(full_text)
        
        # Detect sections
        sections = self._detect_sections(pages_text)
        
        return PaperStructure(
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            references=references,
            metadata={
                'extraction_method': 'text',
                'pages': total_pages
            },
            total_pages=total_pages,
            extraction_method='text'
        )
    
    def _extract_with_ocr(self, doc: Any) -> PaperStructure:
        """Extract text using OCR for scanned PDFs."""
        if not self.use_ocr:
            logger.warning("OCR disabled, returning empty structure")
            return self._create_empty_structure(len(doc))
        
        try:
            # Try to use tesseract for OCR
            import pytesseract
            from PIL import Image
            
            pages_text = []
            total_pages = len(doc)
            
            for page_num in range(min(total_pages, 5)):  # Limit OCR to first 5 pages for speed
                page = doc[page_num]
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                img_data = pix.tobytes("ppm")
                
                # OCR the image
                try:
                    img = Image.open(io.BytesIO(img_data))
                    text = pytesseract.image_to_string(img)
                    pages_text.append({
                        'page_num': page_num,
                        'text': text,
                        'blocks': []
                    })
                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num}: {e}")
                    pages_text.append({
                        'page_num': page_num,
                        'text': "",
                        'blocks': []
                    })
            
            # Combine OCR text
            full_text = '\n'.join([page['text'] for page in pages_text])
            
            # Extract components
            title, authors = self._extract_title_authors(pages_text[0]['text'] if pages_text else "")
            abstract = self._extract_abstract(full_text)
            references = self._extract_references(full_text)
            sections = self._detect_sections(pages_text)
            
            return PaperStructure(
                title=title,
                authors=authors,
                abstract=abstract,
                sections=sections,
                references=references,
                metadata={
                    'extraction_method': 'ocr',
                    'pages': total_pages,
                    'ocr_pages': len(pages_text)
                },
                total_pages=total_pages,
                extraction_method='ocr'
            )
            
        except ImportError:
            logger.error("OCR dependencies not available. Install: pip install pytesseract pillow")
            return self._create_empty_structure(len(doc))
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return self._create_empty_structure(len(doc))
    
    def _needs_ocr_fallback(self, paper_structure: PaperStructure) -> bool:
        """Determine if OCR fallback is needed based on extracted content quality."""
        if not self.use_ocr:
            return False
        
        # Calculate text density
        total_sections_text = sum(len(section.content) for section in paper_structure.sections)
        total_abstract_text = len(paper_structure.abstract)
        total_text = total_sections_text + total_abstract_text
        
        # Heuristic: if we have less than 100 characters per page, likely scanned
        expected_chars_per_page = 1000  # Rough estimate
        expected_total_chars = paper_structure.total_pages * expected_chars_per_page
        text_ratio = total_text / expected_total_chars if expected_total_chars > 0 else 0
        
        needs_ocr = text_ratio < self.ocr_fallback_threshold
        
        if needs_ocr:
            logger.info(f"Text ratio {text_ratio:.3f} below threshold {self.ocr_fallback_threshold}, "
                       f"will attempt OCR")
        
        return needs_ocr
    
    def _extract_title_authors(self, first_page_text: str) -> Tuple[str, List[str]]:
        """Extract paper title and authors from first page."""
        lines = first_page_text.strip().split('\n')
        
        # Simple heuristic: title is usually one of the first few lines with reasonable length
        title = ""
        authors = []
        
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            line = line.strip()
            if not line:
                continue
                
            # Skip obvious headers
            if any(skip in line.lower() for skip in ['arxiv:', 'preprint', 'submitted', 'conference']):
                continue
            
            # Title heuristic: first substantial line
            if not title and len(line) > 10 and len(line) < 200:
                title = line
                continue
            
            # Authors heuristic: look for names after title
            if title and self._looks_like_authors(line):
                authors = self._parse_authors(line)
                break
        
        return title.strip(), authors
    
    def _looks_like_authors(self, line: str) -> bool:
        """Check if a line looks like author names."""
        # Simple heuristics for author detection
        line = line.strip()
        
        # Check for typical author patterns
        author_indicators = [
            r'\w+\s+\w+',  # First Last
            r'\w+\.\s*\w+',  # F. Last
            r'\w+,\s*\w+',  # Last, First
            r'and\s+\w+',  # and Name
            r'&\s+\w+',    # & Name
        ]
        
        return any(re.search(pattern, line) for pattern in author_indicators)
    
    def _parse_authors(self, author_line: str) -> List[str]:
        """Parse author names from a line."""
        # Split by common separators
        separators = [',', ' and ', ' & ', ';']
        authors = [author_line]
        
        for sep in separators:
            new_authors = []
            for author in authors:
                if sep in author:
                    new_authors.extend([a.strip() for a in author.split(sep)])
                else:
                    new_authors.append(author)
            authors = new_authors
        
        # Clean and filter authors
        cleaned_authors = []
        for author in authors:
            author = author.strip()
            if author and len(author) > 2 and len(author) < 100:
                # Remove common non-name text
                if not any(skip in author.lower() for skip in ['university', 'department', 'email', '@']):
                    cleaned_authors.append(author)
        
        return cleaned_authors[:10]  # Limit to reasonable number
    
    def _extract_abstract(self, full_text: str) -> str:
        """Extract abstract from the full text."""
        # Look for abstract section
        abstract_patterns = [
            r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*(?:introduction|1\.|keywords|index\s+terms))',
            r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*\n)',
            r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*[A-Z])'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up the abstract
                abstract = re.sub(r'\s+', ' ', abstract)  # Normalize whitespace
                if len(abstract) > 50 and len(abstract) < 2000:  # Reasonable length
                    return abstract
        
        return ""
    
    def _extract_references(self, full_text: str) -> str:
        """Extract references section from the full text."""
        # Look for references section (usually at the end)
        references_patterns = [
            r'references\s*[:\-]?\s*(.*?)(?=\n\s*(?:appendix|acknowledgment))',
            r'references\s*[:\-]?\s*(.*?)$',
            r'bibliography\s*[:\-]?\s*(.*?)$'
        ]
        
        for pattern in references_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
            if match:
                references = match.group(1).strip()
                if len(references) > 100:  # Reasonable length for references
                    return references
        
        return ""
    
    def _detect_sections(self, pages_text: List[Dict]) -> List[Section]:
        """Detect and extract sections from the text."""
        sections = []
        full_text = '\n'.join([page['text'] for page in pages_text])
        lines = full_text.split('\n')
        
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if this line is a section header
            is_header, header_level = self._is_section_header(line_stripped)
            
            if is_header:
                # Save previous section
                if current_section:
                    sections.append(Section(
                        title=current_section,
                        content='\n'.join(current_content).strip(),
                        page_start=0,  # TODO: Calculate actual page numbers
                        page_end=0,
                        level=header_level
                    ))
                
                # Start new section
                current_section = line_stripped
                current_content = []
            else:
                # Add to current section content
                if current_section and line_stripped:
                    current_content.append(line)
        
        # Add final section
        if current_section:
            sections.append(Section(
                title=current_section,
                content='\n'.join(current_content).strip(),
                page_start=0,
                page_end=0,
                level=1
            ))
        
        # If no sections detected, create a single section with all content
        if not sections and full_text.strip():
            sections.append(Section(
                title="Full Document",
                content=full_text.strip(),
                page_start=0,
                page_end=len(pages_text) - 1,
                level=1
            ))
        
        return sections
    
    def _is_section_header(self, line: str) -> Tuple[bool, int]:
        """Check if a line is a section header and return its level."""
        if not line or len(line) > 200:  # Too long to be a header
            return False, 0
        
        # Check against section patterns
        for pattern in self.section_regex:
            if pattern.match(line):
                # Determine level based on formatting
                level = 1
                if re.match(r'^\s*\d+\.\d+', line):  # Subsection (e.g., "2.1")
                    level = 2
                elif line.isupper():  # ALL CAPS suggests main section
                    level = 1
                elif line.startswith('  ') or line.startswith('\t'):  # Indented
                    level = 2
                
                return True, level
        
        # Additional heuristics for headers
        # Check for standalone lines that could be headers
        if (line.isupper() or line.istitle()) and len(line.split()) <= 8:
            common_headers = [
                'introduction', 'background', 'methodology', 'methods', 'approach',
                'implementation', 'results', 'experiments', 'evaluation', 'discussion',
                'conclusion', 'conclusions', 'future work', 'related work', 'acknowledgments'
            ]
            if any(header in line.lower() for header in common_headers):
                return True, 1
        
        return False, 0
    
    def _create_empty_structure(self, total_pages: int) -> PaperStructure:
        """Create an empty paper structure for failed extractions."""
        return PaperStructure(
            title="",
            authors=[],
            abstract="",
            sections=[],
            references="",
            metadata={'extraction_method': 'failed'},
            total_pages=total_pages,
            extraction_method='failed'
        )
    
    def save_extracted_structure(self, paper_structure: PaperStructure, output_path: str) -> bool:
        """
        Save extracted paper structure to JSON file.
        
        Args:
            paper_structure: Extracted paper structure
            output_path: Path to save the JSON file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dict for JSON serialization
            structure_dict = asdict(paper_structure)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structure_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved extracted structure to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save structure to {output_path}: {e}")
            return False


def extract_pdf_structure(pdf_path: str, output_path: Optional[str] = None) -> PaperStructure:
    """
    Convenience function to extract structure from a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the extracted structure
        
    Returns:
        PaperStructure with extracted content
    """
    extractor = PDFExtractor()
    structure = extractor.extract_from_file(pdf_path)
    
    if output_path:
        extractor.save_extracted_structure(structure, output_path)
    
    return structure


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <pdf_path> [output_path]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        structure = extract_pdf_structure(pdf_path, output_path)
        
        print(f"Extracted structure from: {pdf_path}")
        print(f"Title: {structure.title}")
        print(f"Authors: {', '.join(structure.authors)}")
        print(f"Abstract length: {len(structure.abstract)} chars")
        print(f"Sections: {len(structure.sections)}")
        print(f"Method: {structure.extraction_method}")
        
        for i, section in enumerate(structure.sections):
            print(f"  {i+1}. {section.title} ({len(section.content)} chars)")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
