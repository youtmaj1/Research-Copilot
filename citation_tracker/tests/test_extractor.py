"""
Tests for Citation Extractor Module

This module tests the citation extraction functionality including:
- Text extraction from PDFs
- Citation parsing and normalization
- DOI and arXiv ID extraction
- Title and author extraction
- Confidence scoring
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from extractor import CitationExtractor, ExtractedCitation, normalize_citation_text, extract_dois_from_text, extract_arxiv_ids_from_text

class TestCitationExtractor:
    """Test class for CitationExtractor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = CitationExtractor()
    
    def test_extractor_initialization(self):
        """Test that extractor initializes correctly."""
        assert self.extractor is not None
        assert hasattr(self.extractor, 'doi_pattern')
        assert hasattr(self.extractor, 'arxiv_pattern')
        assert hasattr(self.extractor, 'year_pattern')
        
        # Test statistics
        stats = self.extractor.get_statistics()
        assert 'patterns_loaded' in stats
        assert stats['patterns_loaded']['doi_pattern'] is True
        assert stats['version'] == "1.0.0"
    
    def test_doi_extraction(self):
        """Test DOI extraction from citation text."""
        test_cases = [
            ("Smith, J. Paper Title. Journal 2023. doi:10.1000/123", "10.1000/123"),
            ("Brown, B. Another Paper. DOI: 10.1234/567890", "10.1234/567890"),
            ("Wilson, D. Third Paper. https://doi.org/10.5678/abcdef", "10.5678/abcdef"),
            ("Garcia, F. Fourth Paper. http://dx.doi.org/10.9999/test", "10.9999/test"),
            ("No DOI here", None)
        ]
        
        for text, expected_doi in test_cases:
            match = self.extractor.doi_pattern.search(text)
            if expected_doi:
                assert match is not None, f"Failed to extract DOI from: {text}"
                assert match.group(1) == expected_doi
            else:
                assert match is None, f"Incorrectly extracted DOI from: {text}"
    
    def test_arxiv_extraction(self):
        """Test arXiv ID extraction from citation text."""
        test_cases = [
            ("Smith, J. Paper Title. arXiv:2301.12345", "2301.12345"),
            ("Brown, B. Another Paper. arxiv:2201.67890v2", "2201.67890v2"),
            ("Wilson, D. Old Format. arXiv:cs.AI/0701123", "cs.AI/0701123"),
            ("Garcia, F. Old Format v2. arxiv:math.GT/9901001v1", "math.GT/9901001v1"),
            ("No arXiv here", None)
        ]
        
        for text, expected_arxiv in test_cases:
            # Test new format
            match = self.extractor.arxiv_pattern.search(text)
            old_match = self.extractor.arxiv_old_pattern.search(text)
            
            if expected_arxiv and '/' not in expected_arxiv:
                assert match is not None, f"Failed to extract arXiv ID from: {text}"
                assert match.group(1) == expected_arxiv
            elif expected_arxiv and '/' in expected_arxiv:
                assert old_match is not None, f"Failed to extract old arXiv ID from: {text}"
                assert old_match.group(1) == expected_arxiv
            else:
                assert match is None and old_match is None, f"Incorrectly extracted arXiv ID from: {text}"
    
    def test_year_extraction(self):
        """Test year extraction from citation text."""
        test_cases = [
            ("Smith, J. Paper Title. Journal 2023.", 2023),
            ("Brown, B. (2020) Another Paper.", 2020),
            ("Wilson, D. Third Paper, 1995", 1995),
            ("Garcia, F. Future Paper 2030", 2030),
            ("Too old 1800", None),  # Should not match
            ("Too future 2050", None)  # Should not match
        ]
        
        for text, expected_year in test_cases:
            match = self.extractor.year_pattern.search(text)
            if expected_year:
                assert match is not None, f"Failed to extract year from: {text}"
                assert int(match.group(1)) == expected_year
            else:
                assert match is None, f"Incorrectly extracted year from: {text}"
    
    def test_citation_parsing(self, sample_citation_text):
        """Test parsing of citation text."""
        citations = self.extractor.extract_citations_from_text(sample_citation_text, "test_paper")
        
        assert len(citations) > 0, "No citations extracted from sample text"
        
        # Check first citation
        first_citation = citations[0]
        assert isinstance(first_citation, ExtractedCitation)
        assert first_citation.source_paper_id == "test_paper"
        assert len(first_citation.raw_text) > 0
        assert first_citation.doi is not None or first_citation.arxiv_id is not None or first_citation.title is not None
        assert first_citation.confidence > 0
    
    def test_reference_section_extraction(self):
        """Test extraction of reference section from full paper text."""
        full_text = """
Title: Test Paper

Abstract: This is a test paper...

Introduction: 
Citation analysis is important...

Methods:
We used various techniques...

Results:
Our findings show...

References

[1] Smith, J. First Reference. Journal 2023. doi:10.1000/123
[2] Brown, B. Second Reference. Conference 2022. arXiv:2201.12345

Appendix A
Additional material...
"""
        
        ref_section = self.extractor._extract_reference_section(full_text)
        assert len(ref_section) > 0, "Failed to extract reference section"
        assert "[1] Smith, J." in ref_section
        assert "[2] Brown, B." in ref_section
        assert "Appendix A" not in ref_section  # Should stop before appendix
    
    def test_citation_splitting(self):
        """Test splitting reference section into individual citations."""
        ref_text = """
[1] Smith, J. First Citation. Journal 2023.
[2] Brown, B. Second Citation. Conference 2022.
[3] Wilson, D. Third Citation. Workshop 2021.
"""
        
        citations = self.extractor._split_into_citations(ref_text)
        assert len(citations) >= 3, f"Expected at least 3 citations, got {len(citations)}"
        
        # Check that each citation contains expected content
        citation_texts = " ".join(citations)
        assert "Smith, J." in citation_texts
        assert "Brown, B." in citation_texts
        assert "Wilson, D." in citation_texts
    
    def test_title_extraction(self):
        """Test title extraction from citation text."""
        test_cases = [
            ('Smith, J. "Paper Title in Quotes". Journal 2023.', "Paper Title in Quotes"),
            ("Brown, B. Title Without Quotes. Conference 2022.", "Title Without Quotes"),
            ("[1] Wilson, D. Another Title. Workshop 2021.", "Another Title")
        ]
        
        for citation_text, expected_title in test_cases:
            title = self.extractor._extract_title(citation_text)
            if expected_title:
                assert title is not None, f"Failed to extract title from: {citation_text}"
                assert expected_title.lower() in title.lower(), f"Expected '{expected_title}' in '{title}'"
    
    def test_author_extraction(self):
        """Test author extraction from citation text."""
        test_cases = [
            ("Smith, J. and Brown, B. Paper Title. Journal 2023.", "Smith, J. and Brown, B."),
            ("A. Smith, B. Brown, C. Wilson. Another Paper.", "A. Smith, B. Brown, C. Wilson"),
            ("[1] J. Smith. Single Author Paper.", "J. Smith")
        ]
        
        for citation_text, expected_authors in test_cases:
            authors = self.extractor._extract_authors(citation_text)
            if expected_authors:
                assert authors is not None, f"Failed to extract authors from: {citation_text}"
                # Check that at least part of expected authors is found
                assert any(name in authors for name in expected_authors.split()), \
                    f"Expected part of '{expected_authors}' in '{authors}'"
    
    def test_text_normalization(self):
        """Test citation text normalization."""
        test_cases = [
            ("[1] Smith, J. Paper Title. Journal 2023.", "smith j paper title journal 2023"),
            ("DOI: 10.1000/123", "doi:10.1000/123"),
            ("Extra    spaces   and   punctuation!!!", "extra spaces and punctuation"),
            ("https://example.com/link", "")  # URLs should be removed
        ]
        
        for original, expected in test_cases:
            normalized = self.extractor._normalize_text(original)
            assert expected in normalized.lower(), f"Expected '{expected}' in normalized '{normalized}'"
    
    def test_citation_validation(self):
        """Test citation validation logic."""
        # Valid citation
        valid_citation = ExtractedCitation(
            raw_text="Smith, J. Valid Citation. Journal 2023. doi:10.1000/123",
            doi="10.1000/123",
            title="Valid Citation",
            confidence=0.8,
            source_paper_id="test"
        )
        assert self.extractor._is_valid_citation(valid_citation)
        
        # Invalid citation - too short
        invalid_short = ExtractedCitation(
            raw_text="Too short",
            confidence=0.5,
            source_paper_id="test"
        )
        assert not self.extractor._is_valid_citation(invalid_short)
        
        # Invalid citation - low confidence
        invalid_confidence = ExtractedCitation(
            raw_text="Smith, J. Low Confidence Citation. Journal 2023.",
            confidence=0.05,
            source_paper_id="test"
        )
        assert not self.extractor._is_valid_citation(invalid_confidence)
    
    def test_pdf_extraction(self):
        """Test PDF text extraction (mock test since we don't have actual PDFs)."""
        # Create a mock PDF path
        mock_pdf_path = "mock_paper.pdf"
        
        # Test that the method handles missing files gracefully
        citations = self.extractor.extract_citations_from_pdf(mock_pdf_path, "test_paper")
        assert citations == [], "Should return empty list for missing PDF"
    
    def test_performance_large_text(self):
        """Test performance with large citation text."""
        # Generate large citation text
        large_text = "References\n\n"
        for i in range(100):
            large_text += f"[{i+1}] Author{i}, X. Paper Title {i}. Journal {i} {2020 + i%5}. doi:10.1000/{i}\n"
        
        start_time = datetime.now()
        citations = self.extractor.extract_citations_from_text(large_text, "performance_test")
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        assert len(citations) > 50, "Should extract most citations from large text"
        assert processing_time < 10, f"Processing took too long: {processing_time} seconds"
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty text
        citations = self.extractor.extract_citations_from_text("", "test")
        assert citations == []
        
        # Text with no references
        no_refs = "This is just regular text with no citations or references."
        citations = self.extractor.extract_citations_from_text(no_refs, "test")
        assert len(citations) == 0
        
        # Malformed citations
        malformed = """
        References
        [1] This is not really a citation
        [2] 
        [3] Author. But this might be okay. Journal 2023.
        """
        citations = self.extractor.extract_citations_from_text(malformed, "test")
        # Should extract at least the reasonable one
        assert len(citations) >= 1

class TestUtilityFunctions:
    """Test utility functions in the extractor module."""
    
    def test_normalize_citation_text_function(self):
        """Test standalone text normalization function."""
        test_text = "[1] Smith, J. Paper Title!!! DOI: 10.1000/123"
        normalized = normalize_citation_text(test_text)
        
        assert "smith" in normalized
        assert "paper title" in normalized
        assert "doi:10.1000/123" in normalized
        assert "[1]" not in normalized  # Citation numbering should be removed
    
    def test_extract_dois_function(self):
        """Test standalone DOI extraction function."""
        text = "Multiple papers: doi:10.1000/123 and DOI: 10.5678/456 and https://doi.org/10.9999/789"
        dois = extract_dois_from_text(text)
        
        assert len(dois) == 3
        assert "10.1000/123" in dois
        assert "10.5678/456" in dois
        assert "10.9999/789" in dois
    
    def test_extract_arxiv_ids_function(self):
        """Test standalone arXiv ID extraction function."""
        text = "Multiple papers: arXiv:2301.12345 and arxiv:2201.67890v2 and arXiv:cs.AI/0701123"
        arxiv_ids = extract_arxiv_ids_from_text(text)
        
        assert len(arxiv_ids) == 3
        assert "2301.12345" in arxiv_ids
        assert "2201.67890v2" in arxiv_ids
        assert "cs.AI/0701123" in arxiv_ids

class TestExtractedCitationDataClass:
    """Test the ExtractedCitation data class."""
    
    def test_citation_creation(self):
        """Test creating ExtractedCitation instances."""
        citation = ExtractedCitation(
            raw_text="Smith, J. Test Citation. Journal 2023.",
            doi="10.1000/123",
            title="Test Citation",
            source_paper_id="test_paper"
        )
        
        assert citation.raw_text == "Smith, J. Test Citation. Journal 2023."
        assert citation.doi == "10.1000/123"
        assert citation.title == "Test Citation"
        assert citation.source_paper_id == "test_paper"
        assert citation.normalized_text == ""  # Default value
        assert citation.confidence == 0.0  # Default value
    
    def test_citation_defaults(self):
        """Test default values for ExtractedCitation."""
        citation = ExtractedCitation(
            raw_text="Minimal citation",
            source_paper_id="test"
        )
        
        assert citation.doi is None
        assert citation.arxiv_id is None
        assert citation.title is None
        assert citation.authors is None
        assert citation.year is None
        assert citation.venue is None
        assert citation.normalized_text == ""
        assert citation.confidence == 0.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
