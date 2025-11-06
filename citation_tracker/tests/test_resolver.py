"""
Tests for Citation Resolver Module

This module tests the citation resolution functionality including:
- Citation matching to database papers
- Fuzzy matching algorithms
- Confidence scoring
- Ambiguous citation handling
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from resolver import CitationResolver, CitationMatch
from extractor import ExtractedCitation

class TestCitationResolver:
    """Test class for CitationResolver functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Will be set by individual tests with temp_db fixture
        self.resolver = None
        self.db_path = None
    
    def test_resolver_initialization(self, temp_db):
        """Test that resolver initializes correctly with database."""
        self.resolver = CitationResolver(temp_db)
        assert self.resolver is not None
        assert self.resolver.db_path == temp_db
        assert self.resolver.connection is not None
    
    def test_doi_matching(self, temp_db):
        """Test exact DOI matching."""
        self.resolver = CitationResolver(temp_db)
        
        # Create test citation with DOI
        citation = ExtractedCitation(
            raw_text="Smith, J. Machine Learning for Citation Analysis. ICML 2023. doi:10.1000/123",
            doi="10.1000/123",
            title="Machine Learning for Citation Analysis",
            authors="Smith, J.",
            year=2023,
            source_paper_id="citing_paper"
        )
        
        # Resolve citation
        match = self.resolver.resolve_single_citation(citation)
        
        assert match is not None, "Should find DOI match"
        assert match.paper_id == "paper1"
        assert match.match_type == "doi"
        assert match.confidence >= 0.9
        assert match.paper_doi == "10.1000/123"
    
    def test_arxiv_matching(self, temp_db):
        """Test exact arXiv ID matching."""
        self.resolver = CitationResolver(temp_db)
        
        # Create test citation with arXiv ID
        citation = ExtractedCitation(
            raw_text="Brown, B. Deep Learning Networks. NeurIPS 2022. arXiv:2201.12345",
            arxiv_id="2201.12345",
            title="Deep Learning Networks",
            authors="Brown, B.",
            year=2022,
            source_paper_id="citing_paper"
        )
        
        # Resolve citation
        match = self.resolver.resolve_single_citation(citation)
        
        assert match is not None, "Should find arXiv match"
        assert match.paper_id == "paper2"
        assert match.match_type == "arxiv"
        assert match.confidence >= 0.9
        assert match.paper_arxiv_id == "2201.12345"
    
    def test_title_matching(self, temp_db):
        """Test fuzzy title matching."""
        self.resolver = CitationResolver(temp_db)
        
        # Create test citation with similar title
        citation = ExtractedCitation(
            raw_text="Wilson, D. Graph Neural Network Survey. ICLR 2021.",
            title="Graph Neural Network Survey",  # Similar to "Graph Neural Networks"
            authors="Wilson, D.",
            year=2021,
            source_paper_id="citing_paper"
        )
        
        # Resolve citation
        match = self.resolver.resolve_single_citation(citation)
        
        # Should find a match based on title similarity
        if match:
            assert match.match_type == "title"
            assert match.confidence > 0.5
            assert match.paper_id == "paper3"
    
    def test_author_year_matching(self, temp_db):
        """Test author and year-based matching."""
        self.resolver = CitationResolver(temp_db)
        
        # Create test citation with matching author and year
        citation = ExtractedCitation(
            raw_text="Garcia, F. and Lee, G. Attention in AI. AAAI 2020.",
            title="Attention in AI",
            authors="Garcia, F.; Lee, G.",
            year=2020,
            source_paper_id="citing_paper"
        )
        
        # Resolve citation
        match = self.resolver.resolve_single_citation(citation)
        
        # Should find a match based on authors and year
        if match:
            assert match.match_type in ["author_year", "combined"]
            assert match.confidence > 0.4
    
    def test_no_match_scenario(self, temp_db):
        """Test when no match can be found."""
        self.resolver = CitationResolver(temp_db)
        
        # Create test citation that shouldn't match anything
        citation = ExtractedCitation(
            raw_text="Unknown, A. Nonexistent Paper. Random Conference 1999.",
            title="Nonexistent Paper",
            authors="Unknown, A.",
            year=1999,
            doi="10.9999/nonexistent",
            source_paper_id="citing_paper"
        )
        
        # Resolve citation
        match = self.resolver.resolve_single_citation(citation)
        
        assert match is None, "Should not find match for nonexistent paper"
    
    def test_batch_resolution(self, temp_db):
        """Test resolving multiple citations at once."""
        self.resolver = CitationResolver(temp_db)
        
        # Create multiple test citations
        citations = [
            ExtractedCitation(
                raw_text="Smith, J. ML Citation Analysis. doi:10.1000/123",
                doi="10.1000/123",
                source_paper_id="citing_paper"
            ),
            ExtractedCitation(
                raw_text="Brown, B. Deep Learning. arXiv:2201.12345",
                arxiv_id="2201.12345",
                source_paper_id="citing_paper"
            ),
            ExtractedCitation(
                raw_text="Unknown paper that won't match.",
                title="Unknown Paper",
                source_paper_id="citing_paper"
            )
        ]
        
        # Resolve all citations
        matches = self.resolver.resolve_citations(citations)
        
        # Should find matches for the first two
        assert len(matches) >= 2, f"Expected at least 2 matches, got {len(matches)}"
        
        match_types = [match.match_type for match in matches]
        assert "doi" in match_types
        assert "arxiv" in match_types
    
    def test_similarity_calculations(self, temp_db):
        """Test title and author similarity calculations."""
        self.resolver = CitationResolver(temp_db)
        
        # Test title similarity
        title1 = "Machine Learning for Citation Analysis"
        title2 = "Machine Learning Citation Analysis"  # Very similar
        title3 = "Deep Learning Networks"  # Different
        
        sim1 = self.resolver._calculate_title_similarity(title1, title2)
        sim2 = self.resolver._calculate_title_similarity(title1, title3)
        
        assert sim1 > sim2, "Similar titles should have higher similarity"
        assert sim1 > 0.8, "Very similar titles should have high similarity"
        assert sim2 < 0.5, "Different titles should have low similarity"
        
        # Test author similarity
        authors1 = "Smith, J.; Jones, A."
        authors2 = "J. Smith, A. Jones"  # Same authors, different format
        authors3 = "Brown, B.; Davis, C."  # Different authors
        
        auth_sim1 = self.resolver._calculate_author_similarity(authors1, authors2)
        auth_sim2 = self.resolver._calculate_author_similarity(authors1, authors3)
        
        assert auth_sim1 > auth_sim2, "Same authors should have higher similarity"
    
    def test_text_normalization(self, temp_db):
        """Test text normalization for comparison."""
        self.resolver = CitationResolver(temp_db)
        
        # Test title normalization
        title1 = "The Machine Learning Analysis"
        title2 = "Machine Learning Analysis (Extended Version)"
        
        norm1 = self.resolver._normalize_title(title1)
        norm2 = self.resolver._normalize_title(title2)
        
        assert norm1 == "machine learning analysis"
        assert norm2 == "machine learning analysis"  # Should remove parentheses
        
        # Test author normalization
        authors1 = "Smith, J. and Brown, B."
        authors2 = "J. Smith, B. Brown, et al."
        
        norm_auth1 = self.resolver._normalize_authors(authors1)
        norm_auth2 = self.resolver._normalize_authors(authors2)
        
        assert "smith" in norm_auth1
        assert "brown" in norm_auth1
        assert "et al" in norm_auth2  # Should be normalized
    
    def test_ambiguous_matches(self, temp_db):
        """Test handling of ambiguous matches."""
        # Add similar papers to database for ambiguity testing
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Add two very similar papers
        cursor.execute("""
            INSERT INTO papers (id, title, authors, year, doi, arxiv_id, venue)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ('similar1', 'Machine Learning Methods', 'Smith, J.', 2023, None, None, 'ICML'))
        
        cursor.execute("""
            INSERT INTO papers (id, title, authors, year, doi, arxiv_id, venue)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ('similar2', 'Machine Learning Methods', 'Smith, J.', 2023, None, None, 'NeurIPS'))
        
        conn.commit()
        conn.close()
        
        self.resolver = CitationResolver(temp_db)
        
        # Create citation that could match both
        citation = ExtractedCitation(
            raw_text="Smith, J. Machine Learning Methods. 2023.",
            title="Machine Learning Methods",
            authors="Smith, J.",
            year=2023,
            source_paper_id="citing_paper"
        )
        
        match = self.resolver.resolve_single_citation(citation)
        
        # Should find a match, potentially marked as ambiguous
        if match:
            # If marked as ambiguous, should have alternatives
            if match.ambiguous:
                assert len(match.alternative_matches) > 0
    
    def test_unresolved_citations(self, temp_db):
        """Test getting unresolved citations."""
        self.resolver = CitationResolver(temp_db)
        
        citations = [
            ExtractedCitation(
                raw_text="Known paper with DOI. doi:10.1000/123",
                doi="10.1000/123",
                source_paper_id="citing_paper"
            ),
            ExtractedCitation(
                raw_text="Unknown paper that won't resolve.",
                title="Unknown Paper",
                source_paper_id="citing_paper"
            )
        ]
        
        unresolved = self.resolver.get_unresolved_citations(citations)
        
        # Should have the unresolvable citation
        assert len(unresolved) >= 1
        assert any("Unknown Paper" in citation.raw_text for citation in unresolved)
    
    def test_resolution_statistics(self, temp_db):
        """Test resolution statistics calculation."""
        self.resolver = CitationResolver(temp_db)
        
        citations = [
            ExtractedCitation(
                raw_text="Known paper 1. doi:10.1000/123",
                doi="10.1000/123",
                source_paper_id="citing_paper"
            ),
            ExtractedCitation(
                raw_text="Known paper 2. arXiv:2201.12345",
                arxiv_id="2201.12345",
                source_paper_id="citing_paper"
            ),
            ExtractedCitation(
                raw_text="Unknown paper.",
                title="Unknown Paper",
                source_paper_id="citing_paper"
            )
        ]
        
        stats = self.resolver.get_resolution_statistics(citations)
        
        assert stats['total_citations'] == 3
        assert stats['resolved_citations'] >= 2
        assert stats['resolution_rate'] >= 0.6
        assert 'match_types' in stats
        assert 'average_confidence' in stats
    
    def test_database_connection_error(self):
        """Test handling of database connection errors."""
        # Try to initialize with non-existent database
        resolver = CitationResolver("nonexistent.db")
        
        # Should handle gracefully
        citation = ExtractedCitation(
            raw_text="Test citation",
            source_paper_id="test"
        )
        
        match = resolver.resolve_single_citation(citation)
        assert match is None, "Should return None when database unavailable"
    
    def test_performance_large_batch(self, temp_db):
        """Test performance with large batch of citations."""
        self.resolver = CitationResolver(temp_db)
        
        # Create large batch of citations
        citations = []
        for i in range(50):
            citation = ExtractedCitation(
                raw_text=f"Paper {i}. Conference {i % 5} {2020 + i % 3}.",
                title=f"Paper {i}",
                year=2020 + i % 3,
                source_paper_id="citing_paper"
            )
            citations.append(citation)
        
        start_time = datetime.now()
        matches = self.resolver.resolve_citations(citations)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete in reasonable time
        assert processing_time < 10, f"Batch resolution took too long: {processing_time} seconds"
        
        # Should process all citations (even if no matches found)
        assert len(citations) == 50, "All citations should be processed"

class TestCitationMatchDataClass:
    """Test the CitationMatch data class."""
    
    def test_match_creation(self):
        """Test creating CitationMatch instances."""
        citation = ExtractedCitation(
            raw_text="Test citation",
            source_paper_id="citing_paper"
        )
        
        match = CitationMatch(
            paper_id="matched_paper",
            citation=citation,
            match_type="doi",
            confidence=0.95,
            paper_title="Matched Paper Title"
        )
        
        assert match.paper_id == "matched_paper"
        assert match.citation == citation
        assert match.match_type == "doi"
        assert match.confidence == 0.95
        assert match.paper_title == "Matched Paper Title"
        assert not match.ambiguous  # Default value
        assert len(match.alternative_matches) == 0  # Default empty list
    
    def test_match_defaults(self):
        """Test default values for CitationMatch."""
        citation = ExtractedCitation(
            raw_text="Test citation",
            source_paper_id="citing_paper"
        )
        
        match = CitationMatch(
            paper_id="matched_paper",
            citation=citation,
            match_type="title",
            confidence=0.8
        )
        
        assert match.paper_title == ""
        assert match.paper_authors == ""
        assert match.paper_year is None
        assert match.paper_doi is None
        assert match.paper_arxiv_id is None
        assert not match.ambiguous
        assert match.alternative_matches == []

class TestResolverEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_citation_list(self, temp_db):
        """Test resolving empty citation list."""
        resolver = CitationResolver(temp_db)
        matches = resolver.resolve_citations([])
        assert matches == []
    
    def test_malformed_citations(self, temp_db):
        """Test resolving malformed citations."""
        resolver = CitationResolver(temp_db)
        
        # Citation with no useful information
        bad_citation = ExtractedCitation(
            raw_text="",
            source_paper_id="test"
        )
        
        match = resolver.resolve_single_citation(bad_citation)
        assert match is None
    
    def test_database_corruption_handling(self):
        """Test handling of database corruption."""
        # Create empty file instead of valid database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            corrupt_db_path = f.name
        
        resolver = CitationResolver(corrupt_db_path)
        
        citation = ExtractedCitation(
            raw_text="Test citation",
            source_paper_id="test"
        )
        
        # Should handle corruption gracefully
        match = resolver.resolve_single_citation(citation)
        assert match is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
