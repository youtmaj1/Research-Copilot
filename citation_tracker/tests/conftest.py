"""
Test Suite for Citation Tracker Module

This module contains comprehensive tests for all citation tracker components,
including unit tests, integration tests, and performance tests.

Test Structure:
- test_extractor.py: Tests for citation extraction functionality
- test_resolver.py: Tests for citation resolution and matching  
- test_graph.py: Tests for graph building and analysis
- test_temporal.py: Tests for temporal analysis features
- test_exporter.py: Tests for graph export functionality
- test_database_schema.py: Tests for database schema and operations
- test_integration.py: End-to-end integration tests
- conftest.py: Pytest configuration and fixtures
"""

import pytest
import tempfile
import sqlite3
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Test configuration
TEST_DB_NAME = "test_citations.db"
SAMPLE_PDF_PATH = "test_data/sample_paper.pdf"

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Create basic papers table for testing
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            year INTEGER,
            doi TEXT,
            arxiv_id TEXT,
            venue TEXT,
            abstract TEXT
        )
    """)
    
    # Insert sample papers for testing
    sample_papers = [
        ('paper1', 'Machine Learning for Citation Analysis', 'Smith, J.; Jones, A.', 2023, '10.1000/123', None, 'ICML', 'Abstract for paper 1'),
        ('paper2', 'Deep Learning Networks', 'Brown, B.; Davis, C.', 2022, None, '2201.12345', 'NeurIPS', 'Abstract for paper 2'),
        ('paper3', 'Graph Neural Networks', 'Wilson, D.; Taylor, E.', 2021, '10.1000/456', '2101.67890', 'ICLR', 'Abstract for paper 3'),
        ('paper4', 'Attention Mechanisms in AI', 'Garcia, F.; Lee, G.', 2020, '10.1000/789', None, 'AAAI', 'Abstract for paper 4'),
        ('paper5', 'Natural Language Processing', 'Johnson, H.; Kim, I.', 2019, None, None, 'ACL', 'Abstract for paper 5')
    ]
    
    cursor.executemany("""
        INSERT INTO papers (id, title, authors, year, doi, arxiv_id, venue, abstract)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_papers)
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    os.unlink(db_path)

@pytest.fixture
def sample_citation_text():
    """Sample citation text for testing extraction."""
    return """
References

[1] Smith, J. and Jones, A. Machine Learning for Citation Analysis. In Proceedings of ICML 2023. doi:10.1000/123

[2] Brown, B., Davis, C. Deep Learning Networks. NeurIPS 2022. arXiv:2201.12345

[3] Wilson, D., Taylor, E. "Graph Neural Networks: A Comprehensive Survey". ICLR 2021. doi:10.1000/456, arXiv:2101.67890

[4] Garcia, F.; Lee, G. (2020). Attention Mechanisms in AI. AAAI Conference. DOI: 10.1000/789

[5] Johnson, H. and Kim, I. Natural Language Processing techniques. ACL 2019.
"""

@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return """
Title: Advanced Citation Analysis Methods

Authors: Test Author, Another Author

Abstract: This paper presents novel methods for citation analysis...

Introduction: Citation analysis is crucial for understanding research impact...

Related Work: Previous work in this area includes...

References:
[1] Smith, J. Machine Learning for Citation Analysis. ICML 2023.
[2] Brown, B. Deep Learning Networks. NeurIPS 2022. arXiv:2201.12345
"""

def create_test_data_dir():
    """Create test data directory with sample files."""
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Create sample citation files
    with open(test_data_dir / "sample_citations.txt", "w") as f:
        f.write("""
[1] Smith, J. Machine Learning Analysis. ICML 2023. doi:10.1000/123
[2] Brown, B. Deep Learning. NeurIPS 2022. arXiv:2201.12345
[3] Wilson, D. Graph Networks. ICLR 2021.
""")
    
    # Create sample JSON citation data
    sample_citations = [
        {
            "raw_text": "Smith, J. Machine Learning Analysis. ICML 2023. doi:10.1000/123",
            "doi": "10.1000/123",
            "title": "Machine Learning Analysis",
            "authors": "Smith, J.",
            "year": 2023,
            "venue": "ICML"
        }
    ]
    
    with open(test_data_dir / "sample_citations.json", "w") as f:
        json.dump(sample_citations, f, indent=2)
    
    return test_data_dir

# Performance test fixtures
@pytest.fixture
def large_citation_dataset():
    """Generate a large dataset for performance testing."""
    citations = []
    for i in range(1000):
        citation = {
            "raw_text": f"Author{i}, X. Paper Title {i}. Venue {i%10} {2020 + i%5}. doi:10.1000/{i}",
            "doi": f"10.1000/{i}",
            "title": f"Paper Title {i}",
            "authors": f"Author{i}, X.",
            "year": 2020 + i % 5,
            "venue": f"Venue {i%10}"
        }
        citations.append(citation)
    return citations

@pytest.fixture
def mock_graph_data():
    """Mock graph data for testing."""
    return {
        "nodes": {
            "paper1": {"title": "Paper 1", "authors": "Author A", "year": 2023},
            "paper2": {"title": "Paper 2", "authors": "Author B", "year": 2022},
            "paper3": {"title": "Paper 3", "authors": "Author C", "year": 2021}
        },
        "edges": [
            {"source": "paper1", "target": "paper2", "confidence": 0.9},
            {"source": "paper1", "target": "paper3", "confidence": 0.8},
            {"source": "paper2", "target": "paper3", "confidence": 0.7}
        ]
    }

# Test utilities
def assert_citation_quality(citations: List[Dict[str, Any]], min_confidence: float = 0.5):
    """Assert that citations meet quality thresholds."""
    assert len(citations) > 0, "No citations extracted"
    
    for citation in citations:
        assert citation.get('confidence', 0) >= min_confidence, f"Low confidence citation: {citation}"
        assert len(citation.get('raw_text', '')) > 10, f"Citation text too short: {citation}"

def assert_graph_structure(graph_data: Dict[str, Any]):
    """Assert that graph has valid structure."""
    assert 'nodes' in graph_data, "Graph missing nodes"
    assert 'edges' in graph_data, "Graph missing edges"
    assert len(graph_data['nodes']) > 0, "Graph has no nodes"
    
    # Validate edge references
    node_ids = set(graph_data['nodes'].keys())
    for edge in graph_data['edges']:
        assert edge['source'] in node_ids, f"Edge references unknown source: {edge['source']}"
        assert edge['target'] in node_ids, f"Edge references unknown target: {edge['target']}"

def measure_performance(func, *args, **kwargs):
    """Measure function execution time."""
    start_time = datetime.now()
    result = func(*args, **kwargs)
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    return result, execution_time

# Clean up function
def cleanup_test_files():
    """Clean up test files after test run."""
    test_files = [
        "test_citations.db",
        "test_graph.json",
        "test_export.graphml",
        "test_data"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)

# Add to pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    create_test_data_dir()

def pytest_unconfigure(config):
    """Clean up after pytest."""
    cleanup_test_files()
