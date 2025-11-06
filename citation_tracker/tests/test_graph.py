"""
Tests for Citation Graph Module

This module tests the citation graph functionality including:
- Graph construction and management
- NetworkX integration
- Node and edge operations
- Graph metrics and analysis
- Centrality calculations
- Community detection
"""

import pytest
import tempfile
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from graph import CitationGraph, PaperNode, CitationEdge
from resolver import CitationMatch
from extractor import ExtractedCitation

class TestCitationGraph:
    """Test the CitationGraph class."""
    
    def setup_method(self):
        """Set up test environment with temporary database."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.setup_test_database()
        self.graph = CitationGraph(self.temp_db)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def setup_test_database(self):
        """Create test database with sample papers."""
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                doi TEXT,
                arxiv_id TEXT,
                venue TEXT
            )
        """)
        
        test_papers = [
            ('paper1', 'Machine Learning Fundamentals', 'Smith, J.', 2020, '10.1000/1', None, 'ICML'),
            ('paper2', 'Deep Learning Applications', 'Jones, A.', 2021, '10.1000/2', '2101.12345', 'NeurIPS'),
            ('paper3', 'Neural Network Architectures', 'Brown, B.', 2022, None, '2201.67890', 'ICLR'),
            ('paper4', 'AI Safety Research', 'Davis, C.', 2023, '10.1000/4', None, 'AAAI'),
        ]
        
        cursor.executemany("""
            INSERT INTO papers (id, title, authors, year, doi, arxiv_id, venue)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, test_papers)
        
        conn.commit()
        conn.close()
    
    def test_graph_initialization(self):
        """Test graph initialization."""
        assert self.graph.db_path == self.temp_db
        assert len(self.graph.paper_nodes) == 0
        assert len(self.graph.citation_edges) == 0
        assert self.graph.nx_graph is not None
    
    def test_add_paper(self):
        """Test adding papers to the graph."""
        # Add a paper
        node = self.graph.add_paper('test1', 'Test Paper', 'Test Author', 2023)
        
        assert isinstance(node, PaperNode)
        assert node.paper_id == 'test1'
        assert node.title == 'Test Paper'
        assert node.authors == 'Test Author'
        assert node.year == 2023
        
        # Verify paper is in graph
        assert 'test1' in self.graph.paper_nodes
        assert self.graph.nx_graph.has_node('test1')
        
        # Test duplicate addition
        node2 = self.graph.add_paper('test1', 'Different Title', 'Different Author', 2024)
        assert node2 is None  # Should not add duplicate
    
    def test_add_citation(self):
        """Test adding citations to the graph."""
        # Add papers first
        self.graph.add_paper('citing', 'Citing Paper', 'Author A', 2023)
        self.graph.add_paper('cited', 'Cited Paper', 'Author B', 2022)
        
        # Add citation
        edge = self.graph.add_citation('citing', 'cited', confidence=0.8)
        
        assert isinstance(edge, CitationEdge)
        assert edge.citing_paper_id == 'citing'
        assert edge.cited_paper_id == 'cited'
        assert edge.confidence == 0.8
        
        # Verify citation is in graph
        citation_key = ('citing', 'cited')
        assert citation_key in self.graph.citation_edges
        assert self.graph.nx_graph.has_edge('citing', 'cited')
        
        # Test citation count update
        cited_node = self.graph.paper_nodes['cited']
        assert cited_node.citation_count == 1
        
        # Test duplicate citation
        edge2 = self.graph.add_citation('citing', 'cited', confidence=0.9)
        assert edge2 is None  # Should not add duplicate
    
    def test_has_citation(self):
        """Test citation existence checking."""
        # Add papers and citation
        self.graph.add_paper('p1', 'Paper 1', 'Author 1', 2021)
        self.graph.add_paper('p2', 'Paper 2', 'Author 2', 2022)
        self.graph.add_citation('p1', 'p2')
        
        # Test citation existence
        assert self.graph.has_citation('p1', 'p2') == True
        assert self.graph.has_citation('p2', 'p1') == False
        assert self.graph.has_citation('p1', 'nonexistent') == False
    
    def test_get_citations_for_paper(self):
        """Test retrieving citations for a paper."""
        # Set up test graph
        self.graph.add_paper('central', 'Central Paper', 'Author C', 2020)
        self.graph.add_paper('ref1', 'Reference 1', 'Author 1', 2019)
        self.graph.add_paper('ref2', 'Reference 2', 'Author 2', 2018)
        self.graph.add_paper('citer1', 'Citer 1', 'Author X', 2021)
        
        # Add citations
        self.graph.add_citation('central', 'ref1')  # central cites ref1
        self.graph.add_citation('central', 'ref2')  # central cites ref2
        self.graph.add_citation('citer1', 'central')  # citer1 cites central
        
        # Test outgoing citations (references)
        outgoing = self.graph.get_citations_for_paper('central', direction='out')
        assert len(outgoing) == 2
        assert 'ref1' in outgoing
        assert 'ref2' in outgoing
        
        # Test incoming citations
        incoming = self.graph.get_citations_for_paper('central', direction='in')
        assert len(incoming) == 1
        assert 'citer1' in incoming
        
        # Test both directions
        both = self.graph.get_citations_for_paper('central', direction='both')
        assert len(both) == 3
        assert all(paper in both for paper in ['ref1', 'ref2', 'citer1'])
    
    def test_get_most_cited_papers(self):
        """Test finding most cited papers."""
        # Create papers with different citation counts
        papers = ['p1', 'p2', 'p3', 'p4']
        for i, paper_id in enumerate(papers):
            self.graph.add_paper(paper_id, f'Paper {i+1}', f'Author {i+1}', 2020 + i)
        
        # Add citations to create different citation counts
        # p1: 3 citations, p2: 2 citations, p3: 1 citation, p4: 0 citations
        for i in range(3):
            citer_id = f'citer_{i}'
            self.graph.add_paper(citer_id, f'Citer {i}', f'Citer Author {i}', 2023)
            self.graph.add_citation(citer_id, 'p1')
            if i < 2:
                self.graph.add_citation(citer_id, 'p2')
            if i < 1:
                self.graph.add_citation(citer_id, 'p3')
        
        # Test getting most cited papers
        top_2 = self.graph.get_most_cited_papers(2)
        assert len(top_2) == 2
        assert top_2[0][0] == 'p1'  # Most cited
        assert top_2[0][1] == 3    # Citation count
        assert top_2[1][0] == 'p2'  # Second most cited
        assert top_2[1][1] == 2    # Citation count
        
        # Test getting all papers
        all_cited = self.graph.get_most_cited_papers(10)
        assert len(all_cited) >= 3  # At least p1, p2, p3 have citations
    
    def test_calculate_metrics(self):
        """Test graph metrics calculation."""
        # Create a simple test graph
        for i in range(5):
            self.graph.add_paper(f'p{i}', f'Paper {i}', f'Author {i}', 2020)
        
        # Add some citations
        self.graph.add_citation('p0', 'p1')
        self.graph.add_citation('p0', 'p2')
        self.graph.add_citation('p1', 'p2')
        self.graph.add_citation('p2', 'p3')
        
        metrics = self.graph.calculate_metrics()
        
        # Basic metrics
        assert metrics['nodes'] == 5
        assert metrics['edges'] == 4
        assert metrics['density'] > 0
        
        # Network metrics
        assert 'average_clustering' in metrics
        assert 'connected_components' in metrics
        assert metrics['connected_components'] >= 1
        
        # Citation metrics
        assert 'total_citations' in metrics
        assert metrics['total_citations'] == 4
        assert 'average_citations_per_paper' in metrics
    
    def test_get_paper_centrality(self):
        """Test centrality calculations."""
        # Create a star network for predictable centrality
        center_id = 'center'
        self.graph.add_paper(center_id, 'Center Paper', 'Center Author', 2020)
        
        # Add peripheral papers that all cite the center
        for i in range(4):
            paper_id = f'peripheral_{i}'
            self.graph.add_paper(paper_id, f'Peripheral {i}', f'Author {i}', 2021)
            self.graph.add_citation(paper_id, center_id)
        
        centrality = self.graph.get_paper_centrality(center_id)
        
        assert 'degree' in centrality
        assert 'betweenness' in centrality
        assert 'closeness' in centrality
        assert 'pagerank' in centrality
        
        # Center should have high degree centrality (most cited)
        assert centrality['degree'] > 0.5  # High relative to network size
        
        # Test non-existent paper
        empty_centrality = self.graph.get_paper_centrality('nonexistent')
        assert all(value == 0 for value in empty_centrality.values())
    
    def test_find_citation_paths(self):
        """Test finding citation paths between papers."""
        # Create a chain: p1 -> p2 -> p3 -> p4
        for i in range(4):
            self.graph.add_paper(f'p{i+1}', f'Paper {i+1}', f'Author {i+1}', 2020)
        
        for i in range(3):
            self.graph.add_citation(f'p{i+1}', f'p{i+2}')
        
        # Test path finding
        paths = self.graph.find_citation_paths('p1', 'p4', max_length=5)
        
        assert len(paths) > 0
        assert ['p1', 'p2', 'p3', 'p4'] in paths
        
        # Test non-existent path
        no_paths = self.graph.find_citation_paths('p4', 'p1', max_length=5)
        assert len(no_paths) == 0  # No reverse path in directed graph
    
    def test_get_subgraph(self):
        """Test subgraph extraction."""
        # Create test graph
        papers = ['center', 'neighbor1', 'neighbor2', 'distant']
        for paper_id in papers:
            self.graph.add_paper(paper_id, f'Paper {paper_id}', f'Author {paper_id}', 2020)
        
        # Connect center to neighbors
        self.graph.add_citation('neighbor1', 'center')
        self.graph.add_citation('neighbor2', 'center')
        # Distant paper not connected
        
        # Test 1-hop subgraph around center
        subgraph_nodes = self.graph.get_subgraph('center', hops=1)
        
        assert 'center' in subgraph_nodes
        assert 'neighbor1' in subgraph_nodes
        assert 'neighbor2' in subgraph_nodes
        assert 'distant' not in subgraph_nodes
        
        # Test 0-hop subgraph (just the node itself)
        single_node = self.graph.get_subgraph('center', hops=0)
        assert single_node == {'center'}
    
    def test_load_from_citation_matches(self):
        """Test loading graph from citation matches."""
        # Create citation matches
        citation = ExtractedCitation(
            raw_text="Smith, J. Test Paper. Conference 2020.",
            source_paper_id="citing_paper"
        )
        
        match = CitationMatch(
            citation=citation,
            paper_id="paper1",
            match_type="title",
            confidence=0.8
        )
        
        matches = [match]
        
        # Load into graph
        added_count = self.graph.load_from_citation_matches(matches)
        
        assert added_count == 1
        assert "citing_paper" in self.graph.paper_nodes
        assert "paper1" in self.graph.paper_nodes
        assert self.graph.has_citation("citing_paper", "paper1")
    
    def test_detect_communities(self):
        """Test community detection."""
        # Create two separate communities
        # Community 1: p1 -> p2 -> p3 (chain)
        # Community 2: p4 -> p5 -> p6 (chain)
        # Bridge: p3 -> p4 (connects communities)
        
        for i in range(6):
            self.graph.add_paper(f'p{i+1}', f'Paper {i+1}', f'Author {i+1}', 2020)
        
        # Community 1
        self.graph.add_citation('p1', 'p2')
        self.graph.add_citation('p2', 'p3')
        
        # Community 2
        self.graph.add_citation('p4', 'p5')
        self.graph.add_citation('p5', 'p6')
        
        # Bridge
        self.graph.add_citation('p3', 'p4')
        
        communities = self.graph.detect_communities()
        
        # Should detect some community structure
        assert len(communities) >= 1
        assert isinstance(communities, list)
        
        # Each community should be a set of paper IDs
        for community in communities:
            assert isinstance(community, set)
            assert len(community) > 0
    
    def test_export_to_networkx(self):
        """Test exporting to NetworkX graph."""
        # Create test graph
        self.graph.add_paper('p1', 'Paper 1', 'Author 1', 2020)
        self.graph.add_paper('p2', 'Paper 2', 'Author 2', 2021)
        self.graph.add_citation('p1', 'p2', confidence=0.9)
        
        nx_graph = self.graph.export_to_networkx()
        
        assert nx_graph.has_node('p1')
        assert nx_graph.has_node('p2')
        assert nx_graph.has_edge('p1', 'p2')
        
        # Check node attributes
        p1_attrs = nx_graph.nodes['p1']
        assert p1_attrs['title'] == 'Paper 1'
        assert p1_attrs['authors'] == 'Author 1'
        assert p1_attrs['year'] == 2020
        
        # Check edge attributes
        edge_attrs = nx_graph.edges['p1', 'p2']
        assert edge_attrs['confidence'] == 0.9
    
    def test_clear_graph(self):
        """Test clearing the graph."""
        # Add some data
        self.graph.add_paper('p1', 'Paper 1', 'Author 1', 2020)
        self.graph.add_paper('p2', 'Paper 2', 'Author 2', 2021)
        self.graph.add_citation('p1', 'p2')
        
        assert len(self.graph.paper_nodes) == 2
        assert len(self.graph.citation_edges) == 1
        
        # Clear graph
        self.graph.clear()
        
        assert len(self.graph.paper_nodes) == 0
        assert len(self.graph.citation_edges) == 0
        assert len(self.graph.nx_graph.nodes) == 0
        assert len(self.graph.nx_graph.edges) == 0

class TestPaperNode:
    """Test the PaperNode dataclass."""
    
    def test_paper_node_creation(self):
        """Test PaperNode creation and attributes."""
        node = PaperNode('test_id', 'Test Title', 'Test Author', 2023)
        
        assert node.paper_id == 'test_id'
        assert node.title == 'Test Title'
        assert node.authors == 'Test Author'
        assert node.year == 2023
        assert node.citation_count == 0
        assert node.venue is None
        assert node.keywords == []
    
    def test_paper_node_with_optional_fields(self):
        """Test PaperNode with optional fields."""
        node = PaperNode(
            paper_id='test_id',
            title='Test Title',
            authors='Test Author',
            year=2023,
            citation_count=5,
            venue='Test Conference',
            keywords=['AI', 'ML']
        )
        
        assert node.citation_count == 5
        assert node.venue == 'Test Conference'
        assert node.keywords == ['AI', 'ML']

class TestCitationEdge:
    """Test the CitationEdge dataclass."""
    
    def test_citation_edge_creation(self):
        """Test CitationEdge creation and attributes."""
        edge = CitationEdge('citing_id', 'cited_id')
        
        assert edge.citing_paper_id == 'citing_id'
        assert edge.cited_paper_id == 'cited_id'
        assert edge.confidence == 1.0
        assert edge.citation_context is None
        assert edge.section is None
    
    def test_citation_edge_with_optional_fields(self):
        """Test CitationEdge with optional fields."""
        edge = CitationEdge(
            citing_paper_id='citing_id',
            cited_paper_id='cited_id',
            confidence=0.8,
            citation_context='This work builds on previous research...',
            section='Introduction'
        )
        
        assert edge.confidence == 0.8
        assert edge.citation_context == 'This work builds on previous research...'
        assert edge.section == 'Introduction'

class TestGraphPerformance:
    """Test graph performance with larger datasets."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.setup_large_database()
        self.graph = CitationGraph(self.temp_db)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def setup_large_database(self):
        """Create database with more papers for performance testing."""
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                doi TEXT,
                arxiv_id TEXT,
                venue TEXT
            )
        """)
        
        # Create 100 test papers
        papers = []
        for i in range(100):
            papers.append((
                f'paper_{i:03d}',
                f'Research Paper {i}',
                f'Author {i % 10}; Coauthor {(i+1) % 10}',
                2020 + (i % 4),
                f'10.1000/{i}' if i % 3 == 0 else None,
                f'20{20 + i % 4}.{i:05d}' if i % 5 == 0 else None,
                f'Conference {i % 5}'
            ))
        
        cursor.executemany("""
            INSERT INTO papers (id, title, authors, year, doi, arxiv_id, venue)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, papers)
        
        conn.commit()
        conn.close()
    
    def test_large_graph_construction(self):
        """Test constructing a large graph."""
        import time
        
        start_time = time.time()
        
        # Add all papers
        for i in range(100):
            self.graph.add_paper(f'paper_{i:03d}', f'Paper {i}', f'Author {i}', 2020)
        
        construction_time = time.time() - start_time
        
        assert construction_time < 5, f"Graph construction took too long: {construction_time}s"
        assert len(self.graph.paper_nodes) == 100
    
    def test_large_graph_citations(self):
        """Test adding many citations to a graph."""
        import time
        
        # Add papers first
        for i in range(50):
            self.graph.add_paper(f'paper_{i:03d}', f'Paper {i}', f'Author {i}', 2020)
        
        start_time = time.time()
        
        # Add citations (each paper cites 2-3 previous papers)
        citation_count = 0
        for i in range(1, 50):
            for j in range(max(0, i-3), i):
                self.graph.add_citation(f'paper_{i:03d}', f'paper_{j:03d}')
                citation_count += 1
        
        citation_time = time.time() - start_time
        
        assert citation_time < 10, f"Citation addition took too long: {citation_time}s"
        assert len(self.graph.citation_edges) == citation_count
    
    def test_large_graph_metrics(self):
        """Test calculating metrics on a large graph."""
        import time
        
        # Build a reasonably large graph
        for i in range(30):
            self.graph.add_paper(f'paper_{i:03d}', f'Paper {i}', f'Author {i}', 2020)
        
        # Add citations
        for i in range(1, 30):
            for j in range(max(0, i-2), i):
                self.graph.add_citation(f'paper_{i:03d}', f'paper_{j:03d}')
        
        start_time = time.time()
        metrics = self.graph.calculate_metrics()
        metrics_time = time.time() - start_time
        
        assert metrics_time < 5, f"Metrics calculation took too long: {metrics_time}s"
        assert metrics['nodes'] == 30
        assert metrics['edges'] > 20

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
