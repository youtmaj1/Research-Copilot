"""
Tests for Graph Exporter Module

This module tests the graph export functionality including:
- JSON format export
- GraphML format export  
- CSV format export
- Neo4j database export
- Export validation and error handling
- Large graph export performance
"""

import pytest
import tempfile
import sqlite3
import json
import os
import csv
from pathlib import Path
from xml.etree import ElementTree as ET

import sys
sys.path.append(str(Path(__file__).parent.parent))

from exporter import GraphExporter, Neo4jExporter, GraphMLExporter
from graph import CitationGraph, PaperNode, CitationEdge

class TestGraphExporter:
    """Test the main GraphExporter class."""
    
    def setup_method(self):
        """Set up test environment with sample graph."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.setup_test_database()
        self.graph = CitationGraph(self.temp_db)
        self.exporter = GraphExporter()
        self.setup_sample_graph()
    
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
                venue TEXT,
                doi TEXT,
                abstract TEXT
            )
        """)
        
        test_papers = [
            ('paper1', 'Foundation Paper', 'Smith, J.; Jones, A.', 2020, 'Nature', '10.1000/1', 'Foundational research'),
            ('paper2', 'Building on Foundations', 'Brown, B.', 2021, 'Science', '10.1000/2', 'Extended work'),
            ('paper3', 'Advanced Methods', 'Davis, C.; Wilson, D.', 2022, 'Cell', None, 'Advanced techniques'),
            ('paper4', 'Recent Developments', 'Taylor, E.', 2023, 'PNAS', '10.1000/4', 'Latest findings'),
        ]
        
        cursor.executemany("""
            INSERT INTO papers (id, title, authors, year, venue, doi, abstract)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, test_papers)
        
        conn.commit()
        conn.close()
    
    def setup_sample_graph(self):
        """Create a sample citation graph for testing."""
        # Add papers to graph
        papers_data = [
            ('paper1', 'Foundation Paper', 'Smith, J.; Jones, A.', 2020),
            ('paper2', 'Building on Foundations', 'Brown, B.', 2021),
            ('paper3', 'Advanced Methods', 'Davis, C.; Wilson, D.', 2022),
            ('paper4', 'Recent Developments', 'Taylor, E.', 2023),
        ]
        
        for paper_id, title, authors, year in papers_data:
            self.graph.add_paper(paper_id, title, authors, year)
        
        # Add citation relationships
        citations = [
            ('paper2', 'paper1', 0.95),  # paper2 cites paper1
            ('paper3', 'paper1', 0.90),  # paper3 cites paper1
            ('paper3', 'paper2', 0.85),  # paper3 cites paper2
            ('paper4', 'paper1', 0.88),  # paper4 cites paper1
            ('paper4', 'paper3', 0.92),  # paper4 cites paper3
        ]
        
        for citing, cited, confidence in citations:
            self.graph.add_citation(citing, cited, confidence)
    
    def test_exporter_initialization(self):
        """Test GraphExporter initialization."""
        assert self.exporter is not None
        assert hasattr(self.exporter, 'export_to_json')
        assert hasattr(self.exporter, 'export_to_graphml')
        assert hasattr(self.exporter, 'export_to_csv')
        assert hasattr(self.exporter, 'export_to_neo4j')

class TestJSONExport:
    """Test JSON format export functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.setup_test_database()
        self.graph = CitationGraph(self.temp_db)
        self.exporter = GraphExporter()
        self.setup_sample_graph()
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def setup_test_database(self):
        """Create test database."""
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def setup_sample_graph(self):
        """Create sample graph."""
        self.graph.add_paper('p1', 'Paper 1', 'Author 1', 2020)
        self.graph.add_paper('p2', 'Paper 2', 'Author 2', 2021)
        self.graph.add_citation('p2', 'p1', confidence=0.9)
    
    def test_json_export_basic(self):
        """Test basic JSON export functionality."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            success = self.exporter.export_to_json(self.graph, export_path)
            assert success, "JSON export should succeed"
            
            # Verify file exists
            assert os.path.exists(export_path)
            
            # Load and verify JSON structure
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            assert 'metadata' in data
            assert 'graph' in data
            assert 'nodes' in data['graph']
            assert 'edges' in data['graph']
            
            # Check metadata
            metadata = data['metadata']
            assert 'export_timestamp' in metadata
            assert 'total_nodes' in metadata
            assert 'total_edges' in metadata
            assert metadata['total_nodes'] == 2
            assert metadata['total_edges'] == 1
            
            # Check nodes
            nodes = data['graph']['nodes']
            assert len(nodes) == 2
            
            node_ids = {node['id'] for node in nodes}
            assert 'p1' in node_ids
            assert 'p2' in node_ids
            
            # Verify node attributes
            p1_node = next(node for node in nodes if node['id'] == 'p1')
            assert p1_node['title'] == 'Paper 1'
            assert p1_node['authors'] == 'Author 1'
            assert p1_node['year'] == 2020
            
            # Check edges
            edges = data['graph']['edges']
            assert len(edges) == 1
            
            edge = edges[0]
            assert edge['source'] == 'p2'
            assert edge['target'] == 'p1'
            assert edge['confidence'] == 0.9
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
    
    def test_json_export_with_metrics(self):
        """Test JSON export including graph metrics."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            success = self.exporter.export_to_json(
                self.graph, 
                export_path, 
                include_metrics=True
            )
            assert success
            
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            assert 'metrics' in data
            metrics = data['metrics']
            
            assert 'nodes' in metrics
            assert 'edges' in metrics
            assert 'density' in metrics
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
    
    def test_json_export_empty_graph(self):
        """Test JSON export with empty graph."""
        empty_graph = CitationGraph(self.temp_db)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            success = self.exporter.export_to_json(empty_graph, export_path)
            assert success
            
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            assert data['metadata']['total_nodes'] == 0
            assert data['metadata']['total_edges'] == 0
            assert len(data['graph']['nodes']) == 0
            assert len(data['graph']['edges']) == 0
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)

class TestGraphMLExport:
    """Test GraphML format export functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.setup_test_database()
        self.graph = CitationGraph(self.temp_db)
        self.exporter = GraphExporter()
        self.setup_sample_graph()
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def setup_test_database(self):
        """Create test database."""
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def setup_sample_graph(self):
        """Create sample graph."""
        self.graph.add_paper('p1', 'Paper 1', 'Author 1', 2020)
        self.graph.add_paper('p2', 'Paper 2', 'Author 2', 2021)
        self.graph.add_citation('p2', 'p1', confidence=0.8)
    
    def test_graphml_export_basic(self):
        """Test basic GraphML export functionality."""
        with tempfile.NamedTemporaryFile(suffix='.graphml', delete=False) as f:
            export_path = f.name
        
        try:
            success = self.exporter.export_to_graphml(self.graph, export_path)
            assert success, "GraphML export should succeed"
            
            # Verify file exists
            assert os.path.exists(export_path)
            
            # Parse and verify XML structure
            tree = ET.parse(export_path)
            root = tree.getroot()
            
            # Check GraphML namespace
            assert 'graphml' in root.tag.lower()
            
            # Find graph element
            graph_elem = root.find('.//*[local-name()="graph"]')
            assert graph_elem is not None
            
            # Find nodes and edges
            nodes = root.findall('.//*[local-name()="node"]')
            edges = root.findall('.//*[local-name()="edge"]')
            
            assert len(nodes) == 2
            assert len(edges) == 1
            
            # Check node IDs
            node_ids = {node.get('id') for node in nodes}
            assert 'p1' in node_ids
            assert 'p2' in node_ids
            
            # Check edge
            edge = edges[0]
            assert edge.get('source') == 'p2'
            assert edge.get('target') == 'p1'
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
    
    def test_graphml_export_with_attributes(self):
        """Test GraphML export with node and edge attributes."""
        with tempfile.NamedTemporaryFile(suffix='.graphml', delete=False) as f:
            export_path = f.name
        
        try:
            success = self.exporter.export_to_graphml(self.graph, export_path)
            assert success
            
            # Parse XML and check for attribute definitions
            tree = ET.parse(export_path)
            root = tree.getroot()
            
            # Look for key definitions (attribute declarations)
            keys = root.findall('.//*[local-name()="key"]')
            key_names = {key.get('attr.name') for key in keys}
            
            # Should have node attributes
            assert 'title' in key_names
            assert 'authors' in key_names
            assert 'year' in key_names
            
            # Should have edge attributes
            assert 'confidence' in key_names
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)

class TestCSVExport:
    """Test CSV format export functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.setup_test_database()
        self.graph = CitationGraph(self.temp_db)
        self.exporter = GraphExporter()
        self.setup_sample_graph()
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def setup_test_database(self):
        """Create test database."""
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def setup_sample_graph(self):
        """Create sample graph."""
        self.graph.add_paper('paper1', 'First Paper', 'Smith, J.', 2020)
        self.graph.add_paper('paper2', 'Second Paper', 'Jones, A.', 2021)
        self.graph.add_paper('paper3', 'Third Paper', 'Brown, B.', 2022)
        self.graph.add_citation('paper2', 'paper1', confidence=0.9)
        self.graph.add_citation('paper3', 'paper1', confidence=0.8)
    
    def test_csv_export_basic(self):
        """Test basic CSV export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            success = self.exporter.export_to_csv(self.graph, temp_dir)
            assert success, "CSV export should succeed"
            
            # Check for expected files
            papers_file = os.path.join(temp_dir, 'papers.csv')
            citations_file = os.path.join(temp_dir, 'citations.csv')
            
            assert os.path.exists(papers_file)
            assert os.path.exists(citations_file)
            
            # Verify papers CSV
            with open(papers_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                papers = list(reader)
            
            assert len(papers) == 3
            paper_ids = {paper['paper_id'] for paper in papers}
            assert 'paper1' in paper_ids
            assert 'paper2' in paper_ids
            assert 'paper3' in paper_ids
            
            # Check paper attributes
            paper1 = next(p for p in papers if p['paper_id'] == 'paper1')
            assert paper1['title'] == 'First Paper'
            assert paper1['authors'] == 'Smith, J.'
            assert paper1['year'] == '2020'
            
            # Verify citations CSV
            with open(citations_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                citations = list(reader)
            
            assert len(citations) == 2
            
            # Check citation data
            citation1 = citations[0]
            assert 'citing_paper_id' in citation1
            assert 'cited_paper_id' in citation1
            assert 'confidence' in citation1
    
    def test_csv_export_custom_filenames(self):
        """Test CSV export with custom filenames."""
        with tempfile.TemporaryDirectory() as temp_dir:
            success = self.exporter.export_to_csv(
                self.graph, 
                temp_dir,
                papers_filename='nodes.csv',
                citations_filename='edges.csv'
            )
            assert success
            
            # Check for custom filenames
            assert os.path.exists(os.path.join(temp_dir, 'nodes.csv'))
            assert os.path.exists(os.path.join(temp_dir, 'edges.csv'))

class TestNeo4jExport:
    """Test Neo4j database export functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.setup_test_database()
        self.graph = CitationGraph(self.temp_db)
        self.exporter = GraphExporter()
        self.setup_sample_graph()
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def setup_test_database(self):
        """Create test database."""
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def setup_sample_graph(self):
        """Create sample graph."""
        self.graph.add_paper('p1', 'Paper 1', 'Author 1', 2020)
        self.graph.add_paper('p2', 'Paper 2', 'Author 2', 2021)
        self.graph.add_citation('p2', 'p1')
    
    def test_neo4j_export_connection_handling(self):
        """Test Neo4j export connection handling (without actual Neo4j)."""
        # This test checks the export method without requiring Neo4j
        # In a real environment, this would need a Neo4j instance
        
        # Test with invalid connection parameters
        success = self.exporter.export_to_neo4j(
            self.graph,
            uri="bolt://localhost:7687",
            username="neo4j",
            password="invalid"
        )
        
        # Should fail gracefully without Neo4j
        assert success == False  # Expected to fail without real Neo4j
    
    def test_neo4j_exporter_initialization(self):
        """Test Neo4jExporter initialization."""
        try:
            exporter = Neo4jExporter("bolt://localhost:7687", "neo4j", "password")
            # If neo4j package is available, this should work
            assert exporter is not None
        except ImportError:
            # Expected if neo4j package not installed
            pytest.skip("Neo4j package not available")

class TestExportErrorHandling:
    """Test error handling in export operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.setup_test_database()
        self.graph = CitationGraph(self.temp_db)
        self.exporter = GraphExporter()
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def setup_test_database(self):
        """Create test database."""
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def test_export_to_invalid_path(self):
        """Test export to invalid file paths."""
        # Add some data to graph
        self.graph.add_paper('p1', 'Paper 1', 'Author 1', 2020)
        
        # Test JSON export to invalid path
        invalid_path = '/nonexistent/directory/file.json'
        success = self.exporter.export_to_json(self.graph, invalid_path)
        assert success == False
        
        # Test GraphML export to invalid path
        success = self.exporter.export_to_graphml(self.graph, invalid_path)
        assert success == False
    
    def test_export_empty_graph(self):
        """Test exporting empty graphs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, 'empty.json')
            
            # Should succeed with empty graph
            success = self.exporter.export_to_json(self.graph, json_path)
            assert success
            
            # Verify empty graph structure
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            assert data['metadata']['total_nodes'] == 0
            assert data['metadata']['total_edges'] == 0
    
    def test_csv_export_to_readonly_directory(self):
        """Test CSV export error handling with permission issues."""
        self.graph.add_paper('p1', 'Paper 1', 'Author 1', 2020)
        
        # This test would require a read-only directory
        # For now, just test that the method handles errors gracefully
        with tempfile.TemporaryDirectory() as temp_dir:
            # Make a subdirectory that we'll pretend is read-only
            readonly_dir = os.path.join(temp_dir, 'readonly')
            os.makedirs(readonly_dir)
            
            # The method should handle errors gracefully
            # Even if it can't actually make the directory read-only in this test
            success = self.exporter.export_to_csv(self.graph, readonly_dir)
            # This might succeed in test environment, but the method should handle failures

class TestExportPerformance:
    """Test export performance with larger graphs."""
    
    def setup_method(self):
        """Set up test environment with larger graph."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.setup_large_database()
        self.graph = CitationGraph(self.temp_db)
        self.exporter = GraphExporter()
        self.setup_large_graph()
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def setup_large_database(self):
        """Create database for performance testing."""
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def setup_large_graph(self):
        """Create larger graph for performance testing."""
        # Create 100 papers
        for i in range(100):
            self.graph.add_paper(
                f'paper_{i:03d}',
                f'Paper {i}',
                f'Author {i % 20}',
                2020 + (i % 4)
            )
        
        # Add citations (each paper cites 2-3 previous papers)
        for i in range(1, 100):
            for j in range(max(0, i-3), i):
                if (i + j) % 3 == 0:  # Add some randomness
                    self.graph.add_citation(f'paper_{i:03d}', f'paper_{j:03d}')
    
    def test_large_json_export_performance(self):
        """Test JSON export performance with large graph."""
        import time
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            start_time = time.time()
            success = self.exporter.export_to_json(self.graph, export_path)
            export_time = time.time() - start_time
            
            assert success
            assert export_time < 5, f"Large JSON export took too long: {export_time}s"
            
            # Verify file size is reasonable
            file_size = os.path.getsize(export_path)
            assert file_size > 1000, "Export file should contain substantial data"
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
    
    def test_large_csv_export_performance(self):
        """Test CSV export performance with large graph."""
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            start_time = time.time()
            success = self.exporter.export_to_csv(self.graph, temp_dir)
            export_time = time.time() - start_time
            
            assert success
            assert export_time < 5, f"Large CSV export took too long: {export_time}s"
            
            # Verify files exist and have content
            papers_file = os.path.join(temp_dir, 'papers.csv')
            citations_file = os.path.join(temp_dir, 'citations.csv')
            
            assert os.path.exists(papers_file)
            assert os.path.exists(citations_file)
            assert os.path.getsize(papers_file) > 1000
    
    def test_multiple_format_export(self):
        """Test exporting the same graph to multiple formats."""
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            start_time = time.time()
            
            # Export to all formats
            json_path = os.path.join(temp_dir, 'graph.json')
            graphml_path = os.path.join(temp_dir, 'graph.graphml')
            csv_dir = os.path.join(temp_dir, 'csv')
            os.makedirs(csv_dir)
            
            json_success = self.exporter.export_to_json(self.graph, json_path)
            graphml_success = self.exporter.export_to_graphml(self.graph, graphml_path)
            csv_success = self.exporter.export_to_csv(self.graph, csv_dir)
            
            total_time = time.time() - start_time
            
            assert json_success
            assert graphml_success
            assert csv_success
            assert total_time < 10, f"Multiple exports took too long: {total_time}s"
            
            # Verify all files exist
            assert os.path.exists(json_path)
            assert os.path.exists(graphml_path)
            assert os.path.exists(os.path.join(csv_dir, 'papers.csv'))

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
