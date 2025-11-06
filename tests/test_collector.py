"""
Comprehensive tests for the Paper Collector module.

Tests include unit tests for individual components and integration tests
for the complete workflow.
"""

import unittest
import tempfile
import shutil
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path

# Import the modules to test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collector.arxiv_client import ArxivClient
from collector.scholar_client import ScholarClient
from collector.database import PaperDatabase
from collector.collector import PaperCollector, RetryConfig


class TestArxivClient(unittest.TestCase):
    """Test the ArXiv API client."""
    
    def setUp(self):
        self.client = ArxivClient()
    
    def test_parse_entry(self):
        """Test parsing of ArXiv XML entry."""
        # Mock XML entry structure
        import xml.etree.ElementTree as ET
        
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2301.07041v1</id>
                <title>Test Paper Title</title>
                <summary>This is a test abstract.</summary>
                <author><name>John Doe</name></author>
                <author><name>Jane Smith</name></author>
                <published>2023-01-17T18:59:59Z</published>
                <updated>2023-01-17T18:59:59Z</updated>
                <category term="cs.AI" />
                <link href="http://arxiv.org/pdf/2301.07041v1.pdf" type="application/pdf" />
            </entry>
        </feed>"""
        
        root = ET.fromstring(xml_content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        
        paper = self.client._parse_entry(entry)
        
        self.assertEqual(paper['id'], '2301.07041v1')
        self.assertEqual(paper['title'], 'Test Paper Title')
        self.assertEqual(paper['authors'], ['John Doe', 'Jane Smith'])
        self.assertEqual(paper['abstract'], 'This is a test abstract.')
        self.assertEqual(paper['source'], 'arxiv')
        self.assertIsNotNone(paper['hash'])
    
    @patch('collector.arxiv_client.requests.Session.get')
    def test_search_success(self, mock_get):
        """Test successful search."""
        # Mock response with valid XML
        mock_response = Mock()
        mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
            <opensearch:totalResults>1</opensearch:totalResults>
            <entry>
                <id>http://arxiv.org/abs/2301.07041v1</id>
                <title>Test Paper</title>
                <summary>Test abstract</summary>
                <author><name>Test Author</name></author>
                <published>2023-01-17T18:59:59Z</published>
                <updated>2023-01-17T18:59:59Z</updated>
                <category term="cs.AI" />
                <link href="http://arxiv.org/pdf/2301.07041v1.pdf" type="application/pdf" />
            </entry>
        </feed>"""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        results = self.client.search("test query", max_results=10)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['title'], 'Test Paper')
        mock_get.assert_called_once()
    
    @patch('collector.arxiv_client.requests.Session.get')
    def test_search_no_results(self, mock_get):
        """Test search with no results."""
        mock_response = Mock()
        mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
            <opensearch:totalResults>0</opensearch:totalResults>
        </feed>"""
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        results = self.client.search("nonexistent query")
        
        self.assertEqual(len(results), 0)


class TestPaperDatabase(unittest.TestCase):
    """Test the paper database operations."""
    
    def setUp(self):
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = PaperDatabase(self.temp_db.name)
    
    def tearDown(self):
        # Clean up temporary database
        os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization creates required tables."""
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.cursor()
            
            # Check if papers table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check if indices exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_papers_hash'")
            self.assertIsNotNone(cursor.fetchone())
    
    def test_add_paper(self):
        """Test adding a paper to the database."""
        paper = {
            'id': 'test_001',
            'title': 'Test Paper',
            'authors': ['John Doe', 'Jane Smith'],
            'abstract': 'This is a test abstract.',
            'doi': '10.1000/test',
            'published_date': datetime.now(),
            'source': 'arxiv',
            'hash': 'test_hash_123'
        }
        
        result = self.db.add_paper(paper)
        self.assertTrue(result)
        
        # Verify paper exists
        self.assertTrue(self.db.paper_exists(paper_id='test_001'))
    
    def test_duplicate_detection(self):
        """Test duplicate paper detection."""
        paper = {
            'id': 'test_002',
            'title': 'Duplicate Test Paper',
            'authors': ['Author One'],
            'abstract': 'Test abstract.',
            'source': 'arxiv',
            'hash': 'duplicate_hash_456'
        }
        
        # Add paper first time
        result1 = self.db.add_paper(paper)
        self.assertTrue(result1)
        
        # Try to add same paper again
        result2 = self.db.add_paper(paper)
        self.assertFalse(result2)
    
    def test_search_papers(self):
        """Test searching papers in the database."""
        # Add test papers
        papers = [
            {
                'id': 'search_001',
                'title': 'Machine Learning Paper',
                'authors': ['ML Author'],
                'abstract': 'About machine learning algorithms.',
                'source': 'arxiv',
                'hash': 'ml_hash_1'
            },
            {
                'id': 'search_002', 
                'title': 'Deep Learning Research',
                'authors': ['DL Author'],
                'abstract': 'Deep neural networks study.',
                'source': 'scholar',
                'hash': 'dl_hash_2'
            }
        ]
        
        for paper in papers:
            self.db.add_paper(paper)
        
        # Search by title
        results = self.db.search_papers("machine learning")
        self.assertGreater(len(results), 0)
        
        # Search by source
        arxiv_results = self.db.search_papers(source="arxiv")
        self.assertEqual(len(arxiv_results), 1)
        self.assertEqual(arxiv_results[0]['id'], 'search_001')


class TestPaperCollector(unittest.TestCase):
    """Test the main paper collector orchestrator."""
    
    def setUp(self):
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.collector = PaperCollector(
            data_dir=self.temp_dir,
            db_path=os.path.join(self.temp_dir, "test_papers.db")
        )
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_collector_initialization(self):
        """Test collector initialization creates required directories."""
        self.assertTrue(os.path.exists(self.collector.papers_dir))
        self.assertTrue(os.path.exists(self.collector.metadata_dir))
        self.assertIsNotNone(self.collector.arxiv_client)
        self.assertIsNotNone(self.collector.database)
    
    @patch('collector.collector.requests.Session.get')
    def test_pdf_download(self, mock_get):
        """Test PDF download functionality."""
        # Mock PDF response
        mock_response = Mock()
        mock_response.headers = {'content-type': 'application/pdf'}
        mock_response.iter_content.return_value = [b'fake pdf content']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        pdf_path = self.collector._download_pdf('http://example.com/paper.pdf', 'test_paper')
        
        self.assertIsNotNone(pdf_path)
        self.assertTrue(os.path.exists(pdf_path))
    
    def test_save_metadata(self):
        """Test metadata saving functionality."""
        paper = {
            'id': 'meta_test_001',
            'title': 'Metadata Test Paper',
            'authors': ['Meta Author'],
            'published_date': datetime.now(),
            'source': 'arxiv'
        }
        
        result = self.collector._save_metadata(paper)
        self.assertTrue(result)
        
        # Check if metadata file was created
        metadata_file = self.collector.metadata_dir / f"{paper['id']}_metadata.json"
        self.assertTrue(metadata_file.exists())
        
        # Verify content
        with open(metadata_file, 'r') as f:
            saved_data = json.load(f)
            self.assertEqual(saved_data['title'], paper['title'])
    
    @patch.object(ArxivClient, 'search')
    @patch('collector.collector.ScholarClient')
    def test_search_integration(self, mock_scholar_class, mock_arxiv_search):
        """Test integration search functionality."""
        # Mock ArXiv search results
        mock_papers = [
            {
                'id': 'integration_001',
                'title': 'Integration Test Paper',
                'authors': ['Test Author'],
                'abstract': 'Integration test abstract.',
                'published_date': datetime.now(),
                'source': 'arxiv',
                'hash': 'integration_hash_1',
                'pdf_url': 'http://example.com/paper.pdf'
            }
        ]
        mock_arxiv_search.return_value = mock_papers
        
        # Mock Scholar client to return None (not available)
        mock_scholar_class.return_value = None
        
        # Mock PDF download and other network calls
        with patch.object(self.collector, '_download_pdf', return_value=None):
            with patch.object(self.collector, '_retry_operation', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
                results = self.collector.search("test query", max_results=10, download_pdfs=False)
        
        self.assertEqual(results['total_found'], 1)
        self.assertEqual(results['papers_added'], 1)
        self.assertEqual(results['papers_skipped'], 0)
    
    @patch.object(ArxivClient, 'get_recent_papers')
    def test_update_recent(self, mock_recent):
        """Test recent papers update functionality."""
        # Mock recent papers
        mock_papers = [
            {
                'id': 'recent_001',
                'title': 'Recent Paper',
                'authors': ['Recent Author'],
                'abstract': 'Recent paper abstract.',
                'published_date': datetime.now(),
                'source': 'arxiv',
                'hash': 'recent_hash_1'
            }
        ]
        mock_recent.return_value = mock_papers
        
        # Mock the retry operation to avoid network calls
        with patch.object(self.collector, '_retry_operation', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            results = self.collector.update_recent("cs.AI", days_back=7, download_pdfs=False)
        
        self.assertEqual(results['total_found'], 1)
        self.assertEqual(results['papers_added'], 1)
    
    def test_retry_configuration(self):
        """Test retry configuration and exponential backoff."""
        retry_config = RetryConfig(max_retries=3, base_delay=0.1, max_delay=1.0)
        
        # Test delay calculation
        delay1 = retry_config.get_delay(0)  # First attempt
        delay2 = retry_config.get_delay(1)  # Second attempt
        delay3 = retry_config.get_delay(2)  # Third attempt
        
        # Delays should increase (with some jitter)
        self.assertGreater(delay2, delay1 * 1.5)  # Account for jitter
        self.assertGreater(delay3, delay2 * 1.5)
        self.assertLessEqual(delay3, retry_config.max_delay * 1.5)  # With jitter


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.collector = PaperCollector(
            data_dir=self.temp_dir,
            db_path=os.path.join(self.temp_dir, "integration_test.db")
        )
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch.object(ArxivClient, 'search')
    @patch('collector.collector.scholar_available', return_value=False)
    def test_complete_workflow(self, mock_scholar_available, mock_arxiv_search):
        """Test complete workflow: search -> deduplicate -> store -> retrieve."""
        # Mock search results with potential duplicates
        mock_papers = [
            {
                'id': 'workflow_001',
                'title': 'Workflow Test Paper 1',
                'authors': ['Author 1'],
                'abstract': 'First test paper.',
                'published_date': datetime.now(),
                'source': 'arxiv',
                'hash': 'workflow_hash_1'
            },
            {
                'id': 'workflow_002',
                'title': 'Workflow Test Paper 2', 
                'authors': ['Author 2'],
                'abstract': 'Second test paper.',
                'published_date': datetime.now(),
                'source': 'arxiv',
                'hash': 'workflow_hash_2'
            },
            # Duplicate of first paper (same hash)
            {
                'id': 'workflow_001_dup',
                'title': 'Workflow Test Paper 1',
                'authors': ['Author 1'],
                'abstract': 'First test paper.',
                'published_date': datetime.now(),
                'source': 'arxiv',
                'hash': 'workflow_hash_1'  # Same hash as first paper
            }
        ]
        mock_arxiv_search.return_value = mock_papers
        
        # Mock the retry operation to avoid network calls
        with patch.object(self.collector, '_retry_operation', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
            # Execute search
            results = self.collector.search("test workflow", download_pdfs=False)
        
        # Verify results
        self.assertEqual(results['total_found'], 3)
        self.assertEqual(results['papers_added'], 2)  # Only 2 unique papers
        self.assertEqual(results['papers_skipped'], 1)  # 1 duplicate
        
        # Verify papers are in database
        stored_papers = self.collector.search_local("workflow")
        self.assertEqual(len(stored_papers), 2)
        
        # Verify deduplication worked
        paper_ids = [p['id'] for p in stored_papers]
        self.assertIn('workflow_001', paper_ids)
        self.assertIn('workflow_002', paper_ids)
        self.assertNotIn('workflow_001_dup', paper_ids)
    
    def test_error_handling(self):
        """Test error handling and recovery."""
        # Test with invalid database path (permission denied)
        with self.assertRaises(Exception):
            invalid_collector = PaperCollector(db_path="/invalid/path/papers.db")
    
    def test_statistics_collection(self):
        """Test statistics collection functionality."""
        # Add some test data
        test_papers = [
            {
                'id': 'stats_001',
                'title': 'Stats Test Paper 1',
                'authors': ['Stats Author 1'],
                'source': 'arxiv',
                'hash': 'stats_hash_1'
            },
            {
                'id': 'stats_002',
                'title': 'Stats Test Paper 2',
                'authors': ['Stats Author 2'],
                'source': 'scholar',
                'hash': 'stats_hash_2'
            }
        ]
        
        for paper in test_papers:
            self.collector.database.add_paper(paper)
        
        # Get statistics
        stats = self.collector.get_stats()
        
        self.assertEqual(stats['total_papers'], 2)
        self.assertIn('by_source', stats)
        self.assertEqual(stats['by_source']['arxiv'], 1)
        self.assertEqual(stats['by_source']['scholar'], 1)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestArxivClient,
        TestPaperDatabase,
        TestPaperCollector,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
