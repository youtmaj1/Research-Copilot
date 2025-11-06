"""
Integration Tests for Citation Tracker Module

This module contains end-to-end integration tests that test the complete
citation tracking workflow from extraction to graph analysis.

Test Coverage:
- Full pipeline: PDF -> extraction -> resolution -> graph -> analysis
- Database integration tests
- Performance tests with realistic data volumes
- Error handling and recovery scenarios
"""

import pytest
import tempfile
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

import sys
sys.path.append(str(Path(__file__).parent.parent))

from extractor import CitationExtractor, ExtractedCitation
from resolver import CitationResolver
from graph import CitationGraph
from temporal import TimeSeriesAnalyzer
from exporter import GraphExporter
from database_schema import create_citation_tables, verify_schema

class TestFullPipeline:
    """Test the complete citation tracking pipeline."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create temporary database
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.setup_test_database()
        
        # Initialize components
        self.extractor = CitationExtractor()
        self.resolver = CitationResolver(self.temp_db)
        self.graph = CitationGraph(self.temp_db)
        self.temporal_analyzer = TimeSeriesAnalyzer(self.temp_db, self.graph)
        self.exporter = GraphExporter()
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def setup_test_database(self):
        """Set up test database with sample papers and citation schema."""
        # Create basic papers table
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
                venue TEXT,
                abstract TEXT
            )
        """)
        
        # Insert comprehensive test papers
        test_papers = [
            ('paper1', 'Machine Learning for Citation Analysis', 'Smith, J.; Jones, A.', 2023, '10.1000/123', None, 'ICML', 'ML citation analysis abstract'),
            ('paper2', 'Deep Learning Networks', 'Brown, B.; Davis, C.', 2022, None, '2201.12345', 'NeurIPS', 'Deep learning abstract'),
            ('paper3', 'Graph Neural Networks', 'Wilson, D.; Taylor, E.', 2021, '10.1000/456', '2101.67890', 'ICLR', 'GNN abstract'),
            ('paper4', 'Attention Mechanisms in AI', 'Garcia, F.; Lee, G.', 2020, '10.1000/789', None, 'AAAI', 'Attention mechanisms abstract'),
            ('paper5', 'Natural Language Processing', 'Johnson, H.; Kim, I.', 2019, None, None, 'ACL', 'NLP abstract'),
            ('paper6', 'Computer Vision Methods', 'Anderson, K.; Martinez, L.', 2023, '10.1000/999', None, 'CVPR', 'Computer vision abstract'),
            ('paper7', 'Reinforcement Learning', 'Thompson, M.; White, N.', 2022, None, '2202.54321', 'ICML', 'RL abstract'),
        ]
        
        cursor.executemany("""
            INSERT INTO papers (id, title, authors, year, doi, arxiv_id, venue, abstract)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, test_papers)
        
        conn.commit()
        conn.close()
        
        # Create citation tracking schema
        create_citation_tables(self.temp_db)
    
    def test_complete_workflow(self):
        """Test the complete citation tracking workflow."""
        # Step 1: Extract citations from sample text
        sample_paper_text = """
        References

        [1] Smith, J. and Jones, A. Machine Learning for Citation Analysis. 
        In Proceedings of ICML 2023. doi:10.1000/123

        [2] Brown, B., Davis, C. Deep Learning Networks. NeurIPS 2022. arXiv:2201.12345

        [3] Wilson, D., Taylor, E. Graph Neural Networks: A Survey. 
        ICLR 2021. doi:10.1000/456, arXiv:2101.67890

        [4] Garcia, F.; Lee, G. (2020). Attention Mechanisms in AI. AAAI Conference. DOI: 10.1000/789

        [5] New Paper Author. Novel Citation Method. Future Conference 2024. doi:10.1000/new
        """
        
        # Extract citations
        extracted_citations = self.extractor.extract_citations_from_text(
            sample_paper_text, "test_citing_paper"
        )
        
        assert len(extracted_citations) >= 4, f"Expected at least 4 citations, got {len(extracted_citations)}"
        
        # Step 2: Resolve citations to database papers
        citation_matches = self.resolver.resolve_citations(extracted_citations)
        
        assert len(citation_matches) >= 3, f"Expected at least 3 matches, got {len(citation_matches)}"
        
        # Verify we got expected matches
        match_types = [match.match_type for match in citation_matches]
        assert "doi" in match_types, "Should have DOI matches"
        assert "arxiv" in match_types, "Should have arXiv matches"
        
        # Step 3: Build citation graph
        self.graph.load_from_citation_matches(citation_matches)
        
        # Verify graph structure
        assert len(self.graph.paper_nodes) >= 4, "Graph should have multiple nodes"
        assert len(self.graph.citation_edges) >= 3, "Graph should have multiple edges"
        
        # Step 4: Calculate graph metrics
        metrics = self.graph.calculate_metrics()
        
        assert 'nodes' in metrics
        assert 'edges' in metrics
        assert metrics['nodes'] >= 4
        assert metrics['edges'] >= 3
        
        # Step 5: Temporal analysis
        # Record some citation snapshots
        for paper_id in self.graph.paper_nodes.keys():
            citation_count = self.graph.paper_nodes[paper_id].citation_count
            self.temporal_analyzer.record_citation_snapshot(paper_id, citation_count)
        
        # Analyze trends
        trending_papers = self.temporal_analyzer.analyze_trends(30)
        
        # Should have some trending papers (even with minimal data)
        assert isinstance(trending_papers, list)
        
        # Step 6: Export graph
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            success = self.exporter.export_to_json(self.graph, export_path)
            assert success, "Graph export should succeed"
            
            # Verify exported file
            assert os.path.exists(export_path)
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            assert 'graph' in exported_data
            assert 'nodes' in exported_data['graph']
            assert 'edges' in exported_data['graph']
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
    
    def test_database_integration(self):
        """Test integration with database schema."""
        # Verify citation schema
        schema_info = verify_schema(self.temp_db)
        assert schema_info['schema_exists'], "Citation schema should exist"
        assert len(schema_info['missing_tables']) == 0, "No tables should be missing"
        
        # Test database operations
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Insert a test citation
        cursor.execute("""
            INSERT INTO extracted_citations 
            (source_paper_id, raw_text, doi, title, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, ('test_paper', 'Test citation text', '10.1000/test', 'Test Title', 0.8))
        
        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM extracted_citations")
        count = cursor.fetchone()[0]
        assert count >= 1, "Citation should be inserted"
        
        conn.commit()
        conn.close()
    
    def test_error_handling(self):
        """Test error handling throughout the pipeline."""
        # Test with malformed input
        bad_text = "This is not a proper reference section"
        
        # Extraction should handle gracefully
        citations = self.extractor.extract_citations_from_text(bad_text, "test_paper")
        assert isinstance(citations, list)  # Should return empty list, not error
        
        # Resolution should handle empty input
        matches = self.resolver.resolve_citations([])
        assert matches == []
        
        # Graph should handle empty matches
        added_count = self.graph.load_from_citation_matches([])
        assert added_count == 0
    
    def test_performance_with_realistic_data(self):
        """Test performance with realistic data volumes."""
        # Generate realistic citation text
        large_citation_text = "References\n\n"
        for i in range(50):
            large_citation_text += f"""
[{i+1}] Author{i}, X.; Coauthor{i}, Y. Paper Title {i}: A Study of Topic {i}.
Conference {i % 10} {2020 + i % 4}. doi:10.1000/{i}
"""
        
        # Measure extraction performance
        start_time = datetime.now()
        citations = self.extractor.extract_citations_from_text(large_citation_text, "performance_test")
        extraction_time = (datetime.now() - start_time).total_seconds()
        
        assert extraction_time < 5, f"Extraction took too long: {extraction_time} seconds"
        assert len(citations) >= 25, "Should extract most citations"
        
        # Measure resolution performance
        start_time = datetime.now()
        matches = self.resolver.resolve_citations(citations[:20])  # Limit to avoid too long test
        resolution_time = (datetime.now() - start_time).total_seconds()
        
        assert resolution_time < 10, f"Resolution took too long: {resolution_time} seconds"
        
        # Measure graph building performance
        start_time = datetime.now()
        self.graph.load_from_citation_matches(matches)
        graph_time = (datetime.now() - start_time).total_seconds()
        
        assert graph_time < 3, f"Graph building took too long: {graph_time} seconds"
    
    def test_data_consistency(self):
        """Test data consistency across components."""
        # Create test data
        citation_text = """
        References
        [1] Smith, J. Test Paper. Journal 2023. doi:10.1000/123
        [2] Brown, B. Another Paper. Conference 2022. arXiv:2201.12345
        """
        
        # Extract and resolve
        citations = self.extractor.extract_citations_from_text(citation_text, "consistency_test")
        matches = self.resolver.resolve_citations(citations)
        
        # Build graph
        self.graph.load_from_citation_matches(matches)
        
        # Check consistency between components
        for match in matches:
            # Graph should contain the matched paper
            assert match.paper_id in self.graph.paper_nodes, f"Graph missing paper: {match.paper_id}"
            
            # Graph should contain the citation edge
            citing_id = match.citation.source_paper_id
            cited_id = match.paper_id
            
            if citing_id and cited_id:
                assert self.graph.has_citation(citing_id, cited_id), \
                    f"Graph missing citation: {citing_id} -> {cited_id}"
    
    def test_temporal_integration(self):
        """Test temporal analysis integration."""
        # Set up some historical data
        test_papers = ['paper1', 'paper2', 'paper3']
        base_date = datetime.now() - timedelta(days=60)
        
        # Record historical snapshots
        for i, paper_id in enumerate(test_papers):
            for day_offset in [0, 15, 30, 45, 60]:
                timestamp = base_date + timedelta(days=day_offset)
                citation_count = 10 + i * 5 + day_offset // 10  # Simulated growth
                self.temporal_analyzer.record_citation_snapshot(
                    paper_id, citation_count, timestamp
                )
        
        # Analyze trends
        trending_papers = self.temporal_analyzer.analyze_trends(45)
        
        assert len(trending_papers) > 0, "Should identify trending papers"
        
        # Test citation forecasting
        forecast = self.temporal_analyzer.get_citation_forecast('paper1', 30)
        assert 'forecasted_total_citations' in forecast
        assert forecast['forecasted_total_citations'] > 0
    
    def test_export_integration(self):
        """Test export functionality integration."""
        # Build a simple graph
        self.graph.add_paper('test1', 'Test Paper 1', 'Author A', 2023)
        self.graph.add_paper('test2', 'Test Paper 2', 'Author B', 2022)
        self.graph.add_citation('test1', 'test2', confidence=0.9)
        
        # Test multiple export formats
        with tempfile.TemporaryDirectory() as temp_dir:
            # JSON export
            json_path = os.path.join(temp_dir, 'test_graph.json')
            success = self.exporter.export_to_json(self.graph, json_path)
            assert success, "JSON export should succeed"
            assert os.path.exists(json_path)
            
            # GraphML export
            graphml_path = os.path.join(temp_dir, 'test_graph.graphml')
            success = self.exporter.export_to_graphml(self.graph, graphml_path)
            assert success, "GraphML export should succeed"
            assert os.path.exists(graphml_path)
            
            # CSV export
            csv_dir = os.path.join(temp_dir, 'csv')
            success = self.exporter.export_to_csv(self.graph, csv_dir)
            assert success, "CSV export should succeed"
            assert os.path.exists(os.path.join(csv_dir, 'papers.csv'))
            assert os.path.exists(os.path.join(csv_dir, 'citations.csv'))

class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.setup_realistic_database()
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def setup_realistic_database(self):
        """Set up database with realistic research papers."""
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Create papers table
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
        
        # Add realistic ML/AI papers
        realistic_papers = [
            ('attention2017', 'Attention Is All You Need', 'Vaswani, A.; Shazeer, N.; Parmar, N.', 2017, None, '1706.03762', 'NeurIPS', 'Transformer architecture paper'),
            ('bert2018', 'BERT: Pre-training of Deep Bidirectional Transformers', 'Devlin, J.; Chang, M.W.; Lee, K.', 2018, None, '1810.04805', 'NAACL', 'BERT language model'),
            ('gpt2019', 'Language Models are Unsupervised Multitask Learners', 'Radford, A.; Wu, J.; Child, R.', 2019, None, None, 'OpenAI', 'GPT-2 language model'),
            ('resnet2016', 'Deep Residual Learning for Image Recognition', 'He, K.; Zhang, X.; Ren, S.', 2016, '10.1109/CVPR.2016.90', None, 'CVPR', 'ResNet architecture'),
            ('dropout2014', 'Dropout: A Simple Way to Prevent Neural Networks from Overfitting', 'Srivastava, N.; Hinton, G.; Krizhevsky, A.', 2014, None, None, 'JMLR', 'Dropout regularization'),
        ]
        
        cursor.executemany("""
            INSERT INTO papers (id, title, authors, year, doi, arxiv_id, venue, abstract)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, realistic_papers)
        
        conn.commit()
        conn.close()
        
        # Create citation schema
        create_citation_tables(self.temp_db)
    
    def test_ml_paper_citation_analysis(self):
        """Test citation analysis on realistic ML papers."""
        # Simulate a new paper citing existing ones
        new_paper_refs = """
        References

        [1] Vaswani, A., Shazeer, N., Parmar, N., et al. Attention Is All You Need. 
        In Advances in Neural Information Processing Systems (NeurIPS), 2017. arXiv:1706.03762

        [2] Devlin, J., Chang, M.W., Lee, K., and Toutanova, K. BERT: Pre-training of Deep 
        Bidirectional Transformers for Language Understanding. NAACL 2018. arXiv:1810.04805

        [3] He, K., Zhang, X., Ren, S., and Sun, J. Deep Residual Learning for Image Recognition. 
        IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. doi:10.1109/CVPR.2016.90

        [4] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. 
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting. 
        Journal of Machine Learning Research (JMLR), 2014.
        """
        
        # Run complete pipeline
        extractor = CitationExtractor()
        resolver = CitationResolver(self.temp_db)
        graph = CitationGraph(self.temp_db)
        
        # Extract citations
        citations = extractor.extract_citations_from_text(new_paper_refs, "new_ml_paper_2024")
        assert len(citations) >= 3, "Should extract multiple citations"
        
        # Resolve citations
        matches = resolver.resolve_citations(citations)
        assert len(matches) >= 3, "Should resolve most citations to known papers"
        
        # Check specific matches
        resolved_papers = [match.paper_id for match in matches]
        expected_papers = ['attention2017', 'bert2018', 'resnet2016', 'dropout2014']
        
        # Should match at least 2 of the expected papers
        matched_expected = [paper for paper in expected_papers if paper in resolved_papers]
        assert len(matched_expected) >= 2, f"Should match known papers, got: {resolved_papers}"
        
        # Build citation graph
        graph.load_from_citation_matches(matches)
        
        # Analyze graph
        metrics = graph.calculate_metrics()
        assert metrics['nodes'] >= 4
        assert metrics['edges'] >= 3
        
        # Find most cited papers
        most_cited = graph.get_most_cited_papers(3)
        assert len(most_cited) > 0
    
    def test_longitudinal_citation_tracking(self):
        """Test tracking citations over time."""
        temporal_analyzer = TimeSeriesAnalyzer(self.temp_db)
        
        # Simulate citation growth over time for key papers
        papers_to_track = ['attention2017', 'bert2018', 'resnet2016']
        base_date = datetime.now() - timedelta(days=365)
        
        # Simulate realistic citation growth patterns
        citation_patterns = {
            'attention2017': [100, 150, 220, 320, 450, 600, 800, 1000, 1200, 1500, 1800, 2200],  # High impact
            'bert2018': [50, 80, 120, 180, 280, 420, 650, 950, 1300, 1700, 2100, 2600],      # Explosive growth
            'resnet2016': [200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420]       # Steady growth
        }
        
        # Record monthly snapshots
        for month in range(12):
            timestamp = base_date + timedelta(days=month * 30)
            for paper_id in papers_to_track:
                citation_count = citation_patterns[paper_id][month]
                temporal_analyzer.record_citation_snapshot(paper_id, citation_count, timestamp)
        
        # Analyze trends
        trending_papers = temporal_analyzer.analyze_trends(180)  # 6 months
        
        assert len(trending_papers) > 0, "Should identify trending papers"
        
        # BERT should show high growth rate
        bert_trend = next((p for p in trending_papers if p.paper_id == 'bert2018'), None)
        if bert_trend:
            assert bert_trend.growth_rate > 50, "BERT should show high growth rate"
            assert bert_trend.trend_type in ['rising', 'burst'], "BERT should be rising/bursting"
        
        # Test burst detection
        bert_bursts = temporal_analyzer.detect_citation_bursts('bert2018', 365)
        # Should detect some bursts given the explosive growth pattern
        
        # Test forecasting
        forecast = temporal_analyzer.get_citation_forecast('attention2017', 30)
        assert forecast['forecasted_total_citations'] > 2200, "Should forecast continued growth"

class TestRobustnessAndRecovery:
    """Test system robustness and error recovery."""
    
    def test_corrupted_input_handling(self):
        """Test handling of corrupted or malformed input."""
        extractor = CitationExtractor()
        
        # Test various corrupted inputs
        corrupted_inputs = [
            "",  # Empty string
            "Not a reference section at all",  # No references
            "References\n\n[1]",  # Incomplete citation
            "References\n\n[1] \n[2] \n[3]",  # Empty citations
            "Ref" + "x" * 10000,  # Extremely long text
            "References\n\n" + "\n".join([f"[{i}] " for i in range(1000)]),  # Too many citations
        ]
        
        for corrupted_input in corrupted_inputs:
            # Should not crash
            citations = extractor.extract_citations_from_text(corrupted_input, "test")
            assert isinstance(citations, list), "Should always return a list"
    
    def test_database_recovery(self):
        """Test recovery from database issues."""
        # Test with non-existent database
        resolver = CitationResolver("nonexistent.db")
        
        citation = ExtractedCitation(
            raw_text="Test citation",
            source_paper_id="test"
        )
        
        # Should handle gracefully
        match = resolver.resolve_single_citation(citation)
        assert match is None, "Should return None for database issues"
        
        # Test empty database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            empty_db = f.name
        
        try:
            resolver = CitationResolver(empty_db)
            match = resolver.resolve_single_citation(citation)
            assert match is None, "Should handle empty database"
        finally:
            os.unlink(empty_db)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # This test would monitor memory usage in a real scenario
        # For now, we'll test that large operations complete without issues
        
        extractor = CitationExtractor()
        
        # Generate large citation text
        large_text = "References\n\n"
        for i in range(500):
            large_text += f"[{i+1}] Author{i}. Paper {i}. Venue {i} {2020}.\n"
        
        # Should handle large inputs
        citations = extractor.extract_citations_from_text(large_text, "memory_test")
        assert len(citations) > 0, "Should process large inputs"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
