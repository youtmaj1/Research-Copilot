"""
Tests for Temporal Analysis Module

This module tests the time-series analysis functionality including:
- Citation count tracking over time
- Trend detection and analysis
- Citation burst detection
- Citation velocity calculations
- Forecasting capabilities
- Temporal database operations
"""

import pytest
import tempfile
import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from temporal import TimeSeriesAnalyzer, TrendingPaper, CitationTimePoint
from graph import CitationGraph
from database_schema import create_citation_tables

class TestTimeSeriesAnalyzer:
    """Test the TimeSeriesAnalyzer class."""
    
    def setup_method(self):
        """Set up test environment with temporal database."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.setup_test_database()
        self.graph = CitationGraph(self.temp_db)
        self.analyzer = TimeSeriesAnalyzer(self.temp_db, self.graph)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def setup_test_database(self):
        """Create test database with papers and temporal schema."""
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Create papers table
        cursor.execute("""
            CREATE TABLE papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                venue TEXT
            )
        """)
        
        # Add test papers
        test_papers = [
            ('paper1', 'Breakthrough in AI', 'Smith, J.', 2020, 'Nature'),
            ('paper2', 'Machine Learning Survey', 'Jones, A.', 2021, 'Science'),
            ('paper3', 'Deep Learning Methods', 'Brown, B.', 2022, 'ICML'),
            ('paper4', 'Computer Vision Advances', 'Davis, C.', 2023, 'CVPR'),
        ]
        
        cursor.executemany("""
            INSERT INTO papers (id, title, authors, year, venue)
            VALUES (?, ?, ?, ?, ?)
        """, test_papers)
        
        conn.commit()
        conn.close()
        
        # Create temporal analysis schema
        create_citation_tables(self.temp_db)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.db_path == self.temp_db
        assert self.analyzer.graph is not None
        
        # Test without graph
        analyzer_no_graph = TimeSeriesAnalyzer(self.temp_db)
        assert analyzer_no_graph.graph is None
    
    def test_record_citation_snapshot(self):
        """Test recording citation snapshots."""
        paper_id = 'paper1'
        citation_count = 10
        timestamp = datetime.now()
        
        # Record snapshot
        success = self.analyzer.record_citation_snapshot(paper_id, citation_count, timestamp)
        assert success, "Should successfully record snapshot"
        
        # Verify in database
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT paper_id, citation_count, timestamp 
            FROM citation_snapshots 
            WHERE paper_id = ?
        """, (paper_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        assert result is not None
        assert result[0] == paper_id
        assert result[1] == citation_count
    
    def test_record_multiple_snapshots(self):
        """Test recording multiple snapshots for the same paper."""
        paper_id = 'paper1'
        base_time = datetime.now() - timedelta(days=30)
        
        # Record snapshots over time
        for i in range(5):
            timestamp = base_time + timedelta(days=i * 7)
            citation_count = 10 + i * 5  # Growing citations
            self.analyzer.record_citation_snapshot(paper_id, citation_count, timestamp)
        
        # Verify all snapshots recorded
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM citation_snapshots WHERE paper_id = ?
        """, (paper_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 5, "Should have recorded 5 snapshots"
    
    def test_get_citation_history(self):
        """Test retrieving citation history for a paper."""
        paper_id = 'paper2'
        base_time = datetime.now() - timedelta(days=60)
        
        # Create citation history
        snapshots = []
        for i in range(6):
            timestamp = base_time + timedelta(days=i * 10)
            citation_count = 5 + i * 3
            snapshots.append((paper_id, citation_count, timestamp.isoformat()))
            self.analyzer.record_citation_snapshot(paper_id, citation_count, timestamp)
        
        # Retrieve history
        history = self.analyzer.get_citation_history(paper_id, days=60)
        
        assert len(history) == 6
        assert all(isinstance(snapshot, CitationTimePoint) for snapshot in history)
        
        # Check chronological order
        timestamps = [snapshot.timestamp for snapshot in history]
        assert timestamps == sorted(timestamps)
        
        # Check citation counts
        citation_counts = [snapshot.citation_count for snapshot in history]
        assert citation_counts[0] == 5
        assert citation_counts[-1] == 20  # 5 + 5*3
    
    def test_calculate_citation_velocity(self):
        """Test citation velocity calculation."""
        paper_id = 'paper3'
        base_time = datetime.now() - timedelta(days=100)
        
        # Create citation growth pattern
        for i in range(10):
            timestamp = base_time + timedelta(days=i * 10)
            citation_count = 2 * i  # Linear growth: 0, 2, 4, 6, 8, ...
            self.analyzer.record_citation_snapshot(paper_id, citation_count, timestamp)
        
        # Calculate velocity
        velocity = self.analyzer.calculate_citation_velocity(paper_id, window_days=50)
        
        assert velocity is not None
        assert velocity > 0  # Should show positive velocity
        assert abs(velocity - 0.2) < 0.1  # Approximately 0.2 citations per day
    
    def test_analyze_trends(self):
        """Test trend analysis."""
        base_time = datetime.now() - timedelta(days=90)
        
        # Create different growth patterns for different papers
        papers_data = {
            'paper1': [10, 15, 25, 40, 60],  # Explosive growth
            'paper2': [5, 7, 9, 11, 13],    # Steady growth
            'paper3': [20, 18, 16, 14, 12], # Declining
            'paper4': [8, 8, 8, 8, 8],      # Stable
        }
        
        for paper_id, counts in papers_data.items():
            for i, count in enumerate(counts):
                timestamp = base_time + timedelta(days=i * 20)
                self.analyzer.record_citation_snapshot(paper_id, count, timestamp)
        
        # Analyze trends
        trending_papers = self.analyzer.analyze_trends(days=80)
        
        assert len(trending_papers) > 0
        assert all(isinstance(paper, TrendingPaper) for paper in trending_papers)
        
        # Find paper1 in results (should have highest growth)
        paper1_trend = next((p for p in trending_papers if p.paper_id == 'paper1'), None)
        if paper1_trend:
            assert paper1_trend.growth_rate > 0
            assert paper1_trend.trend_type in ['rising', 'burst']
    
    def test_detect_citation_bursts(self):
        """Test citation burst detection."""
        paper_id = 'paper1'
        base_time = datetime.now() - timedelta(days=120)
        
        # Create burst pattern: slow growth, then sudden spike, then plateau
        burst_pattern = [5, 6, 7, 8, 25, 30, 35, 36, 37, 38]
        
        for i, count in enumerate(burst_pattern):
            timestamp = base_time + timedelta(days=i * 12)
            self.analyzer.record_citation_snapshot(paper_id, count, timestamp)
        
        # Detect bursts
        bursts = self.analyzer.detect_citation_bursts(paper_id, days=120)
        
        assert len(bursts) > 0
        assert all(isinstance(burst, TrendingPaper) for burst in bursts)
        
        # Should detect the burst around day 48-84 (indices 4-7)
        main_burst = bursts[0]  # Assuming first burst is the main one
        assert main_burst.intensity > 1.5  # Should be significant burst
        assert main_burst.duration_days > 0
    
    def test_get_trending_keywords(self):
        """Test trending keywords extraction."""
        # This would require more complex setup with paper keywords
        # For now, test basic functionality
        
        trending_keywords = self.analyzer.get_trending_keywords(days=30)
        
        # Should return a list (might be empty with limited test data)
        assert isinstance(trending_keywords, list)
    
    def test_get_citation_forecast(self):
        """Test citation forecasting."""
        paper_id = 'paper2'
        base_time = datetime.now() - timedelta(days=60)
        
        # Create predictable growth pattern
        for i in range(8):
            timestamp = base_time + timedelta(days=i * 7)
            citation_count = 10 + i * 2  # Linear growth
            self.analyzer.record_citation_snapshot(paper_id, citation_count, timestamp)
        
        # Get forecast
        forecast = self.analyzer.get_citation_forecast(paper_id, forecast_days=30)
        
        assert 'forecasted_total_citations' in forecast
        assert 'confidence_interval' in forecast
        assert 'trend' in forecast
        
        # Should forecast continued growth
        current_citations = 10 + 7 * 2  # Last recorded value
        forecasted = forecast['forecasted_total_citations']
        assert forecasted > current_citations
    
    def test_compare_papers_temporally(self):
        """Test temporal comparison of papers."""
        base_time = datetime.now() - timedelta(days=60)
        
        # Create different patterns for two papers
        papers = ['paper1', 'paper2']
        patterns = {
            'paper1': [10, 15, 22, 30, 40],  # Faster growth
            'paper2': [12, 16, 19, 22, 25],  # Slower growth
        }
        
        for paper_id, counts in patterns.items():
            for i, count in enumerate(counts):
                timestamp = base_time + timedelta(days=i * 12)
                self.analyzer.record_citation_snapshot(paper_id, count, timestamp)
        
        # Compare papers
        comparison = self.analyzer.compare_papers_temporally(papers, days=50)
        
        assert len(comparison) == 2
        assert 'paper1' in comparison
        assert 'paper2' in comparison
        
        # Each comparison should have velocity and trend info
        for paper_id, stats in comparison.items():
            assert 'velocity' in stats
            assert 'total_citations' in stats
            assert 'growth_rate' in stats
    
    def test_get_citation_distribution(self):
        """Test citation distribution analysis."""
        base_time = datetime.now() - timedelta(days=30)
        
        # Add citation snapshots for multiple papers
        papers_citations = {
            'paper1': 50,
            'paper2': 25,
            'paper3': 75,
            'paper4': 10,
        }
        
        for paper_id, citations in papers_citations.items():
            self.analyzer.record_citation_snapshot(paper_id, citations, base_time)
        
        distribution = self.analyzer.get_citation_distribution(days=30)
        
        assert 'total_papers' in distribution
        assert 'citation_stats' in distribution
        assert distribution['total_papers'] >= 4
        
        stats = distribution['citation_stats']
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert stats['mean'] > 0

class TestTrendingPaper:
    """Test the TrendingPaper dataclass."""
    
    def test_trending_paper_creation(self):
        """Test TrendingPaper creation."""
        paper = TrendingPaper(
            paper_id='test_paper',
            title='Test Paper',
            authors='Test Author',
            current_citations=100,
            growth_rate=25.5,
            trend_type='rising',
            velocity=2.5
        )
        
        assert paper.paper_id == 'test_paper'
        assert paper.title == 'Test Paper'
        assert paper.current_citations == 100
        assert paper.growth_rate == 25.5
        assert paper.trend_type == 'rising'
        assert paper.velocity == 2.5

class TestCitationSnapshot:
    """Test the CitationSnapshot dataclass."""
    
    def test_citation_snapshot_creation(self):
        """Test CitationSnapshot creation."""
        timestamp = datetime.now()
        snapshot = CitationTimePoint('test_paper', 50, timestamp)
        
        assert snapshot.paper_id == 'test_paper'
        assert snapshot.citation_count == 50
        assert snapshot.timestamp == timestamp

class TestCitationBurst:
    """Test the CitationBurst dataclass."""
    
    def test_citation_burst_creation(self):
        """Test CitationBurst creation."""
        start_time = datetime.now() - timedelta(days=30)
        end_time = datetime.now() - timedelta(days=20)
        
        burst = TrendingPaper(
            paper_id='test_paper',
            current_citations=45,
            growth_rate=3.5,
            trend_type='burst',
            velocity=4.5,
            acceleration=0.5,
            analysis_date=start_time
        )
        
        assert burst.start_date == start_time
        assert burst.end_date == end_time
        assert burst.intensity == 3.5
        assert burst.duration_days == 10
        assert burst.citations_during_burst == 45

class TestTemporalPerformance:
    """Test temporal analysis performance with larger datasets."""
    
    def setup_method(self):
        """Set up test environment with larger dataset."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        self.setup_performance_database()
        self.graph = CitationGraph(self.temp_db)
        self.analyzer = TimeSeriesAnalyzer(self.temp_db, self.graph)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def setup_performance_database(self):
        """Create database with many papers for performance testing."""
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                year INTEGER,
                venue TEXT
            )
        """)
        
        # Create 50 test papers
        papers = []
        for i in range(50):
            papers.append((
                f'perf_paper_{i:03d}',
                f'Performance Test Paper {i}',
                f'Author {i % 10}',
                2020 + (i % 4),
                f'Venue {i % 5}'
            ))
        
        cursor.executemany("""
            INSERT INTO papers (id, title, authors, year, venue)
            VALUES (?, ?, ?, ?, ?)
        """, papers)
        
        conn.commit()
        conn.close()
        
        create_citation_tables(self.temp_db)
    
    def test_large_scale_snapshot_recording(self):
        """Test recording many snapshots efficiently."""
        import time
        
        base_time = datetime.now() - timedelta(days=365)
        
        start_time = time.time()
        
        # Record daily snapshots for 50 papers over 1 year
        for paper_idx in range(20):  # Reduce to 20 papers for reasonable test time
            paper_id = f'perf_paper_{paper_idx:03d}'
            for day in range(0, 365, 7):  # Weekly snapshots
                timestamp = base_time + timedelta(days=day)
                citation_count = paper_idx * 2 + day // 10  # Simulated growth
                self.analyzer.record_citation_snapshot(paper_id, citation_count, timestamp)
        
        recording_time = time.time() - start_time
        
        assert recording_time < 10, f"Recording took too long: {recording_time}s"
        
        # Verify some snapshots were recorded
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM citation_snapshots")
        total_snapshots = cursor.fetchone()[0]
        conn.close()
        
        assert total_snapshots > 500, "Should have recorded many snapshots"
    
    def test_large_scale_trend_analysis(self):
        """Test trend analysis on many papers."""
        import time
        
        # First, record some data
        base_time = datetime.now() - timedelta(days=180)
        
        for paper_idx in range(30):
            paper_id = f'perf_paper_{paper_idx:03d}'
            for week in range(25):  # 25 weeks of data
                timestamp = base_time + timedelta(days=week * 7)
                # Different growth patterns
                if paper_idx % 3 == 0:
                    citation_count = 10 + week * 2  # Linear growth
                elif paper_idx % 3 == 1:
                    citation_count = 10 + week ** 1.2  # Accelerating growth
                else:
                    citation_count = 10 + week // 2  # Slow growth
                
                self.analyzer.record_citation_snapshot(paper_id, int(citation_count), timestamp)
        
        # Analyze trends
        start_time = time.time()
        trending_papers = self.analyzer.analyze_trends(days=150)
        analysis_time = time.time() - start_time
        
        assert analysis_time < 5, f"Trend analysis took too long: {analysis_time}s"
        assert len(trending_papers) > 0, "Should identify some trending papers"
    
    def test_bulk_citation_history_retrieval(self):
        """Test retrieving citation history for multiple papers efficiently."""
        import time
        
        # Set up data for multiple papers
        paper_ids = [f'perf_paper_{i:03d}' for i in range(10)]
        base_time = datetime.now() - timedelta(days=90)
        
        for paper_id in paper_ids:
            for week in range(12):
                timestamp = base_time + timedelta(days=week * 7)
                citation_count = 5 + week * 3
                self.analyzer.record_citation_snapshot(paper_id, citation_count, timestamp)
        
        # Retrieve histories
        start_time = time.time()
        histories = {}
        for paper_id in paper_ids:
            histories[paper_id] = self.analyzer.get_citation_history(paper_id, days=90)
        
        retrieval_time = time.time() - start_time
        
        assert retrieval_time < 3, f"History retrieval took too long: {retrieval_time}s"
        assert len(histories) == 10
        assert all(len(history) > 0 for history in histories.values())

class TestTemporalErrorHandling:
    """Test error handling in temporal analysis."""
    
    def setup_method(self):
        """Set up minimal test environment."""
        self.temp_db = tempfile.mktemp(suffix='.db')
        create_citation_tables(self.temp_db)
        self.analyzer = TimeSeriesAnalyzer(self.temp_db)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_db):
            os.unlink(self.temp_db)
    
    def test_nonexistent_paper_handling(self):
        """Test handling of non-existent papers."""
        # Should handle gracefully
        history = self.analyzer.get_citation_history('nonexistent_paper', days=30)
        assert history == []
        
        velocity = self.analyzer.calculate_citation_velocity('nonexistent_paper')
        assert velocity == 0
        
        bursts = self.analyzer.detect_citation_bursts('nonexistent_paper')
        assert bursts == []
    
    def test_invalid_database_handling(self):
        """Test handling of database issues."""
        # Test with non-existent database
        bad_analyzer = TimeSeriesAnalyzer('nonexistent.db')
        
        success = bad_analyzer.record_citation_snapshot('test', 10)
        assert success == False
        
        history = bad_analyzer.get_citation_history('test')
        assert history == []
    
    def test_invalid_date_ranges(self):
        """Test handling invalid date ranges."""
        # Record a snapshot
        self.analyzer.record_citation_snapshot('test_paper', 10)
        
        # Test with invalid days parameter
        history = self.analyzer.get_citation_history('test_paper', days=-10)
        assert isinstance(history, list)  # Should handle gracefully
        
        trends = self.analyzer.analyze_trends(days=0)
        assert isinstance(trends, list)  # Should handle gracefully

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
