"""
Temporal Analysis Module

This module provides time-series analysis capabilities for citation data,
tracking citation trends, identifying trending papers, and analyzing temporal
patterns in research networks.

Key Features:
- Track citation counts over time
- Identify trending and declining papers
- Analyze temporal citation patterns
- Calculate citation velocity and acceleration
- Detect citation bursts and anomalies
- Generate time-series forecasts
- Support for rolling window analysis

Classes:
    CitationTimePoint: Data class for citation measurements at specific times
    TrendingPaper: Data class for papers with temporal trend information
    TimeSeriesAnalyzer: Main class for temporal citation analysis
"""

import logging
import sqlite3
import json
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import math
import statistics

try:
    from .graph import CitationGraph, PaperNode
except ImportError:
    from graph import CitationGraph, PaperNode

try:
    from .resolver import CitationMatch
except ImportError:
    from resolver import CitationMatch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CitationTimePoint:
    """
    Represents citation count at a specific point in time.
    
    Attributes:
        paper_id (str): Paper identifier
        timestamp (datetime): Time of measurement
        citation_count (int): Total citations at this time
        new_citations (int): New citations since last measurement
        cumulative_citations (int): Cumulative citations to date
        citation_rate (float): Citations per day since last measurement
        acceleration (float): Change in citation rate
        measurement_period_days (int): Days since last measurement
    """
    paper_id: str
    timestamp: datetime
    citation_count: int
    new_citations: int = 0
    cumulative_citations: int = 0
    citation_rate: float = 0.0
    acceleration: float = 0.0
    measurement_period_days: int = 0

@dataclass
class TrendingPaper:
    """
    Represents a paper with trending information.
    
    Attributes:
        paper_id (str): Paper identifier
        title (str): Paper title
        authors (str): Paper authors
        current_citations (int): Current citation count
        previous_citations (int): Citation count in previous period
        citation_growth (int): Absolute growth in citations
        growth_rate (float): Relative growth rate (percentage)
        trend_score (float): Composite trending score
        velocity (float): Current citation velocity
        acceleration (float): Citation acceleration
        trend_type (str): Type of trend (rising, declining, stable, burst)
        confidence (float): Confidence in trend detection
        time_window_days (int): Analysis time window in days
        last_updated (datetime): When trend was last calculated
    """
    paper_id: str
    title: str = ""
    authors: str = ""
    current_citations: int = 0
    previous_citations: int = 0
    citation_growth: int = 0
    growth_rate: float = 0.0
    trend_score: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    trend_type: str = "stable"
    confidence: float = 0.0
    time_window_days: int = 30
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class TimeSeriesAnalyzer:
    """
    Provides temporal analysis capabilities for citation data.
    
    This class tracks citation patterns over time, identifies trending papers,
    and provides various temporal analytics for research impact analysis.
    """
    
    def __init__(self, db_path: str = "papers.db", citation_graph: Optional[CitationGraph] = None):
        """
        Initialize the TimeSeriesAnalyzer.
        
        Args:
            db_path (str): Path to the papers database
            citation_graph (Optional[CitationGraph]): Pre-existing citation graph
        """
        self.db_path = db_path
        self.citation_graph = citation_graph
        self.connection = None
        
        # Time series data storage
        self.citation_history: Dict[str, List[CitationTimePoint]] = defaultdict(list)
        self.trending_papers: Dict[str, TrendingPaper] = {}
        
        # Analysis parameters
        self.default_time_window = 30  # days
        self.min_citations_for_trending = 5
        self.burst_detection_threshold = 3.0  # standard deviations
        
        # Initialize database connection
        self._connect_to_database()
        self._ensure_temporal_tables()
        
        logger.info("TimeSeriesAnalyzer initialized")
    
    def _connect_to_database(self):
        """Establish connection to the database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            self.connection = None
    
    def _ensure_temporal_tables(self):
        """Ensure temporal analysis tables exist in the database."""
        if not self.connection:
            return
        
        try:
            cursor = self.connection.cursor()
            
            # Citation history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS citation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    citation_count INTEGER NOT NULL,
                    new_citations INTEGER DEFAULT 0,
                    cumulative_citations INTEGER DEFAULT 0,
                    citation_rate REAL DEFAULT 0.0,
                    acceleration REAL DEFAULT 0.0,
                    measurement_period_days INTEGER DEFAULT 0,
                    FOREIGN KEY (paper_id) REFERENCES papers (id)
                )
            """)
            
            # Trending papers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trending_papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    current_citations INTEGER NOT NULL,
                    previous_citations INTEGER NOT NULL,
                    citation_growth INTEGER NOT NULL,
                    growth_rate REAL NOT NULL,
                    trend_score REAL NOT NULL,
                    velocity REAL NOT NULL,
                    acceleration REAL NOT NULL,
                    trend_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    time_window_days INTEGER NOT NULL,
                    last_updated TEXT NOT NULL,
                    FOREIGN KEY (paper_id) REFERENCES papers (id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_citation_history_paper_timestamp ON citation_history (paper_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trending_papers_trend_score ON trending_papers (trend_score DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trending_papers_growth_rate ON trending_papers (growth_rate DESC)")
            
            self.connection.commit()
            logger.info("Temporal analysis tables ensured")
            
        except sqlite3.Error as e:
            logger.error(f"Error creating temporal tables: {e}")
    
    def record_citation_snapshot(self, paper_id: str, citation_count: int, 
                                timestamp: Optional[datetime] = None) -> bool:
        """
        Record a citation count snapshot for a paper.
        
        Args:
            paper_id (str): Paper identifier
            citation_count (int): Current citation count
            timestamp (Optional[datetime]): Snapshot timestamp
            
        Returns:
            bool: True if recorded successfully
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # Get previous snapshot for this paper
            previous_snapshot = self._get_latest_snapshot(paper_id)
            
            # Calculate derived metrics
            new_citations = 0
            citation_rate = 0.0
            acceleration = 0.0
            measurement_period_days = 0
            
            if previous_snapshot:
                time_diff = timestamp - previous_snapshot.timestamp
                measurement_period_days = time_diff.days
                
                if measurement_period_days > 0:
                    new_citations = citation_count - previous_snapshot.citation_count
                    citation_rate = new_citations / measurement_period_days
                    
                    # Calculate acceleration (change in citation rate)
                    if len(self.citation_history[paper_id]) > 1:
                        prev_rate = previous_snapshot.citation_rate
                        acceleration = (citation_rate - prev_rate) / measurement_period_days
            
            # Create time point
            time_point = CitationTimePoint(
                paper_id=paper_id,
                timestamp=timestamp,
                citation_count=citation_count,
                new_citations=new_citations,
                cumulative_citations=citation_count,
                citation_rate=citation_rate,
                acceleration=acceleration,
                measurement_period_days=measurement_period_days
            )
            
            # Store in memory
            self.citation_history[paper_id].append(time_point)
            
            # Store in database
            if self.connection:
                cursor = self.connection.cursor()
                cursor.execute("""
                    INSERT INTO citation_history 
                    (paper_id, timestamp, citation_count, new_citations, cumulative_citations,
                     citation_rate, acceleration, measurement_period_days)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper_id, timestamp.isoformat(), citation_count, new_citations,
                    citation_count, citation_rate, acceleration, measurement_period_days
                ))
                self.connection.commit()
            
            logger.debug(f"Recorded citation snapshot for {paper_id}: {citation_count} citations")
            return True
            
        except Exception as e:
            logger.error(f"Error recording citation snapshot for {paper_id}: {e}")
            return False
    
    def _get_latest_snapshot(self, paper_id: str) -> Optional[CitationTimePoint]:
        """Get the latest citation snapshot for a paper."""
        if paper_id in self.citation_history and self.citation_history[paper_id]:
            return self.citation_history[paper_id][-1]
        
        # Try to load from database
        if self.connection:
            try:
                cursor = self.connection.cursor()
                cursor.execute("""
                    SELECT * FROM citation_history 
                    WHERE paper_id = ? 
                    ORDER BY timestamp DESC LIMIT 1
                """, (paper_id,))
                
                row = cursor.fetchone()
                if row:
                    return CitationTimePoint(
                        paper_id=row['paper_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        citation_count=row['citation_count'],
                        new_citations=row['new_citations'],
                        cumulative_citations=row['cumulative_citations'],
                        citation_rate=row['citation_rate'],
                        acceleration=row['acceleration'],
                        measurement_period_days=row['measurement_period_days']
                    )
            except sqlite3.Error as e:
                logger.error(f"Error getting latest snapshot for {paper_id}: {e}")
        
        return None
    
    def analyze_trends(self, time_window_days: int = 30) -> List[TrendingPaper]:
        """
        Analyze citation trends for all papers in the time window.
        
        Args:
            time_window_days (int): Analysis time window in days
            
        Returns:
            List[TrendingPaper]: List of trending papers sorted by trend score
        """
        logger.info(f"Analyzing citation trends over {time_window_days} days")
        
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        trending_papers = []
        
        # Get all papers with citation history
        papers_to_analyze = set()
        if self.citation_graph:
            papers_to_analyze.update(self.citation_graph.paper_nodes.keys())
        papers_to_analyze.update(self.citation_history.keys())
        
        for paper_id in papers_to_analyze:
            trend_info = self._analyze_paper_trend(paper_id, time_window_days, cutoff_date)
            if trend_info and trend_info.current_citations >= self.min_citations_for_trending:
                trending_papers.append(trend_info)
        
        # Sort by trend score
        trending_papers.sort(key=lambda x: x.trend_score, reverse=True)
        
        # Update in-memory storage
        self.trending_papers = {paper.paper_id: paper for paper in trending_papers}
        
        # Store in database
        self._store_trending_papers(trending_papers)
        
        logger.info(f"Analyzed trends for {len(trending_papers)} papers")
        return trending_papers
    
    def _analyze_paper_trend(self, paper_id: str, time_window_days: int, 
                           cutoff_date: datetime) -> Optional[TrendingPaper]:
        """Analyze trend for a single paper."""
        try:
            # Get paper metadata
            paper_title = ""
            paper_authors = ""
            if self.citation_graph and paper_id in self.citation_graph.paper_nodes:
                node = self.citation_graph.paper_nodes[paper_id]
                paper_title = node.title
                paper_authors = node.authors
            
            # Get citation history for the paper
            history = self._get_citation_history(paper_id, cutoff_date)
            if len(history) < 2:
                return None
            
            # Calculate current and previous citation counts
            current_citations = history[-1].citation_count
            
            # Find citation count at the beginning of the window
            previous_citations = history[0].citation_count
            for point in history:
                if point.timestamp >= cutoff_date:
                    break
                previous_citations = point.citation_count
            
            # Calculate growth metrics
            citation_growth = current_citations - previous_citations
            growth_rate = (citation_growth / max(previous_citations, 1)) * 100
            
            # Calculate velocity and acceleration
            velocity = self._calculate_velocity(history, time_window_days)
            acceleration = self._calculate_acceleration(history, time_window_days)
            
            # Calculate trend score (composite metric)
            trend_score = self._calculate_trend_score(
                citation_growth, growth_rate, velocity, acceleration, current_citations
            )
            
            # Determine trend type
            trend_type = self._classify_trend(growth_rate, velocity, acceleration)
            
            # Calculate confidence
            confidence = self._calculate_trend_confidence(history, time_window_days)
            
            return TrendingPaper(
                paper_id=paper_id,
                title=paper_title,
                authors=paper_authors,
                current_citations=current_citations,
                previous_citations=previous_citations,
                citation_growth=citation_growth,
                growth_rate=growth_rate,
                trend_score=trend_score,
                velocity=velocity,
                acceleration=acceleration,
                trend_type=trend_type,
                confidence=confidence,
                time_window_days=time_window_days
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend for paper {paper_id}: {e}")
            return None
    
    def _get_citation_history(self, paper_id: str, since_date: datetime) -> List[CitationTimePoint]:
        """Get citation history for a paper since a specific date."""
        # First check memory
        if paper_id in self.citation_history:
            history = [point for point in self.citation_history[paper_id] 
                      if point.timestamp >= since_date]
            if history:
                return sorted(history, key=lambda x: x.timestamp)
        
        # Load from database
        history = []
        if self.connection:
            try:
                cursor = self.connection.cursor()
                cursor.execute("""
                    SELECT * FROM citation_history 
                    WHERE paper_id = ? AND timestamp >= ?
                    ORDER BY timestamp
                """, (paper_id, since_date.isoformat()))
                
                for row in cursor.fetchall():
                    point = CitationTimePoint(
                        paper_id=row['paper_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        citation_count=row['citation_count'],
                        new_citations=row['new_citations'],
                        cumulative_citations=row['cumulative_citations'],
                        citation_rate=row['citation_rate'],
                        acceleration=row['acceleration'],
                        measurement_period_days=row['measurement_period_days']
                    )
                    history.append(point)
                    
            except sqlite3.Error as e:
                logger.error(f"Error loading citation history for {paper_id}: {e}")
        
        return history
    
    def _calculate_velocity(self, history: List[CitationTimePoint], window_days: int) -> float:
        """Calculate citation velocity (citations per day)."""
        if len(history) < 2:
            return 0.0
        
        recent_points = [p for p in history if p.citation_rate > 0]
        if not recent_points:
            # Fallback: calculate from first and last points
            first_point = history[0]
            last_point = history[-1]
            time_diff = (last_point.timestamp - first_point.timestamp).days
            if time_diff > 0:
                return (last_point.citation_count - first_point.citation_count) / time_diff
            return 0.0
        
        # Average citation rate from recent measurements
        return sum(p.citation_rate for p in recent_points) / len(recent_points)
    
    def _calculate_acceleration(self, history: List[CitationTimePoint], window_days: int) -> float:
        """Calculate citation acceleration (change in citation rate)."""
        if len(history) < 3:
            return 0.0
        
        # Get acceleration values from time points
        accelerations = [p.acceleration for p in history[-3:] if p.acceleration != 0]
        
        if accelerations:
            return sum(accelerations) / len(accelerations)
        
        # Fallback: calculate from velocity changes
        velocities = []
        for i in range(1, len(history)):
            if history[i].citation_rate > 0:
                velocities.append(history[i].citation_rate)
        
        if len(velocities) >= 2:
            # Linear regression slope of velocities
            n = len(velocities)
            x_sum = sum(range(n))
            y_sum = sum(velocities)
            xy_sum = sum(i * v for i, v in enumerate(velocities))
            x2_sum = sum(i * i for i in range(n))
            
            if n * x2_sum - x_sum * x_sum != 0:
                slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
                return slope
        
        return 0.0
    
    def _calculate_trend_score(self, growth: int, growth_rate: float, velocity: float, 
                             acceleration: float, current_citations: int) -> float:
        """Calculate composite trend score."""
        # Normalize components
        growth_score = min(growth / 10.0, 10.0)  # Cap at 10
        rate_score = min(growth_rate / 50.0, 5.0)  # Cap at 5
        velocity_score = min(velocity * 10.0, 5.0)  # Scale velocity
        accel_score = min(acceleration * 100.0, 3.0)  # Scale acceleration
        
        # Popularity boost for highly cited papers
        popularity_boost = min(math.log(current_citations + 1) / 5.0, 2.0)
        
        # Weighted combination
        trend_score = (
            growth_score * 0.3 +
            rate_score * 0.25 +
            velocity_score * 0.25 +
            accel_score * 0.2 +
            popularity_boost
        )
        
        return max(trend_score, 0.0)
    
    def _classify_trend(self, growth_rate: float, velocity: float, acceleration: float) -> str:
        """Classify the type of trend."""
        if acceleration > 0.5:
            return "burst"
        elif growth_rate > 20 and velocity > 0.1:
            return "rising"
        elif growth_rate < -10 or velocity < -0.05:
            return "declining"
        elif abs(growth_rate) < 5 and abs(velocity) < 0.02:
            return "stable"
        else:
            return "fluctuating"
    
    def _calculate_trend_confidence(self, history: List[CitationTimePoint], window_days: int) -> float:
        """Calculate confidence in trend analysis."""
        # Base confidence on data quality
        confidence = 0.5  # Base confidence
        
        # More data points increase confidence
        if len(history) >= 5:
            confidence += 0.2
        elif len(history) >= 3:
            confidence += 0.1
        
        # Regular measurements increase confidence
        if len(history) > 1:
            intervals = []
            for i in range(1, len(history)):
                interval = (history[i].timestamp - history[i-1].timestamp).days
                intervals.append(interval)
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                if avg_interval <= 7:  # Weekly or more frequent
                    confidence += 0.2
                elif avg_interval <= 30:  # Monthly
                    confidence += 0.1
        
        # Longer observation window increases confidence
        if window_days >= 90:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _store_trending_papers(self, trending_papers: List[TrendingPaper]):
        """Store trending papers in the database."""
        if not self.connection or not trending_papers:
            return
        
        try:
            cursor = self.connection.cursor()
            
            # Clear existing trending papers
            cursor.execute("DELETE FROM trending_papers")
            
            # Insert new trending papers
            for paper in trending_papers:
                cursor.execute("""
                    INSERT INTO trending_papers 
                    (paper_id, title, authors, current_citations, previous_citations,
                     citation_growth, growth_rate, trend_score, velocity, acceleration,
                     trend_type, confidence, time_window_days, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper.paper_id, paper.title, paper.authors, paper.current_citations,
                    paper.previous_citations, paper.citation_growth, paper.growth_rate,
                    paper.trend_score, paper.velocity, paper.acceleration, paper.trend_type,
                    paper.confidence, paper.time_window_days, paper.last_updated.isoformat()
                ))
            
            self.connection.commit()
            logger.info(f"Stored {len(trending_papers)} trending papers in database")
            
        except sqlite3.Error as e:
            logger.error(f"Error storing trending papers: {e}")
    
    def get_trending_papers(self, n: int = 10, trend_type: Optional[str] = None) -> List[TrendingPaper]:
        """
        Get top trending papers.
        
        Args:
            n (int): Number of papers to return
            trend_type (Optional[str]): Filter by trend type
            
        Returns:
            List[TrendingPaper]: Top trending papers
        """
        papers = list(self.trending_papers.values())
        
        if trend_type:
            papers = [p for p in papers if p.trend_type == trend_type]
        
        # Sort by trend score
        papers.sort(key=lambda x: x.trend_score, reverse=True)
        
        return papers[:n]
    
    def detect_citation_bursts(self, paper_id: str, lookback_days: int = 90) -> List[Dict[str, Any]]:
        """
        Detect citation bursts for a specific paper.
        
        Args:
            paper_id (str): Paper to analyze
            lookback_days (int): Days to look back for burst detection
            
        Returns:
            List[Dict[str, Any]]: List of detected citation bursts
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        history = self._get_citation_history(paper_id, cutoff_date)
        
        if len(history) < 5:
            return []
        
        # Calculate rolling statistics
        citation_rates = [p.citation_rate for p in history if p.citation_rate > 0]
        if len(citation_rates) < 3:
            return []
        
        mean_rate = statistics.mean(citation_rates)
        std_rate = statistics.stdev(citation_rates) if len(citation_rates) > 1 else 0
        
        if std_rate == 0:
            return []
        
        # Detect bursts (citations significantly above normal)
        bursts = []
        for point in history:
            if point.citation_rate > 0:
                z_score = (point.citation_rate - mean_rate) / std_rate
                if z_score > self.burst_detection_threshold:
                    bursts.append({
                        'timestamp': point.timestamp,
                        'citation_rate': point.citation_rate,
                        'z_score': z_score,
                        'new_citations': point.new_citations,
                        'burst_intensity': min(z_score / self.burst_detection_threshold, 5.0)
                    })
        
        return sorted(bursts, key=lambda x: x['timestamp'])
    
    def get_citation_forecast(self, paper_id: str, forecast_days: int = 30) -> Dict[str, Any]:
        """
        Generate a simple citation forecast for a paper.
        
        Args:
            paper_id (str): Paper to forecast
            forecast_days (int): Days to forecast ahead
            
        Returns:
            Dict[str, Any]: Forecast information
        """
        # Get recent history (90 days)
        cutoff_date = datetime.now() - timedelta(days=90)
        history = self._get_citation_history(paper_id, cutoff_date)
        
        if len(history) < 3:
            return {"error": "Insufficient data for forecasting"}
        
        # Simple linear trend forecast
        current_citations = history[-1].citation_count
        
        # Calculate average daily citation rate
        rates = [p.citation_rate for p in history if p.citation_rate > 0]
        if not rates:
            avg_rate = 0.0
        else:
            avg_rate = sum(rates) / len(rates)
        
        # Calculate trend (acceleration)
        acceleration = self._calculate_acceleration(history, 90)
        
        # Forecast
        forecasted_new_citations = (avg_rate + acceleration * forecast_days / 2) * forecast_days
        forecasted_total = current_citations + max(0, int(forecasted_new_citations))
        
        return {
            'paper_id': paper_id,
            'current_citations': current_citations,
            'forecast_days': forecast_days,
            'forecasted_total_citations': forecasted_total,
            'forecasted_new_citations': int(forecasted_new_citations),
            'average_daily_rate': avg_rate,
            'trend_acceleration': acceleration,
            'confidence': 'low' if len(history) < 10 else 'medium'
        }
    
    def get_temporal_statistics(self) -> Dict[str, Any]:
        """
        Get overall temporal analysis statistics.
        
        Returns:
            Dict[str, Any]: Temporal statistics
        """
        stats = {
            'papers_tracked': len(self.citation_history),
            'total_measurements': sum(len(history) for history in self.citation_history.values()),
            'trending_papers': len(self.trending_papers),
            'trend_types': {}
        }
        
        # Count trend types
        for paper in self.trending_papers.values():
            trend_type = paper.trend_type
            stats['trend_types'][trend_type] = stats['trend_types'].get(trend_type, 0) + 1
        
        # Calculate measurement frequency
        if stats['total_measurements'] > 0:
            avg_measurements_per_paper = stats['total_measurements'] / stats['papers_tracked']
            stats['average_measurements_per_paper'] = avg_measurements_per_paper
        
        return stats
    
    def export_temporal_data(self, filepath: str, format: str = 'json') -> bool:
        """
        Export temporal analysis data to a file.
        
        Args:
            filepath (str): Output file path
            format (str): Export format ('json', 'csv')
            
        Returns:
            bool: True if exported successfully
        """
        try:
            if format.lower() == 'json':
                export_data = {
                    'citation_history': {
                        paper_id: [asdict(point) for point in history]
                        for paper_id, history in self.citation_history.items()
                    },
                    'trending_papers': {
                        paper_id: asdict(paper) for paper_id, paper in self.trending_papers.items()
                    },
                    'statistics': self.get_temporal_statistics(),
                    'export_timestamp': datetime.now().isoformat()
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
                    
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Temporal data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting temporal data: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

def analyze_paper_trends(citation_matches: List[CitationMatch], 
                        time_window_days: int = 30) -> List[TrendingPaper]:
    """
    Utility function to analyze trends from citation matches.
    
    Args:
        citation_matches (List[CitationMatch]): Citation matches to analyze
        time_window_days (int): Analysis time window
        
    Returns:
        List[TrendingPaper]: Trending papers
    """
    analyzer = TimeSeriesAnalyzer()
    
    # Record current snapshots from matches
    citation_counts = defaultdict(int)
    for match in citation_matches:
        citation_counts[match.paper_id] += 1
    
    # Record snapshots
    for paper_id, count in citation_counts.items():
        analyzer.record_citation_snapshot(paper_id, count)
    
    # Analyze trends
    return analyzer.analyze_trends(time_window_days)
