"""
Citation Graph Module

This module creates and manages citation knowledge graphs using NetworkX.
It builds directed graphs where nodes represent papers and edges represent citations,
providing various graph analysis capabilities.

Key Features:
- Build citation networks from resolved citations
- Add papers and citation relationships
- Calculate graph metrics (centrality, clustering, etc.)
- Identify influential papers and citation patterns
- Support for subgraph extraction and filtering
- Integration with database and visualization tools

Classes:
    PaperNode: Data class representing a paper node in the graph
    CitationEdge: Data class representing a citation edge
    CitationGraph: Main class for building and analyzing citation graphs
"""

import logging
import sqlite3
import json
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, Counter
import networkx as nx
from datetime import datetime, timedelta

try:
    from .resolver import CitationMatch, CitationResolver
except ImportError:
    from resolver import CitationMatch, CitationResolver
try:
    from .extractor import ExtractedCitation
except ImportError:
    from extractor import ExtractedCitation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PaperNode:
    """
    Represents a paper node in the citation graph.
    
    Attributes:
        paper_id (str): Unique identifier for the paper
        title (str): Paper title
        authors (str): Paper authors
        year (Optional[int]): Publication year
        doi (Optional[str]): DOI if available
        arxiv_id (Optional[str]): arXiv ID if available
        venue (Optional[str]): Publication venue
        citation_count (int): Number of times this paper is cited
        reference_count (int): Number of papers this paper cites
        h_index (float): H-index score if calculated
        pagerank (float): PageRank score in the graph
        betweenness_centrality (float): Betweenness centrality score
        closeness_centrality (float): Closeness centrality score
        clustering_coefficient (float): Clustering coefficient
        added_timestamp (datetime): When this node was added to the graph
    """
    paper_id: str
    title: str = ""
    authors: str = ""
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    venue: Optional[str] = None
    citation_count: int = 0
    reference_count: int = 0
    h_index: float = 0.0
    pagerank: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    added_timestamp: datetime = None
    
    def __post_init__(self):
        if self.added_timestamp is None:
            self.added_timestamp = datetime.now()

@dataclass
class CitationEdge:
    """
    Represents a citation edge in the graph.
    
    Attributes:
        citing_paper_id (str): ID of the paper making the citation
        cited_paper_id (str): ID of the paper being cited
        confidence (float): Confidence in the citation match
        match_type (str): Type of match used to resolve citation
        extraction_method (str): Method used to extract the citation
        added_timestamp (datetime): When this edge was added
        weight (float): Edge weight (default 1.0)
        context (Optional[str]): Context around the citation if available
    """
    citing_paper_id: str
    cited_paper_id: str
    confidence: float = 1.0
    match_type: str = ""
    extraction_method: str = ""
    added_timestamp: datetime = None
    weight: float = 1.0
    context: Optional[str] = None
    
    def __post_init__(self):
        if self.added_timestamp is None:
            self.added_timestamp = datetime.now()

class CitationGraph:
    """
    Manages citation knowledge graphs using NetworkX.
    
    This class provides functionality to build, analyze, and manipulate
    citation networks, including adding papers and citations, calculating
    metrics, and extracting insights about research impact and connections.
    """
    
    def __init__(self, db_path: str = "papers.db"):
        """
        Initialize the CitationGraph.
        
        Args:
            db_path (str): Path to the papers database
        """
        self.db_path = db_path
        self.graph = nx.DiGraph()  # Directed graph for citations
        self.paper_nodes: Dict[str, PaperNode] = {}
        self.citation_edges: Dict[Tuple[str, str], CitationEdge] = {}
        self.connection = None
        
        # Initialize database connection
        self._connect_to_database()
        
        logger.info("CitationGraph initialized")
    
    def _connect_to_database(self):
        """Establish connection to the papers database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            self.connection = None
    
    def add_paper(self, paper_id: str, title: str = "", authors: str = "", 
                  year: Optional[int] = None, doi: Optional[str] = None,
                  arxiv_id: Optional[str] = None, venue: Optional[str] = None) -> bool:
        """
        Add a paper node to the graph.
        
        Args:
            paper_id (str): Unique paper identifier
            title (str): Paper title
            authors (str): Paper authors
            year (Optional[int]): Publication year
            doi (Optional[str]): DOI
            arxiv_id (Optional[str]): arXiv ID
            venue (Optional[str]): Publication venue
            
        Returns:
            bool: True if paper was added successfully
        """
        try:
            # Create paper node
            paper_node = PaperNode(
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                arxiv_id=arxiv_id,
                venue=venue
            )
            
            # Add to NetworkX graph
            self.graph.add_node(paper_id, **asdict(paper_node))
            
            # Store in our tracking dict
            self.paper_nodes[paper_id] = paper_node
            
            logger.debug(f"Added paper node: {paper_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding paper {paper_id}: {e}")
            return False
    
    def add_citation(self, citing_paper_id: str, cited_paper_id: str,
                    confidence: float = 1.0, match_type: str = "",
                    extraction_method: str = "", context: Optional[str] = None) -> bool:
        """
        Add a citation edge to the graph.
        
        Args:
            citing_paper_id (str): ID of paper making the citation
            cited_paper_id (str): ID of paper being cited
            confidence (float): Confidence in the citation match
            match_type (str): Type of match used
            extraction_method (str): Extraction method used
            context (Optional[str]): Citation context
            
        Returns:
            bool: True if citation was added successfully
        """
        try:
            # Ensure both papers exist in graph
            if citing_paper_id not in self.graph:
                self.add_paper(citing_paper_id)
            if cited_paper_id not in self.graph:
                self.add_paper(cited_paper_id)
            
            # Create citation edge
            citation_edge = CitationEdge(
                citing_paper_id=citing_paper_id,
                cited_paper_id=cited_paper_id,
                confidence=confidence,
                match_type=match_type,
                extraction_method=extraction_method,
                context=context
            )
            
            # Add to NetworkX graph
            self.graph.add_edge(
                citing_paper_id, 
                cited_paper_id, 
                **asdict(citation_edge)
            )
            
            # Store in our tracking dict
            edge_key = (citing_paper_id, cited_paper_id)
            self.citation_edges[edge_key] = citation_edge
            
            # Update citation counts
            if citing_paper_id in self.paper_nodes:
                self.paper_nodes[citing_paper_id].reference_count += 1
            if cited_paper_id in self.paper_nodes:
                self.paper_nodes[cited_paper_id].citation_count += 1
            
            logger.debug(f"Added citation: {citing_paper_id} -> {cited_paper_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding citation {citing_paper_id} -> {cited_paper_id}: {e}")
            return False
    
    def load_from_citation_matches(self, matches: List[CitationMatch]) -> int:
        """
        Load graph from a list of citation matches.
        
        Args:
            matches (List[CitationMatch]): Citation matches to load
            
        Returns:
            int: Number of citations successfully added
        """
        logger.info(f"Loading {len(matches)} citation matches into graph")
        added_count = 0
        
        for match in matches:
            # Add citing paper
            citing_id = match.citation.source_paper_id
            if citing_id and not self.has_paper(citing_id):
                self.add_paper(citing_id)
            
            # Add cited paper with full metadata
            cited_id = match.paper_id
            if not self.has_paper(cited_id):
                self.add_paper(
                    paper_id=cited_id,
                    title=match.paper_title,
                    authors=match.paper_authors,
                    year=match.paper_year,
                    doi=match.paper_doi,
                    arxiv_id=match.paper_arxiv_id
                )
            
            # Add citation edge
            if citing_id and self.add_citation(
                citing_paper_id=citing_id,
                cited_paper_id=cited_id,
                confidence=match.confidence,
                match_type=match.match_type,
                extraction_method="resolver"
            ):
                added_count += 1
        
        logger.info(f"Successfully loaded {added_count} citations into graph")
        return added_count
    
    def load_from_database(self, limit: Optional[int] = None) -> int:
        """
        Load papers from the database into the graph.
        
        Args:
            limit (Optional[int]): Maximum number of papers to load
            
        Returns:
            int: Number of papers loaded
        """
        if not self.connection:
            logger.error("No database connection available")
            return 0
        
        try:
            cursor = self.connection.cursor()
            query = "SELECT id, title, authors, year, doi, arxiv_id, venue FROM papers"
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            loaded_count = 0
            for row in rows:
                if self.add_paper(
                    paper_id=row['id'],
                    title=row['title'] or '',
                    authors=row['authors'] or '',
                    year=row['year'],
                    doi=row['doi'],
                    arxiv_id=row['arxiv_id'],
                    venue=row.get('venue')
                ):
                    loaded_count += 1
            
            logger.info(f"Loaded {loaded_count} papers from database")
            return loaded_count
            
        except sqlite3.Error as e:
            logger.error(f"Database loading error: {e}")
            return 0
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate various graph metrics and update node attributes.
        
        Returns:
            Dict[str, Any]: Dictionary of calculated metrics
        """
        logger.info("Calculating graph metrics")
        
        if self.graph.number_of_nodes() == 0:
            return {"error": "Empty graph"}
        
        metrics = {}
        
        try:
            # Basic graph statistics
            metrics['nodes'] = self.graph.number_of_nodes()
            metrics['edges'] = self.graph.number_of_edges()
            metrics['density'] = nx.density(self.graph)
            
            # Calculate centrality measures
            if self.graph.number_of_nodes() > 1:
                pagerank = nx.pagerank(self.graph, alpha=0.85)
                betweenness = nx.betweenness_centrality(self.graph)
                closeness = nx.closeness_centrality(self.graph)
                
                # Update node attributes
                for node_id in self.graph.nodes():
                    if node_id in self.paper_nodes:
                        self.paper_nodes[node_id].pagerank = pagerank.get(node_id, 0.0)
                        self.paper_nodes[node_id].betweenness_centrality = betweenness.get(node_id, 0.0)
                        self.paper_nodes[node_id].closeness_centrality = closeness.get(node_id, 0.0)
                
                metrics['average_pagerank'] = sum(pagerank.values()) / len(pagerank)
                metrics['max_pagerank'] = max(pagerank.values())
                metrics['average_betweenness'] = sum(betweenness.values()) / len(betweenness)
                metrics['max_betweenness'] = max(betweenness.values())
            
            # Calculate clustering coefficients
            clustering = nx.clustering(self.graph.to_undirected())
            for node_id, coeff in clustering.items():
                if node_id in self.paper_nodes:
                    self.paper_nodes[node_id].clustering_coefficient = coeff
            
            metrics['average_clustering'] = sum(clustering.values()) / len(clustering) if clustering else 0
            
            # Connected components analysis
            if self.graph.number_of_nodes() > 1:
                weakly_connected = list(nx.weakly_connected_components(self.graph))
                strongly_connected = list(nx.strongly_connected_components(self.graph))
                
                metrics['weakly_connected_components'] = len(weakly_connected)
                metrics['strongly_connected_components'] = len(strongly_connected)
                metrics['largest_component_size'] = max(len(comp) for comp in weakly_connected)
            
            # Citation statistics
            citation_counts = [self.paper_nodes[node_id].citation_count 
                             for node_id in self.paper_nodes]
            reference_counts = [self.paper_nodes[node_id].reference_count 
                              for node_id in self.paper_nodes]
            
            if citation_counts:
                metrics['average_citations'] = sum(citation_counts) / len(citation_counts)
                metrics['max_citations'] = max(citation_counts)
                metrics['median_citations'] = sorted(citation_counts)[len(citation_counts) // 2]
            
            if reference_counts:
                metrics['average_references'] = sum(reference_counts) / len(reference_counts)
                metrics['max_references'] = max(reference_counts)
            
            logger.info("Graph metrics calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def get_most_cited_papers(self, n: int = 10) -> List[Tuple[str, PaperNode]]:
        """
        Get the most cited papers in the graph.
        
        Args:
            n (int): Number of top papers to return
            
        Returns:
            List[Tuple[str, PaperNode]]: List of (paper_id, node) tuples
        """
        sorted_papers = sorted(
            self.paper_nodes.items(),
            key=lambda x: x[1].citation_count,
            reverse=True
        )
        return sorted_papers[:n]
    
    def get_most_influential_papers(self, n: int = 10, metric: str = 'pagerank') -> List[Tuple[str, PaperNode]]:
        """
        Get the most influential papers based on a centrality metric.
        
        Args:
            n (int): Number of top papers to return
            metric (str): Metric to use ('pagerank', 'betweenness', 'closeness')
            
        Returns:
            List[Tuple[str, PaperNode]]: List of (paper_id, node) tuples
        """
        if metric == 'pagerank':
            key_func = lambda x: x[1].pagerank
        elif metric == 'betweenness':
            key_func = lambda x: x[1].betweenness_centrality
        elif metric == 'closeness':
            key_func = lambda x: x[1].closeness_centrality
        else:
            logger.warning(f"Unknown metric: {metric}, using pagerank")
            key_func = lambda x: x[1].pagerank
        
        sorted_papers = sorted(
            self.paper_nodes.items(),
            key=key_func,
            reverse=True
        )
        return sorted_papers[:n]
    
    def get_citation_network_for_paper(self, paper_id: str, depth: int = 2) -> nx.DiGraph:
        """
        Extract citation network around a specific paper.
        
        Args:
            paper_id (str): Central paper ID
            depth (int): Network depth (1 = direct citations, 2 = citations of citations, etc.)
            
        Returns:
            nx.DiGraph: Subgraph containing the citation network
        """
        if paper_id not in self.graph:
            logger.warning(f"Paper {paper_id} not found in graph")
            return nx.DiGraph()
        
        # Start with the central paper
        nodes_to_include = {paper_id}
        
        # Expand by depth levels
        current_level = {paper_id}
        for _ in range(depth):
            next_level = set()
            
            for node in current_level:
                # Add papers that cite this paper
                citing_papers = set(self.graph.predecessors(node))
                next_level.update(citing_papers)
                
                # Add papers cited by this paper
                cited_papers = set(self.graph.successors(node))
                next_level.update(cited_papers)
            
            nodes_to_include.update(next_level)
            current_level = next_level
        
        # Create subgraph
        subgraph = self.graph.subgraph(nodes_to_include).copy()
        logger.info(f"Extracted network for {paper_id}: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        
        return subgraph
    
    def find_citation_paths(self, source_paper_id: str, target_paper_id: str, 
                           max_length: int = 5) -> List[List[str]]:
        """
        Find citation paths between two papers.
        
        Args:
            source_paper_id (str): Starting paper ID
            target_paper_id (str): Target paper ID
            max_length (int): Maximum path length to search
            
        Returns:
            List[List[str]]: List of citation paths (each path is a list of paper IDs)
        """
        if source_paper_id not in self.graph or target_paper_id not in self.graph:
            return []
        
        try:
            # Find all simple paths (no cycles)
            paths = list(nx.all_simple_paths(
                self.graph, 
                source_paper_id, 
                target_paper_id, 
                cutoff=max_length
            ))
            
            logger.info(f"Found {len(paths)} citation paths from {source_paper_id} to {target_paper_id}")
            return paths
            
        except nx.NetworkXNoPath:
            logger.info(f"No citation path found from {source_paper_id} to {target_paper_id}")
            return []
        except Exception as e:
            logger.error(f"Error finding citation paths: {e}")
            return []
    
    def get_papers_by_year(self, year: int) -> List[Tuple[str, PaperNode]]:
        """
        Get all papers published in a specific year.
        
        Args:
            year (int): Publication year
            
        Returns:
            List[Tuple[str, PaperNode]]: Papers from that year
        """
        papers = [(paper_id, node) for paper_id, node in self.paper_nodes.items()
                 if node.year == year]
        return sorted(papers, key=lambda x: x[1].citation_count, reverse=True)
    
    def get_venue_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about publication venues.
        
        Returns:
            Dict[str, Dict[str, Any]]: Venue statistics
        """
        venue_stats = defaultdict(lambda: {
            'paper_count': 0,
            'total_citations': 0,
            'average_citations': 0.0,
            'years': set(),
            'top_papers': []
        })
        
        for paper_id, node in self.paper_nodes.items():
            if node.venue:
                venue = node.venue
                venue_stats[venue]['paper_count'] += 1
                venue_stats[venue]['total_citations'] += node.citation_count
                if node.year:
                    venue_stats[venue]['years'].add(node.year)
                venue_stats[venue]['top_papers'].append((paper_id, node))
        
        # Calculate averages and sort top papers
        for venue in venue_stats:
            stats = venue_stats[venue]
            if stats['paper_count'] > 0:
                stats['average_citations'] = stats['total_citations'] / stats['paper_count']
            stats['years'] = sorted(list(stats['years']))
            stats['top_papers'] = sorted(stats['top_papers'], 
                                       key=lambda x: x[1].citation_count, 
                                       reverse=True)[:5]
        
        return dict(venue_stats)
    
    def has_paper(self, paper_id: str) -> bool:
        """Check if paper exists in the graph."""
        return paper_id in self.graph
    
    def has_citation(self, citing_paper_id: str, cited_paper_id: str) -> bool:
        """Check if citation edge exists in the graph."""
        return self.graph.has_edge(citing_paper_id, cited_paper_id)
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the citation graph.
        
        Returns:
            Dict[str, Any]: Graph summary statistics
        """
        summary = {
            'total_papers': len(self.paper_nodes),
            'total_citations': len(self.citation_edges),
            'creation_date': min(node.added_timestamp for node in self.paper_nodes.values()) if self.paper_nodes else None,
            'last_updated': max(node.added_timestamp for node in self.paper_nodes.values()) if self.paper_nodes else None
        }
        
        if self.paper_nodes:
            years = [node.year for node in self.paper_nodes.values() if node.year]
            if years:
                summary['earliest_paper'] = min(years)
                summary['latest_paper'] = max(years)
                summary['year_span'] = max(years) - min(years)
        
        return summary
    
    def save_to_file(self, filepath: str, format: str = 'json') -> bool:
        """
        Save the graph to a file.
        
        Args:
            filepath (str): Output file path
            format (str): Output format ('json', 'gml', 'graphml')
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if format.lower() == 'json':
                # Save as JSON with our custom node/edge data
                graph_data = {
                    'nodes': {node_id: asdict(node) for node_id, node in self.paper_nodes.items()},
                    'edges': {f"{edge[0]}->{edge[1]}": asdict(edge_data) 
                             for edge, edge_data in self.citation_edges.items()},
                    'summary': self.get_graph_summary()
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2, default=str)
                    
            elif format.lower() == 'gml':
                nx.write_gml(self.graph, filepath)
            elif format.lower() == 'graphml':
                nx.write_graphml(self.graph, filepath)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Graph saved to {filepath} in {format} format")
            return True
            
        except Exception as e:
            logger.error(f"Error saving graph to {filepath}: {e}")
            return False
    
    def load_from_file(self, filepath: str, format: str = 'json') -> bool:
        """
        Load graph from a file.
        
        Args:
            filepath (str): Input file path
            format (str): Input format ('json', 'gml', 'graphml')
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            if format.lower() == 'json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                
                # Load nodes
                if 'nodes' in graph_data:
                    for node_id, node_data in graph_data['nodes'].items():
                        # Convert timestamp strings back to datetime objects
                        if 'added_timestamp' in node_data and isinstance(node_data['added_timestamp'], str):
                            node_data['added_timestamp'] = datetime.fromisoformat(node_data['added_timestamp'])
                        
                        paper_node = PaperNode(**node_data)
                        self.paper_nodes[node_id] = paper_node
                        self.graph.add_node(node_id, **asdict(paper_node))
                
                # Load edges
                if 'edges' in graph_data:
                    for edge_key, edge_data in graph_data['edges'].items():
                        citing_id, cited_id = edge_key.split('->')
                        
                        # Convert timestamp strings back to datetime objects
                        if 'added_timestamp' in edge_data and isinstance(edge_data['added_timestamp'], str):
                            edge_data['added_timestamp'] = datetime.fromisoformat(edge_data['added_timestamp'])
                        
                        citation_edge = CitationEdge(**edge_data)
                        edge_tuple = (citing_id, cited_id)
                        self.citation_edges[edge_tuple] = citation_edge
                        self.graph.add_edge(citing_id, cited_id, **asdict(citation_edge))
                        
            elif format.lower() == 'gml':
                self.graph = nx.read_gml(filepath, destringizer=int)
            elif format.lower() == 'graphml':
                self.graph = nx.read_graphml(filepath)
            else:
                logger.error(f"Unsupported format: {format}")
                return False
            
            logger.info(f"Graph loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading graph from {filepath}: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

def build_citation_graph_from_matches(matches: List[CitationMatch], 
                                     db_path: str = "papers.db") -> CitationGraph:
    """
    Utility function to build a citation graph from citation matches.
    
    Args:
        matches (List[CitationMatch]): Citation matches to build graph from
        db_path (str): Path to papers database
        
    Returns:
        CitationGraph: Built citation graph
    """
    graph = CitationGraph(db_path)
    graph.load_from_citation_matches(matches)
    graph.calculate_metrics()
    return graph
