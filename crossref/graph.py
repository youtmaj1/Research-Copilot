"""
Knowledge Graph Builder

Constructs and manages knowledge graphs of paper relationships.
Supports NetworkX graphs and exports to JSON/Neo4j formats.
"""

import logging
import json
import csv
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import pickle

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node (paper) in the knowledge graph."""
    node_id: str
    title: str
    authors: List[str]
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = None
    citation_count: int = 0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class GraphEdge:
    """Represents an edge (relationship) in the knowledge graph."""
    source_id: str
    target_id: str
    relation_type: str  # 'cites', 'similar_to', 'same_author', 'same_topic'
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Dict = None
    created_date: str = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_date is None:
            self.created_date = datetime.now().isoformat()


class CrossRefGraph:
    """
    Knowledge graph for paper cross-references.
    
    Manages nodes (papers) and edges (relationships) with support for:
    - Citation relationships
    - Semantic similarity connections
    - Author co-occurrence
    - Topic clustering
    """
    
    def __init__(self, graph_id: Optional[str] = None):
        """
        Initialize cross-reference graph.
        
        Args:
            graph_id: Unique identifier for this graph
        """
        self.graph_id = graph_id or f"crossref_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize NetworkX graph if available
        if nx is not None:
            self.nx_graph = nx.DiGraph()
        else:
            self.nx_graph = None
            logger.warning("NetworkX not available. Some graph operations will be limited.")
        
        # Internal storage
        self.nodes = {}  # node_id -> GraphNode
        self.edges = {}  # (source, target, relation) -> GraphEdge
        
        logger.info(f"Initialized cross-reference graph: {self.graph_id}")
    
    def add_node(self, node: GraphNode):
        """Add a node (paper) to the graph."""
        self.nodes[node.node_id] = node
        
        if self.nx_graph is not None:
            # Prepare attributes, avoiding conflicts with metadata
            node_attrs = {
                'title': node.title,
                'authors': node.authors,
                'year': node.year,
                'doi': node.doi,
                'arxiv_id': node.arxiv_id,
                'citation_count': node.citation_count
            }
            
            # Add metadata without overriding core attributes
            for key, value in node.metadata.items():
                if key not in node_attrs:
                    node_attrs[key] = value
            
            # Add to NetworkX with attributes
            self.nx_graph.add_node(node.node_id, **node_attrs)
    
    def add_edge(self, edge: GraphEdge):
        """Add an edge (relationship) to the graph."""
        edge_key = (edge.source_id, edge.target_id, edge.relation_type)
        self.edges[edge_key] = edge
        
        if self.nx_graph is not None:
            # Add to NetworkX with attributes
            self.nx_graph.add_edge(
                edge.source_id,
                edge.target_id,
                relation=edge.relation_type,
                weight=edge.weight,
                confidence=edge.confidence,
                created_date=edge.created_date,
                **edge.metadata
            )
    
    def add_papers(self, papers: Dict[str, Dict]):
        """
        Add multiple papers as nodes.
        
        Args:
            papers: Dict of paper_id -> paper metadata
        """
        for paper_id, paper_data in papers.items():
            node = self._create_node_from_paper_data(paper_id, paper_data)
            self.add_node(node)
        
        logger.info(f"Added {len(papers)} nodes to graph")
    
    def _create_node_from_paper_data(self, paper_id: str, paper_data: Dict) -> GraphNode:
        """Create GraphNode from paper metadata."""
        # Extract year from published_date
        year = None
        if paper_data.get('published_date'):
            try:
                year = int(paper_data['published_date'][:4])
            except (ValueError, TypeError):
                pass
        
        # Extract keywords from various sources
        keywords = []
        if 'keywords' in paper_data:
            keywords = paper_data['keywords']
        elif 'tags' in paper_data:
            keywords = paper_data['tags']
        
        return GraphNode(
            node_id=paper_id,
            title=paper_data.get('title', ''),
            authors=paper_data.get('authors', []),
            year=year,
            doi=paper_data.get('doi'),
            arxiv_id=paper_data.get('arxiv_id'),
            abstract=paper_data.get('abstract'),
            keywords=keywords,
            metadata=paper_data.copy()
        )
    
    def add_citation_relationships(self, citation_matches: List):
        """
        Add citation relationships to the graph.
        
        Args:
            citation_matches: List of CitationMatch objects
        """
        added_edges = 0
        
        for match in citation_matches:
            # Ensure both nodes exist
            if (match.source_paper_id not in self.nodes or 
                match.cited_paper_id not in self.nodes):
                continue
            
            edge = GraphEdge(
                source_id=match.source_paper_id,
                target_id=match.cited_paper_id,
                relation_type='cites',
                weight=1.0,
                confidence=match.confidence,
                metadata={
                    'match_type': match.match_type,
                    'citation_text': match.citation.raw_text[:200],  # Truncate for storage
                    'doi': match.citation.doi,
                    'arxiv_id': match.citation.arxiv_id
                }
            )
            
            self.add_edge(edge)
            added_edges += 1
            
            # Update citation count for cited paper
            if match.cited_paper_id in self.nodes:
                self.nodes[match.cited_paper_id].citation_count += 1
        
        logger.info(f"Added {added_edges} citation relationships")
    
    def add_similarity_relationships(self, similarity_results: List):
        """
        Add semantic similarity relationships to the graph.
        
        Args:
            similarity_results: List of SimilarityResult objects
        """
        added_edges = 0
        
        for result in similarity_results:
            # Ensure both nodes exist
            if (result.source_paper_id not in self.nodes or 
                result.target_paper_id not in self.nodes):
                continue
            
            edge = GraphEdge(
                source_id=result.source_paper_id,
                target_id=result.target_paper_id,
                relation_type='similar_to',
                weight=result.similarity_score,
                confidence=result.similarity_score,
                metadata={
                    'match_type': result.match_type,
                    'similarity_score': result.similarity_score
                }
            )
            
            self.add_edge(edge)
            added_edges += 1
        
        logger.info(f"Added {added_edges} similarity relationships")
    
    def add_author_relationships(self):
        """Add co-authorship relationships between papers."""
        added_edges = 0
        
        # Group papers by authors
        author_papers = {}
        for paper_id, node in self.nodes.items():
            for author in node.authors:
                author_key = author.lower().strip()
                if author_key not in author_papers:
                    author_papers[author_key] = []
                author_papers[author_key].append(paper_id)
        
        # Add edges between papers with shared authors
        for author, paper_ids in author_papers.items():
            if len(paper_ids) > 1:
                for i, paper1 in enumerate(paper_ids):
                    for paper2 in paper_ids[i+1:]:
                        # Calculate shared author ratio
                        authors1 = set(a.lower().strip() for a in self.nodes[paper1].authors)
                        authors2 = set(a.lower().strip() for a in self.nodes[paper2].authors)
                        
                        shared_authors = authors1 & authors2
                        total_authors = authors1 | authors2
                        
                        if shared_authors:
                            author_overlap = len(shared_authors) / len(total_authors)
                            
                            edge = GraphEdge(
                                source_id=paper1,
                                target_id=paper2,
                                relation_type='same_author',
                                weight=author_overlap,
                                confidence=1.0,
                                metadata={
                                    'shared_authors': list(shared_authors),
                                    'author_overlap': author_overlap
                                }
                            )
                            
                            self.add_edge(edge)
                            added_edges += 1
        
        logger.info(f"Added {added_edges} author relationships")
    
    def detect_topic_communities(self, similarity_threshold: float = 0.7) -> Dict[int, List[str]]:
        """
        Detect topic communities using graph clustering.
        
        Args:
            similarity_threshold: Minimum similarity for community detection
            
        Returns:
            Dict mapping community_id to list of paper_ids
        """
        if self.nx_graph is None:
            logger.error("NetworkX not available for community detection")
            return {}
        
        # Create subgraph with only similarity edges above threshold
        similarity_graph = nx.Graph()  # Undirected for community detection
        
        for node_id in self.nodes:
            similarity_graph.add_node(node_id)
        
        for edge_key, edge in self.edges.items():
            if (edge.relation_type == 'similar_to' and 
                edge.weight >= similarity_threshold):
                similarity_graph.add_edge(
                    edge.source_id, 
                    edge.target_id, 
                    weight=edge.weight
                )
        
        # Use connected components as simple communities
        communities = {}
        for i, component in enumerate(nx.connected_components(similarity_graph)):
            if len(component) > 1:  # Only keep multi-paper communities
                communities[i] = list(component)
        
        logger.info(f"Detected {len(communities)} topic communities")
        return communities
    
    def detect_communities(self, similarity_threshold: float = 0.7) -> Dict[int, List[str]]:
        """Alias for detect_topic_communities for compatibility."""
        return self.detect_topic_communities(similarity_threshold)
    
    def get_node_statistics(self) -> Dict[str, Any]:
        """Get statistics about nodes in the graph."""
        if not self.nodes:
            return {}
        
        # Year distribution
        years = [node.year for node in self.nodes.values() if node.year]
        year_stats = {
            'min_year': min(years) if years else None,
            'max_year': max(years) if years else None,
            'year_count': len(years)
        }
        
        # Author statistics
        all_authors = []
        for node in self.nodes.values():
            all_authors.extend(node.authors)
        
        unique_authors = set(all_authors)
        
        # Citation statistics
        citation_counts = [node.citation_count for node in self.nodes.values()]
        
        return {
            'total_nodes': len(self.nodes),
            'total_unique_authors': len(unique_authors),
            'total_citations': sum(citation_counts),
            'max_citations': max(citation_counts) if citation_counts else 0,
            'avg_citations': sum(citation_counts) / len(citation_counts) if citation_counts else 0,
            **year_stats
        }
    
    def get_edge_statistics(self) -> Dict[str, Any]:
        """Get statistics about edges in the graph."""
        if not self.edges:
            return {}
        
        # Count by relation type
        relation_counts = {}
        for edge in self.edges.values():
            relation_type = edge.relation_type
            relation_counts[relation_type] = relation_counts.get(relation_type, 0) + 1
        
        # Weight statistics
        weights = [edge.weight for edge in self.edges.values()]
        confidences = [edge.confidence for edge in self.edges.values()]
        
        return {
            'total_edges': len(self.edges),
            'relation_types': relation_counts,
            'avg_weight': sum(weights) / len(weights) if weights else 0,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'max_weight': max(weights) if weights else 0,
            'min_weight': min(weights) if weights else 0
        }
    
    def compute_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute centrality metrics for all nodes in the graph."""
        return self.get_centrality_metrics()
    
    def get_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get centrality metrics for all nodes in the graph."""
        if not self.nx_graph or len(self.nx_graph.nodes) == 0:
            return {"degree": {}, "betweenness": {}, "closeness": {}, "pagerank": {}}
        
        try:
            # PageRank (considering direction and weights)
            pagerank = nx.pagerank(self.nx_graph, weight='weight')
            
            # In-degree centrality (how many papers cite this paper)
            in_degree = dict(self.nx_graph.in_degree())
            
            # Out-degree centrality (how many papers this paper cites)
            out_degree = dict(self.nx_graph.out_degree())
            
            # Betweenness centrality
            betweenness = nx.betweenness_centrality(self.nx_graph)
            
            # Calculate total degree (in + out)
            total_degree = {}
            for node in self.nx_graph.nodes():
                total_degree[node] = in_degree.get(node, 0) + out_degree.get(node, 0)
            
            # Closeness centrality
            try:
                closeness = nx.closeness_centrality(self.nx_graph)
            except:
                closeness = {}
            
            return {
                'degree': total_degree,
                'pagerank': pagerank,
                'in_degree': in_degree,
                'out_degree': out_degree,
                'betweenness': betweenness,
                'closeness': closeness
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate centrality metrics: {e}")
            return {}
    
    def find_influential_papers(self, top_k: int = 10) -> List[Tuple[str, Dict]]:
        """
        Find most influential papers based on centrality metrics.
        
        Args:
            top_k: Number of top papers to return
            
        Returns:
            List of (paper_id, metrics) tuples
        """
        centrality = self.get_centrality_metrics()
        
        if not centrality:
            return []
        
        # Combine metrics with weights
        combined_scores = {}
        for paper_id in self.nodes:
            score = (
                centrality.get('pagerank', {}).get(paper_id, 0) * 0.4 +
                centrality.get('in_degree', {}).get(paper_id, 0) * 0.3 +
                centrality.get('betweenness', {}).get(paper_id, 0) * 0.3
            )
            combined_scores[paper_id] = score
        
        # Sort by combined score
        influential_papers = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Add node information
        results = []
        for paper_id, score in influential_papers:
            node = self.nodes[paper_id]
            metrics = {
                'combined_score': score,
                'pagerank': centrality.get('pagerank', {}).get(paper_id, 0),
                'in_degree': centrality.get('in_degree', {}).get(paper_id, 0),
                'out_degree': centrality.get('out_degree', {}).get(paper_id, 0),
                'betweenness': centrality.get('betweenness', {}).get(paper_id, 0),
                'title': node.title,
                'authors': node.authors,
                'year': node.year,
                'citation_count': node.citation_count
            }
            results.append((paper_id, metrics))
        
        return results


class GraphExporter:
    """Handles exporting graphs to various formats."""
    
    @staticmethod
    def export_to_json(graph: CrossRefGraph, output_path: str):
        """Export graph to JSON format."""
        # Prepare nodes
        nodes_data = {}
        for node_id, node in graph.nodes.items():
            nodes_data[node_id] = asdict(node)
        
        # Prepare edges
        edges_data = []
        for edge_key, edge in graph.edges.items():
            edge_dict = asdict(edge)
            edges_data.append(edge_dict)
        
        # Create export data
        export_data = {
            'graph_id': graph.graph_id,
            'created_date': datetime.now().isoformat(),
            'statistics': {
                'nodes': graph.get_node_statistics(),
                'edges': graph.get_edge_statistics()
            },
            'nodes': nodes_data,
            'edges': edges_data
        }
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported graph to JSON: {output_path}")
    
    @staticmethod
    def export_to_neo4j_csv(graph: CrossRefGraph, output_dir: str):
        """Export graph to Neo4j-compatible CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export nodes
        nodes_file = output_path / "papers.csv"
        nodes_data = []
        
        for node_id, node in graph.nodes.items():
            node_data = {
                'paper_id:ID': node_id,
                'title': node.title,
                'authors': ';'.join(node.authors),
                'year:int': node.year,
                'doi': node.doi,
                'arxiv_id': node.arxiv_id,
                'abstract': node.abstract,
                'keywords': ';'.join(node.keywords),
                'citation_count:int': node.citation_count,
                ':LABEL': 'Paper'
            }
            nodes_data.append(node_data)
        
        # Write nodes CSV
        if pd is not None:
            df_nodes = pd.DataFrame(nodes_data)
            df_nodes.to_csv(nodes_file, index=False)
        else:
            # Fallback to manual CSV writing
            with open(nodes_file, 'w', newline='') as f:
                if nodes_data:
                    writer = csv.DictWriter(f, fieldnames=nodes_data[0].keys())
                    writer.writeheader()
                    writer.writerows(nodes_data)
        
        # Export edges
        edges_file = output_path / "relationships.csv"
        edges_data = []
        
        for edge_key, edge in graph.edges.items():
            edge_data = {
                ':START_ID': edge.source_id,
                ':END_ID': edge.target_id,
                ':TYPE': edge.relation_type.upper(),
                'weight:float': edge.weight,
                'confidence:float': edge.confidence,
                'created_date': edge.created_date
            }
            edges_data.append(edge_data)
        
        # Write edges CSV
        if pd is not None:
            df_edges = pd.DataFrame(edges_data)
            df_edges.to_csv(edges_file, index=False)
        else:
            # Fallback to manual CSV writing
            with open(edges_file, 'w', newline='') as f:
                if edges_data:
                    writer = csv.DictWriter(f, fieldnames=edges_data[0].keys())
                    writer.writeheader()
                    writer.writerows(edges_data)
        
        logger.info(f"Exported Neo4j CSV files to: {output_path}")
    
    @staticmethod
    def export_to_networkx_pickle(graph: CrossRefGraph, output_path: str):
        """Export NetworkX graph to pickle format."""
        if graph.nx_graph is None:
            logger.error("NetworkX graph not available")
            return
        
        with open(output_path, 'wb') as f:
            pickle.dump(graph.nx_graph, f)
        
        logger.info(f"Exported NetworkX graph to: {output_path}")
    
    @staticmethod
    def export_edge_list(graph: CrossRefGraph, output_path: str, relation_type: Optional[str] = None):
        """Export simple edge list format."""
        edges_data = []
        
        for edge_key, edge in graph.edges.items():
            if relation_type is None or edge.relation_type == relation_type:
                edges_data.append({
                    'source': edge.source_id,
                    'target': edge.target_id,
                    'relation': edge.relation_type,
                    'weight': edge.weight,
                    'confidence': edge.confidence
                })
        
        if pd is not None:
            df = pd.DataFrame(edges_data)
            df.to_csv(output_path, index=False)
        else:
            with open(output_path, 'w', newline='') as f:
                if edges_data:
                    writer = csv.DictWriter(f, fieldnames=edges_data[0].keys())
                    writer.writeheader()
                    writer.writerows(edges_data)
        
        logger.info(f"Exported edge list to: {output_path}")


def load_graph_from_json(json_path: str) -> CrossRefGraph:
    """Load graph from JSON export file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create graph
    graph = CrossRefGraph(graph_id=data.get('graph_id'))
    
    # Load nodes
    for node_id, node_data in data['nodes'].items():
        node = GraphNode(**node_data)
        graph.add_node(node)
    
    # Load edges
    for edge_data in data['edges']:
        edge = GraphEdge(**edge_data)
        graph.add_edge(edge)
    
    logger.info(f"Loaded graph from JSON: {json_path}")
    return graph


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python graph.py <command> [args...]")
        print("Commands:")
        print("  create <papers.json> <output.json> - Create graph from papers")
        print("  export <graph.json> <format> <output> - Export graph")
        print("  stats <graph.json> - Show graph statistics")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create" and len(sys.argv) >= 4:
        papers_file = sys.argv[2]
        output_file = sys.argv[3]
        
        # Load papers
        with open(papers_file, 'r') as f:
            papers = json.load(f)
        
        # Create graph
        graph = CrossRefGraph()
        graph.add_papers(papers)
        
        # Export
        GraphExporter.export_to_json(graph, output_file)
        print(f"Created graph with {len(graph.nodes)} nodes")
    
    elif command == "export" and len(sys.argv) >= 5:
        graph_file = sys.argv[2]
        export_format = sys.argv[3]
        output_path = sys.argv[4]
        
        # Load graph
        graph = load_graph_from_json(graph_file)
        
        # Export in requested format
        if export_format == "neo4j":
            GraphExporter.export_to_neo4j_csv(graph, output_path)
        elif export_format == "edgelist":
            GraphExporter.export_edge_list(graph, output_path)
        elif export_format == "networkx":
            GraphExporter.export_to_networkx_pickle(graph, output_path)
        else:
            print(f"Unknown export format: {export_format}")
    
    elif command == "stats" and len(sys.argv) >= 3:
        graph_file = sys.argv[2]
        
        # Load graph
        graph = load_graph_from_json(graph_file)
        
        # Show statistics
        node_stats = graph.get_node_statistics()
        edge_stats = graph.get_edge_statistics()
        
        print("Graph Statistics:")
        print("\nNodes:")
        for key, value in node_stats.items():
            print(f"  {key}: {value}")
        
        print("\nEdges:")
        for key, value in edge_stats.items():
            print(f"  {key}: {value}")
        
        # Show influential papers
        influential = graph.find_influential_papers(top_k=5)
        print(f"\nTop 5 Influential Papers:")
        for i, (paper_id, metrics) in enumerate(influential, 1):
            print(f"{i}. {metrics['title']}")
            print(f"   Score: {metrics['combined_score']:.3f}")
            print(f"   Citations: {metrics['citation_count']}")
    
    else:
        print("Invalid command or arguments")
        sys.exit(1)
