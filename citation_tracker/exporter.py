"""
Graph Exporter Module

This module exports citation graphs to various formats including Neo4j, GraphML,
JSON, and other graph databases and visualization tools.

Key Features:
- Export to Neo4j graph database
- Export to GraphML format for visualization tools
- Export to JSON with custom schema
- Export to CSV for data analysis
- Export to Gephi-compatible formats
- Support for custom filtering and subgraph export
- Batch export capabilities for large graphs

Classes:
    Neo4jExporter: Exports graphs to Neo4j database
    GraphMLExporter: Exports graphs to GraphML format
    JSONExporter: Exports graphs to JSON format
    CSVExporter: Exports graphs to CSV format
    GraphExporter: Main class coordinating all export operations
"""

import logging
import json
import csv
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
from xml.dom import minidom

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j driver not available. Neo4j export will be disabled.")

import networkx as nx

try:
    from .graph import CitationGraph, PaperNode, CitationEdge
except ImportError:
    from graph import CitationGraph, PaperNode, CitationEdge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jExporter:
    """
    Exports citation graphs to Neo4j graph database.
    
    This class handles the conversion of citation graphs to Neo4j's property
    graph model with papers as nodes and citations as relationships.
    """
    
    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize Neo4j exporter.
        
        Args:
            uri (str): Neo4j database URI
            username (str): Database username
            password (str): Database password
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
        
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
    
    def export_graph(self, citation_graph: CitationGraph, clear_existing: bool = False) -> bool:
        """
        Export citation graph to Neo4j.
        
        Args:
            citation_graph (CitationGraph): Graph to export
            clear_existing (bool): Whether to clear existing data
            
        Returns:
            bool: True if export successful
        """
        if not self.driver:
            logger.error("No Neo4j connection available")
            return False
        
        try:
            with self.driver.session() as session:
                # Clear existing data if requested
                if clear_existing:
                    session.run("MATCH (n) DETACH DELETE n")
                    logger.info("Cleared existing Neo4j data")
                
                # Create constraints and indexes
                self._create_constraints_and_indexes(session)
                
                # Export nodes (papers)
                self._export_papers(session, citation_graph)
                
                # Export relationships (citations)
                self._export_citations(session, citation_graph)
                
                logger.info(f"Successfully exported graph to Neo4j: {len(citation_graph.paper_nodes)} papers, {len(citation_graph.citation_edges)} citations")
                return True
                
        except Exception as e:
            logger.error(f"Error exporting to Neo4j: {e}")
            return False
    
    def _create_constraints_and_indexes(self, session):
        """Create Neo4j constraints and indexes for performance."""
        try:
            # Create constraint on paper ID
            session.run("CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
            
            # Create indexes for common queries
            session.run("CREATE INDEX paper_title_index IF NOT EXISTS FOR (p:Paper) ON (p.title)")
            session.run("CREATE INDEX paper_year_index IF NOT EXISTS FOR (p:Paper) ON (p.year)")
            session.run("CREATE INDEX paper_doi_index IF NOT EXISTS FOR (p:Paper) ON (p.doi)")
            session.run("CREATE INDEX paper_arxiv_index IF NOT EXISTS FOR (p:Paper) ON (p.arxiv_id)")
            
        except Exception as e:
            logger.warning(f"Error creating Neo4j constraints/indexes: {e}")
    
    def _export_papers(self, session, citation_graph: CitationGraph):
        """Export paper nodes to Neo4j."""
        papers_data = []
        for paper_id, paper_node in citation_graph.paper_nodes.items():
            paper_data = {
                'id': paper_id,
                'title': paper_node.title or '',
                'authors': paper_node.authors or '',
                'year': paper_node.year,
                'doi': paper_node.doi,
                'arxiv_id': paper_node.arxiv_id,
                'venue': paper_node.venue,
                'citation_count': paper_node.citation_count,
                'reference_count': paper_node.reference_count,
                'pagerank': paper_node.pagerank,
                'betweenness_centrality': paper_node.betweenness_centrality,
                'closeness_centrality': paper_node.closeness_centrality,
                'clustering_coefficient': paper_node.clustering_coefficient,
                'added_timestamp': paper_node.added_timestamp.isoformat() if paper_node.added_timestamp else None
            }
            papers_data.append(paper_data)
        
        # Batch insert papers
        batch_size = 1000
        for i in range(0, len(papers_data), batch_size):
            batch = papers_data[i:i + batch_size]
            session.run("""
                UNWIND $papers AS paper
                MERGE (p:Paper {id: paper.id})
                SET p.title = paper.title,
                    p.authors = paper.authors,
                    p.year = paper.year,
                    p.doi = paper.doi,
                    p.arxiv_id = paper.arxiv_id,
                    p.venue = paper.venue,
                    p.citation_count = paper.citation_count,
                    p.reference_count = paper.reference_count,
                    p.pagerank = paper.pagerank,
                    p.betweenness_centrality = paper.betweenness_centrality,
                    p.closeness_centrality = paper.closeness_centrality,
                    p.clustering_coefficient = paper.clustering_coefficient,
                    p.added_timestamp = paper.added_timestamp
            """, papers=batch)
        
        logger.info(f"Exported {len(papers_data)} papers to Neo4j")
    
    def _export_citations(self, session, citation_graph: CitationGraph):
        """Export citation edges to Neo4j."""
        citations_data = []
        for (citing_id, cited_id), citation_edge in citation_graph.citation_edges.items():
            citation_data = {
                'citing_id': citing_id,
                'cited_id': cited_id,
                'confidence': citation_edge.confidence,
                'match_type': citation_edge.match_type,
                'extraction_method': citation_edge.extraction_method,
                'weight': citation_edge.weight,
                'context': citation_edge.context,
                'added_timestamp': citation_edge.added_timestamp.isoformat() if citation_edge.added_timestamp else None
            }
            citations_data.append(citation_data)
        
        # Batch insert citations
        batch_size = 1000
        for i in range(0, len(citations_data), batch_size):
            batch = citations_data[i:i + batch_size]
            session.run("""
                UNWIND $citations AS citation
                MATCH (citing:Paper {id: citation.citing_id})
                MATCH (cited:Paper {id: citation.cited_id})
                MERGE (citing)-[r:CITES]->(cited)
                SET r.confidence = citation.confidence,
                    r.match_type = citation.match_type,
                    r.extraction_method = citation.extraction_method,
                    r.weight = citation.weight,
                    r.context = citation.context,
                    r.added_timestamp = citation.added_timestamp
            """, citations=batch)
        
        logger.info(f"Exported {len(citations_data)} citations to Neo4j")
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

class GraphMLExporter:
    """
    Exports citation graphs to GraphML format for visualization tools.
    
    GraphML is a comprehensive and easy-to-use file format for graphs
    supported by many visualization tools like Gephi, Cytoscape, etc.
    """
    
    def export_graph(self, citation_graph: CitationGraph, filepath: str) -> bool:
        """
        Export citation graph to GraphML format.
        
        Args:
            citation_graph (CitationGraph): Graph to export
            filepath (str): Output file path
            
        Returns:
            bool: True if export successful
        """
        try:
            # Create GraphML XML structure
            graphml = ET.Element('graphml')
            graphml.set('xmlns', 'http://graphml.graphdrawing.org/xmlns')
            graphml.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
            graphml.set('xsi:schemaLocation', 
                       'http://graphml.graphdrawing.org/xmlns '
                       'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd')
            
            # Define node attributes
            node_attributes = [
                ('title', 'string', 'Paper Title'),
                ('authors', 'string', 'Authors'),
                ('year', 'int', 'Publication Year'),
                ('doi', 'string', 'DOI'),
                ('arxiv_id', 'string', 'arXiv ID'),
                ('venue', 'string', 'Venue'),
                ('citation_count', 'int', 'Citation Count'),
                ('reference_count', 'int', 'Reference Count'),
                ('pagerank', 'double', 'PageRank'),
                ('betweenness_centrality', 'double', 'Betweenness Centrality'),
                ('closeness_centrality', 'double', 'Closeness Centrality'),
                ('clustering_coefficient', 'double', 'Clustering Coefficient')
            ]
            
            # Define edge attributes
            edge_attributes = [
                ('confidence', 'double', 'Match Confidence'),
                ('match_type', 'string', 'Match Type'),
                ('extraction_method', 'string', 'Extraction Method'),
                ('weight', 'double', 'Edge Weight'),
                ('context', 'string', 'Citation Context')
            ]
            
            # Add attribute definitions
            for i, (attr_name, attr_type, description) in enumerate(node_attributes):
                key = ET.SubElement(graphml, 'key')
                key.set('id', f'n{i}')
                key.set('for', 'node')
                key.set('attr.name', attr_name)
                key.set('attr.type', attr_type)
                desc = ET.SubElement(key, 'desc')
                desc.text = description
            
            for i, (attr_name, attr_type, description) in enumerate(edge_attributes):
                key = ET.SubElement(graphml, 'key')
                key.set('id', f'e{i}')
                key.set('for', 'edge')
                key.set('attr.name', attr_name)
                key.set('attr.type', attr_type)
                desc = ET.SubElement(key, 'desc')
                desc.text = description
            
            # Create graph element
            graph = ET.SubElement(graphml, 'graph')
            graph.set('id', 'CitationGraph')
            graph.set('edgedefault', 'directed')
            
            # Add nodes
            for paper_id, paper_node in citation_graph.paper_nodes.items():
                node = ET.SubElement(graph, 'node')
                node.set('id', paper_id)
                
                node_data = [
                    paper_node.title or '',
                    paper_node.authors or '',
                    paper_node.year or 0,
                    paper_node.doi or '',
                    paper_node.arxiv_id or '',
                    paper_node.venue or '',
                    paper_node.citation_count,
                    paper_node.reference_count,
                    paper_node.pagerank,
                    paper_node.betweenness_centrality,
                    paper_node.closeness_centrality,
                    paper_node.clustering_coefficient
                ]
                
                for i, value in enumerate(node_data):
                    if value is not None and str(value).strip():
                        data = ET.SubElement(node, 'data')
                        data.set('key', f'n{i}')
                        data.text = str(value)
            
            # Add edges
            edge_id = 0
            for (citing_id, cited_id), citation_edge in citation_graph.citation_edges.items():
                edge = ET.SubElement(graph, 'edge')
                edge.set('id', f'e{edge_id}')
                edge.set('source', citing_id)
                edge.set('target', cited_id)
                
                edge_data = [
                    citation_edge.confidence,
                    citation_edge.match_type or '',
                    citation_edge.extraction_method or '',
                    citation_edge.weight,
                    citation_edge.context or ''
                ]
                
                for i, value in enumerate(edge_data):
                    if value is not None and str(value).strip():
                        data = ET.SubElement(edge, 'data')
                        data.set('key', f'e{i}')
                        data.text = str(value)
                
                edge_id += 1
            
            # Write to file with pretty formatting
            rough_string = ET.tostring(graphml, encoding='unicode')
            reparsed = minidom.parseString(rough_string)
            pretty_xml = reparsed.toprettyxml(indent='  ')
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
            
            logger.info(f"Graph exported to GraphML: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to GraphML: {e}")
            return False

class JSONExporter:
    """
    Exports citation graphs to JSON format with custom schema.
    
    This exporter creates a JSON representation that includes both
    the graph structure and metadata for easy programmatic access.
    """
    
    def export_graph(self, citation_graph: CitationGraph, filepath: str, 
                    include_metadata: bool = True, compact: bool = False) -> bool:
        """
        Export citation graph to JSON format.
        
        Args:
            citation_graph (CitationGraph): Graph to export
            filepath (str): Output file path
            include_metadata (bool): Whether to include metadata
            compact (bool): Whether to use compact JSON format
            
        Returns:
            bool: True if export successful
        """
        try:
            # Build JSON structure
            graph_data = {
                'format': 'citation_graph_json',
                'version': '1.0',
                'exported_at': datetime.now().isoformat(),
                'graph': {
                    'nodes': {},
                    'edges': []
                }
            }
            
            # Add metadata if requested
            if include_metadata:
                graph_data['metadata'] = {
                    'total_papers': len(citation_graph.paper_nodes),
                    'total_citations': len(citation_graph.citation_edges),
                    'summary': citation_graph.get_graph_summary()
                }
                
                # Add graph metrics if available
                try:
                    metrics = citation_graph.calculate_metrics()
                    graph_data['metadata']['metrics'] = metrics
                except Exception as e:
                    logger.warning(f"Could not calculate metrics for export: {e}")
            
            # Export nodes
            for paper_id, paper_node in citation_graph.paper_nodes.items():
                node_data = {
                    'id': paper_id,
                    'title': paper_node.title,
                    'authors': paper_node.authors,
                    'year': paper_node.year,
                    'doi': paper_node.doi,
                    'arxiv_id': paper_node.arxiv_id,
                    'venue': paper_node.venue,
                    'citation_count': paper_node.citation_count,
                    'reference_count': paper_node.reference_count,
                    'pagerank': paper_node.pagerank,
                    'betweenness_centrality': paper_node.betweenness_centrality,
                    'closeness_centrality': paper_node.closeness_centrality,
                    'clustering_coefficient': paper_node.clustering_coefficient,
                    'added_timestamp': paper_node.added_timestamp.isoformat() if paper_node.added_timestamp else None
                }
                graph_data['graph']['nodes'][paper_id] = node_data
            
            # Export edges
            for (citing_id, cited_id), citation_edge in citation_graph.citation_edges.items():
                edge_data = {
                    'source': citing_id,
                    'target': cited_id,
                    'confidence': citation_edge.confidence,
                    'match_type': citation_edge.match_type,
                    'extraction_method': citation_edge.extraction_method,
                    'weight': citation_edge.weight,
                    'context': citation_edge.context,
                    'added_timestamp': citation_edge.added_timestamp.isoformat() if citation_edge.added_timestamp else None
                }
                graph_data['graph']['edges'].append(edge_data)
            
            # Write to file
            indent = None if compact else 2
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=indent, default=str, ensure_ascii=False)
            
            logger.info(f"Graph exported to JSON: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False

class CSVExporter:
    """
    Exports citation graphs to CSV format for data analysis.
    
    This exporter creates separate CSV files for nodes and edges,
    suitable for analysis in spreadsheet applications or data science tools.
    """
    
    def export_graph(self, citation_graph: CitationGraph, output_dir: str) -> bool:
        """
        Export citation graph to CSV files.
        
        Args:
            citation_graph (CitationGraph): Graph to export
            output_dir (str): Output directory for CSV files
            
        Returns:
            bool: True if export successful
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export nodes to CSV
            nodes_file = output_path / 'papers.csv'
            with open(nodes_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'id', 'title', 'authors', 'year', 'doi', 'arxiv_id', 'venue',
                    'citation_count', 'reference_count', 'pagerank', 
                    'betweenness_centrality', 'closeness_centrality', 
                    'clustering_coefficient', 'added_timestamp'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for paper_id, paper_node in citation_graph.paper_nodes.items():
                    row = {
                        'id': paper_id,
                        'title': paper_node.title or '',
                        'authors': paper_node.authors or '',
                        'year': paper_node.year,
                        'doi': paper_node.doi or '',
                        'arxiv_id': paper_node.arxiv_id or '',
                        'venue': paper_node.venue or '',
                        'citation_count': paper_node.citation_count,
                        'reference_count': paper_node.reference_count,
                        'pagerank': paper_node.pagerank,
                        'betweenness_centrality': paper_node.betweenness_centrality,
                        'closeness_centrality': paper_node.closeness_centrality,
                        'clustering_coefficient': paper_node.clustering_coefficient,
                        'added_timestamp': paper_node.added_timestamp.isoformat() if paper_node.added_timestamp else ''
                    }
                    writer.writerow(row)
            
            # Export edges to CSV
            edges_file = output_path / 'citations.csv'
            with open(edges_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'citing_paper_id', 'cited_paper_id', 'confidence', 'match_type',
                    'extraction_method', 'weight', 'context', 'added_timestamp'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for (citing_id, cited_id), citation_edge in citation_graph.citation_edges.items():
                    row = {
                        'citing_paper_id': citing_id,
                        'cited_paper_id': cited_id,
                        'confidence': citation_edge.confidence,
                        'match_type': citation_edge.match_type or '',
                        'extraction_method': citation_edge.extraction_method or '',
                        'weight': citation_edge.weight,
                        'context': citation_edge.context or '',
                        'added_timestamp': citation_edge.added_timestamp.isoformat() if citation_edge.added_timestamp else ''
                    }
                    writer.writerow(row)
            
            logger.info(f"Graph exported to CSV files in: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

class GraphExporter:
    """
    Main class coordinating all graph export operations.
    
    This class provides a unified interface for exporting citation graphs
    to various formats and destinations.
    """
    
    def __init__(self):
        """Initialize the GraphExporter."""
        self.neo4j_exporter = None
        self.graphml_exporter = GraphMLExporter()
        self.json_exporter = JSONExporter()
        self.csv_exporter = CSVExporter()
        
        logger.info("GraphExporter initialized")
    
    def setup_neo4j(self, uri: str, username: str, password: str) -> bool:
        """
        Setup Neo4j exporter with connection details.
        
        Args:
            uri (str): Neo4j database URI
            username (str): Database username
            password (str): Database password
            
        Returns:
            bool: True if setup successful
        """
        try:
            self.neo4j_exporter = Neo4jExporter(uri, username, password)
            return True
        except Exception as e:
            logger.error(f"Failed to setup Neo4j exporter: {e}")
            return False
    
    def export_to_neo4j(self, citation_graph: CitationGraph, clear_existing: bool = False) -> bool:
        """
        Export citation graph to Neo4j.
        
        Args:
            citation_graph (CitationGraph): Graph to export
            clear_existing (bool): Whether to clear existing data
            
        Returns:
            bool: True if export successful
        """
        if not self.neo4j_exporter:
            logger.error("Neo4j exporter not configured. Call setup_neo4j() first.")
            return False
        
        return self.neo4j_exporter.export_graph(citation_graph, clear_existing)
    
    def export_to_graphml(self, citation_graph: CitationGraph, filepath: str) -> bool:
        """
        Export citation graph to GraphML format.
        
        Args:
            citation_graph (CitationGraph): Graph to export
            filepath (str): Output file path
            
        Returns:
            bool: True if export successful
        """
        return self.graphml_exporter.export_graph(citation_graph, filepath)
    
    def export_to_json(self, citation_graph: CitationGraph, filepath: str, 
                      include_metadata: bool = True, compact: bool = False) -> bool:
        """
        Export citation graph to JSON format.
        
        Args:
            citation_graph (CitationGraph): Graph to export
            filepath (str): Output file path
            include_metadata (bool): Whether to include metadata
            compact (bool): Whether to use compact format
            
        Returns:
            bool: True if export successful
        """
        return self.json_exporter.export_graph(citation_graph, filepath, include_metadata, compact)
    
    def export_to_csv(self, citation_graph: CitationGraph, output_dir: str) -> bool:
        """
        Export citation graph to CSV files.
        
        Args:
            citation_graph (CitationGraph): Graph to export
            output_dir (str): Output directory
            
        Returns:
            bool: True if export successful
        """
        return self.csv_exporter.export_graph(citation_graph, output_dir)
    
    def export_subgraph(self, citation_graph: CitationGraph, paper_ids: Set[str], 
                       format: str, output_path: str, **kwargs) -> bool:
        """
        Export a subgraph containing only specified papers.
        
        Args:
            citation_graph (CitationGraph): Source graph
            paper_ids (Set[str]): Paper IDs to include in subgraph
            format (str): Export format ('neo4j', 'graphml', 'json', 'csv')
            output_path (str): Output path
            **kwargs: Additional format-specific arguments
            
        Returns:
            bool: True if export successful
        """
        try:
            # Create subgraph
            subgraph_nx = citation_graph.graph.subgraph(paper_ids).copy()
            
            # Create new CitationGraph instance for subgraph
            subgraph = CitationGraph()
            
            # Copy relevant nodes
            for paper_id in paper_ids:
                if paper_id in citation_graph.paper_nodes:
                    node = citation_graph.paper_nodes[paper_id]
                    subgraph.add_paper(
                        paper_id=node.paper_id,
                        title=node.title,
                        authors=node.authors,
                        year=node.year,
                        doi=node.doi,
                        arxiv_id=node.arxiv_id,
                        venue=node.venue
                    )
            
            # Copy relevant edges
            for (citing_id, cited_id), edge in citation_graph.citation_edges.items():
                if citing_id in paper_ids and cited_id in paper_ids:
                    subgraph.add_citation(
                        citing_paper_id=citing_id,
                        cited_paper_id=cited_id,
                        confidence=edge.confidence,
                        match_type=edge.match_type,
                        extraction_method=edge.extraction_method,
                        context=edge.context
                    )
            
            # Export using appropriate format
            if format.lower() == 'neo4j':
                return self.export_to_neo4j(subgraph, **kwargs)
            elif format.lower() == 'graphml':
                return self.export_to_graphml(subgraph, output_path)
            elif format.lower() == 'json':
                return self.export_to_json(subgraph, output_path, **kwargs)
            elif format.lower() == 'csv':
                return self.export_to_csv(subgraph, output_path)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting subgraph: {e}")
            return False
    
    def export_multiple_formats(self, citation_graph: CitationGraph, 
                              output_dir: str, formats: List[str]) -> Dict[str, bool]:
        """
        Export citation graph to multiple formats.
        
        Args:
            citation_graph (CitationGraph): Graph to export
            output_dir (str): Output directory
            formats (List[str]): List of formats to export to
            
        Returns:
            Dict[str, bool]: Results for each format
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for format_name in formats:
            try:
                if format_name.lower() == 'graphml':
                    filepath = output_path / 'citation_graph.graphml'
                    results[format_name] = self.export_to_graphml(citation_graph, str(filepath))
                
                elif format_name.lower() == 'json':
                    filepath = output_path / 'citation_graph.json'
                    results[format_name] = self.export_to_json(citation_graph, str(filepath))
                
                elif format_name.lower() == 'csv':
                    csv_dir = output_path / 'csv'
                    results[format_name] = self.export_to_csv(citation_graph, str(csv_dir))
                
                elif format_name.lower() == 'neo4j':
                    if self.neo4j_exporter:
                        results[format_name] = self.export_to_neo4j(citation_graph)
                    else:
                        logger.warning("Neo4j exporter not configured")
                        results[format_name] = False
                
                else:
                    logger.warning(f"Unknown format: {format_name}")
                    results[format_name] = False
                    
            except Exception as e:
                logger.error(f"Error exporting to {format_name}: {e}")
                results[format_name] = False
        
        return results
    
    def close(self):
        """Close all connections."""
        if self.neo4j_exporter:
            self.neo4j_exporter.close()
        logger.info("GraphExporter connections closed")

def export_citation_graph(citation_graph: CitationGraph, output_path: str, 
                         format: str = 'json', **kwargs) -> bool:
    """
    Utility function to export a citation graph.
    
    Args:
        citation_graph (CitationGraph): Graph to export
        output_path (str): Output path
        format (str): Export format
        **kwargs: Additional format-specific arguments
        
    Returns:
        bool: True if export successful
    """
    exporter = GraphExporter()
    
    if format.lower() == 'graphml':
        return exporter.export_to_graphml(citation_graph, output_path)
    elif format.lower() == 'json':
        return exporter.export_to_json(citation_graph, output_path, **kwargs)
    elif format.lower() == 'csv':
        return exporter.export_to_csv(citation_graph, output_path)
    else:
        logger.error(f"Unsupported format: {format}")
        return False
