"""
Citation Tracker Module

This module provides comprehensive citation extraction, resolution, and graph building
capabilities for tracking citation relationships and their evolution over time.

Key Components:
- CitationExtractor: Extracts citations from paper reference sections
- CitationResolver: Matches extracted references to known papers
- CitationGraph: Builds and analyzes citation networks with temporal features
- TimeSeriesAnalyzer: Analyzes citation evolution over time
- GraphExporter: Exports graphs in various formats (JSON, CSV, Neo4j)

Usage:
    from citation_tracker import CitationExtractor, CitationResolver, CitationGraph
    
    # Extract citations from a paper
    extractor = CitationExtractor()
    citations = extractor.extract_citations_from_paper(paper_path)
    
    # Resolve citations to known papers
    resolver = CitationResolver(papers_db_path)
    resolved_citations = resolver.resolve_citations(citations)
    
    # Build citation graph
    graph = CitationGraph()
    graph.add_citations(resolved_citations)
    graph.export_to_json("citation_graph.json")

Version: 1.0.0
Author: Research Copilot Team
"""

from .extractor import CitationExtractor, ExtractedCitation
from .resolver import CitationResolver, CitationMatch
from .graph import CitationGraph, PaperNode, CitationEdge
from .temporal import TimeSeriesAnalyzer
from .exporter import GraphExporter

__version__ = "1.0.0"
__all__ = [
    "CitationExtractor",
    "ExtractedCitation", 
    "CitationResolver",
    "ResolvedCitation",
    "CitationGraph",
    "TimeSeriesAnalyzer",
    "GraphExporter",
    "ExportFormat"
]
