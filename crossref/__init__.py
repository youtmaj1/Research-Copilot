"""
Cross-Reference Module for Research Papers

This module provides comprehensive cross-referencing capabilities for research papers,
including citation extraction, semantic similarity analysis, and knowledge graph construction.

Components:
- CitationExtractor: Extract and match citations from PDF papers
- SimilarityEngine: Compute semantic similarities using embeddings
- CrossRefGraph: Build and analyze knowledge graphs
- CrossRefPipeline: Orchestrate the complete cross-referencing workflow
- IntegratedResearchPipeline: Connect with other modules for complete research workflow
"""

from .citation_extractor import CitationExtractor, Citation, CitationMatch
from .similarity import SimilarityEngine, SimilarityResult
from .graph import CrossRefGraph, GraphNode, GraphEdge, GraphExporter
from .pipeline import CrossRefPipeline, CrossRefConfig, create_crossref_pipeline
from .integration import IntegratedResearchPipeline, create_integrated_pipeline

__version__ = "1.0.0"
__author__ = "Research Copilot Team"

__all__ = [
    # Citation extraction
    'CitationExtractor',
    'Citation', 
    'CitationMatch',
    
    # Similarity analysis
    'SimilarityEngine',
    'SimilarityResult',
    
    # Knowledge graphs
    'CrossRefGraph',
    'GraphNode',
    'GraphEdge', 
    'GraphExporter',
    
    # Pipeline
    'CrossRefPipeline',
    'CrossRefConfig',
    'create_crossref_pipeline',
    
    # Integration
    'IntegratedResearchPipeline',
    'create_integrated_pipeline'
]
