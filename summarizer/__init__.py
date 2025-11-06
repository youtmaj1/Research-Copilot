"""
Research Paper Summarizer

A comprehensive system for processing research papers with PDF extraction,
intelligent chunking, LLM-based summarization, and vector storage.

Main Components:
- PDFExtractor: Extract structured text from PDFs with OCR fallback
- ResearchPaperChunker: Intelligent text chunking with multiple strategies  
- ResearchPaperSummarizer: LLM-based summarization with local/cloud support
- FaissVectorIndex: Vector storage and similarity search
- SummarizationPipeline: Complete orchestration pipeline

Usage:
    from summarizer import create_pipeline, summarize_paper
    
    # Quick summarization
    result = summarize_paper("paper.pdf")
    
    # Full pipeline
    pipeline = create_pipeline()
    result = pipeline.process_paper("paper.pdf")
"""

from .pdf_extractor import (
    PDFExtractor,
    PaperStructure,
    Section,
    extract_pdf_structure
)

from .chunker import (
    ResearchPaperChunker,
    TextChunk,
    ChunkMetadata,
    ChunkingStrategy,
    chunk_paper_structure
)

from .summarizer import (
    ResearchPaperSummarizer,
    SummaryResult,
    SummaryConfig,
    SummaryType,
    LLMProvider,
    summarize_paper
)

from .faiss_index import (
    FaissVectorIndex,
    EmbeddingMetadata,
    SearchResult,
    create_vector_index
)

from .pipeline import (
    SummarizationPipeline,
    PipelineConfig,
    ProcessingResult,
    ProcessingStatus,
    create_pipeline
)

__version__ = "1.0.0"
__author__ = "Research Copilot"

__all__ = [
    # PDF Extraction
    "PDFExtractor",
    "PaperStructure", 
    "Section",
    "extract_pdf_structure",
    
    # Chunking
    "ResearchPaperChunker",
    "TextChunk",
    "ChunkMetadata", 
    "ChunkingStrategy",
    "chunk_paper_structure",
    
    # Summarization
    "ResearchPaperSummarizer",
    "SummaryResult",
    "SummaryConfig",
    "SummaryType",
    "LLMProvider", 
    "summarize_paper",
    
    # Vector Index
    "FaissVectorIndex",
    "EmbeddingMetadata",
    "SearchResult",
    "create_vector_index",
    
    # Pipeline
    "SummarizationPipeline",
    "PipelineConfig",
    "ProcessingResult",
    "ProcessingStatus",
    "create_pipeline"
]
