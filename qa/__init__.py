"""
QA Module - Question Answerer with RAG Pipeline

This module implements a Retrieval-Augmented Generation (RAG) pipeline
for querying research papers using FAISS embeddings and knowledge graphs.

Components:
- Retriever: FAISS + BM25 hybrid retrieval
- RAG Pipeline: LLM-based answer generation with citations
- Query Rewriter: Optional query expansion for better recall
- Formatter: Structured answer formatting with citations
"""

# Import from advanced Q&A pipeline
try:
    from .advanced_qa_pipeline import EnterpriseQAPipeline, HybridRetriever, AdvancedAnswerGenerator
    __all__ = ["EnterpriseQAPipeline", "HybridRetriever", "AdvancedAnswerGenerator"]
except ImportError:
    # Fallback to old modules if available
    try:
        from .retriever import HybridRetriever
        from .rag import RAGPipeline
        from .query_rewriter import QueryRewriter
        from .formatter import AnswerFormatter
        __all__ = ["HybridRetriever", "RAGPipeline", "QueryRewriter", "AnswerFormatter"]
    except ImportError:
        __all__ = []

__version__ = "2.0.0"
