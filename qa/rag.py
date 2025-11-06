"""
RAG Pipeline for Question Answering

Implements a Retrieval-Augmented Generation pipeline that combines
document retrieval with LLM generation for answering questions about research papers.
"""

import os
import logging
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import re

# LangChain imports
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Local imports
from .retriever import HybridRetriever, RetrievedChunk
from .formatter import AnswerFormatter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    answer: str
    citations: List[str]
    retrieved_chunks: List[RetrievedChunk]
    query: str
    timestamp: str
    confidence: float
    processing_time: float

class OllamaLLM(LLM):
    """
    Custom LangChain LLM wrapper for Ollama.
    """
    
    model_name: str = "deepseek-coder-v2"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 2048
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call Ollama API."""
        try:
            import requests
            
            # Prepare request
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                }
            }
            
            if stop:
                data["stop"] = stop
            
            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            return f"Error generating response: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "ollama"

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for research paper Q&A.
    
    This pipeline combines document retrieval with LLM generation to answer
    questions about research papers, providing citations and context.
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_model_name: str = "phi4-mini:3.8b",
        ollama_base_url: str = "http://localhost:11434",
        max_chunks: int = 5,
        max_context_length: int = 8000,
        temperature: float = 0.2,
        papers_db_path: Optional[str] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Hybrid retriever for document search
            llm_model_name: Name of Ollama model to use
            ollama_base_url: Base URL for Ollama API
            max_chunks: Maximum number of chunks to retrieve
            max_context_length: Maximum context length for LLM
            temperature: LLM temperature for generation
            papers_db_path: Path to papers metadata database
        """
        self.retriever = retriever
        self.llm_model_name = llm_model_name
        self.ollama_base_url = ollama_base_url
        self.max_chunks = max_chunks
        self.max_context_length = max_context_length
        self.papers_db_path = papers_db_path
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model_name=llm_model_name,
            base_url=ollama_base_url,
            temperature=temperature
        )
        
        # Initialize formatter
        self.formatter = AnswerFormatter()
        
        # Load paper metadata if available
        self.papers_metadata = {}
        if papers_db_path and os.path.exists(papers_db_path):
            self._load_papers_metadata()
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        logger.info(f"RAG Pipeline initialized with model: {llm_model_name}")
    
    def _load_papers_metadata(self):
        """Load paper metadata from database."""
        try:
            conn = sqlite3.connect(self.papers_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, title, authors, doi, arxiv_id, published_date, abstract
                FROM papers_metadata
            """)
            
            for row in cursor.fetchall():
                paper_id, title, authors, doi, arxiv_id, pub_date, abstract = row
                self.papers_metadata[paper_id] = {
                    'title': title,
                    'authors': authors,
                    'doi': doi,
                    'arxiv_id': arxiv_id,
                    'published_date': pub_date,
                    'abstract': abstract
                }
            
            conn.close()
            logger.info(f"Loaded metadata for {len(self.papers_metadata)} papers")
            
        except Exception as e:
            logger.warning(f"Failed to load papers metadata: {e}")
            self.papers_metadata = {}
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create prompt template for RAG."""
        template = """You are a knowledgeable research assistant with expertise in analyzing academic papers. Your task is to answer questions based ONLY on the provided research context.

INSTRUCTIONS:
1. Answer the question using ONLY the information provided in the context below
2. If the context doesn't contain enough information to answer the question, say so explicitly
3. Always cite specific papers when referencing information
4. Use the format [Paper_ID] when citing (e.g., [arxiv:2301.12345] or [paper_1])
5. Be precise and factual - do not add information not present in the context
6. If asked about comparisons, methodologies, or results, quote specific details from the papers

CONTEXT:
{context}

QUESTION: {question}

ANSWER: Provide a comprehensive answer based on the context above, with proper citations."""

        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    def _prepare_context(self, chunks: List[RetrievedChunk]) -> Tuple[str, List[str]]:
        """
        Prepare context from retrieved chunks for LLM prompt.
        
        Returns:
            Tuple of (context_text, citation_list)
        """
        context_parts = []
        citations = []
        current_length = 0
        
        for i, chunk in enumerate(chunks):
            # Get paper metadata for citation
            paper_info = self.papers_metadata.get(chunk.paper_id, {})
            
            # Create citation
            if paper_info.get('arxiv_id'):
                citation = f"arxiv:{paper_info['arxiv_id']}"
            elif paper_info.get('doi'):
                citation = f"doi:{paper_info['doi']}"
            else:
                citation = f"paper_{chunk.paper_id}"
            
            citations.append(citation)
            
            # Format chunk with metadata
            chunk_text = f"""
[{citation}] - {paper_info.get('title', 'Unknown Title')}
Authors: {paper_info.get('authors', 'Unknown Authors')}
Section: {chunk.metadata.get('section', 'Unknown')}
Content: {chunk.content}
"""
            
            # Check if adding this chunk would exceed context length
            if current_length + len(chunk_text) > self.max_context_length:
                logger.warning(f"Context length limit reached, using {i} chunks")
                break
            
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        context = "\n".join(context_parts)
        return context, citations
    
    def _extract_citations_from_answer(self, answer: str) -> List[str]:
        """Extract citation references from the generated answer."""
        # Pattern to match [arxiv:xxx], [doi:xxx], [paper_xxx] format
        citation_pattern = r'\[([^\]]+)\]'
        citations = re.findall(citation_pattern, answer)
        
        # Clean and deduplicate citations
        unique_citations = list(set(citations))
        return unique_citations
    
    def _calculate_confidence(
        self,
        chunks: List[RetrievedChunk],
        answer: str,
        citations: List[str]
    ) -> float:
        """
        Calculate confidence score for the generated answer.
        
        Based on:
        - Average retrieval scores
        - Number of citations in answer
        - Length and detail of answer
        """
        if not chunks:
            return 0.0
        
        # Average retrieval score (0-1)
        avg_retrieval_score = sum(chunk.score for chunk in chunks) / len(chunks)
        
        # Citations coverage (0-1)
        citations_score = min(len(citations) / 3, 1.0)  # Ideal: 3+ citations
        
        # Answer completeness (0-1)
        answer_length_score = min(len(answer) / 500, 1.0)  # Ideal: 500+ chars
        
        # Combine scores
        confidence = (
            0.4 * avg_retrieval_score +
            0.3 * citations_score +
            0.3 * answer_length_score
        )
        
        return min(confidence, 1.0)
    
    def query(
        self,
        question: str,
        max_chunks: Optional[int] = None,
        use_query_rewriter: bool = False
    ) -> RAGResponse:
        """
        Process a question through the RAG pipeline.
        
        Args:
            question: User question
            max_chunks: Override default max chunks
            use_query_rewriter: Whether to use query expansion (if available)
            
        Returns:
            RAG response with answer and citations
        """
        start_time = datetime.now()
        logger.info(f"Processing query: {question}")
        
        max_chunks = max_chunks or self.max_chunks
        
        try:
            # Optional: Rewrite/expand query
            processed_query = question
            if use_query_rewriter and hasattr(self, 'query_rewriter'):
                processed_query = self.query_rewriter.rewrite(question)
                logger.info(f"Expanded query: {processed_query}")
            
            # Retrieve relevant chunks
            chunks = self.retriever.retrieve(processed_query, k=max_chunks)
            
            if not chunks:
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    citations=[],
                    retrieved_chunks=[],
                    query=question,
                    timestamp=datetime.now().isoformat(),
                    confidence=0.0,
                    processing_time=0.0
                )
            
            # Prepare context
            context, retrieved_citations = self._prepare_context(chunks)
            
            # Generate prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            logger.info(f"Generated prompt length: {len(prompt)} characters")
            
            # Generate answer using LLM
            raw_answer = self.llm(prompt)
            
            # Extract citations from answer
            answer_citations = self._extract_citations_from_answer(raw_answer)
            
            # Combine all citations
            all_citations = list(set(retrieved_citations + answer_citations))
            
            # Calculate confidence
            confidence = self._calculate_confidence(chunks, raw_answer, all_citations)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Format the final answer
            formatted_answer = self.formatter.format_answer(
                raw_answer, all_citations, chunks
            )
            
            response = RAGResponse(
                answer=formatted_answer,
                citations=all_citations,
                retrieved_chunks=chunks,
                query=question,
                timestamp=datetime.now().isoformat(),
                confidence=confidence,
                processing_time=processing_time
            )
            
            logger.info(f"Query processed in {processing_time:.2f}s with confidence {confidence:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return RAGResponse(
                answer=f"I encountered an error while processing your question: {str(e)}",
                citations=[],
                retrieved_chunks=[],
                query=question,
                timestamp=datetime.now().isoformat(),
                confidence=0.0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def batch_query(self, questions: List[str]) -> List[RAGResponse]:
        """Process multiple questions in batch."""
        responses = []
        for question in questions:
            response = self.query(question)
            responses.append(response)
        return responses
    
    def get_paper_summary(self, paper_id: str) -> Optional[RAGResponse]:
        """Get a summary of a specific paper."""
        chunks = self.retriever.get_paper_chunks(paper_id)
        if not chunks:
            return None
        
        question = f"Please provide a comprehensive summary of the paper {paper_id}, including its main contributions, methodology, and results."
        
        # Use all chunks from the paper
        context, citations = self._prepare_context(chunks)
        
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        answer = self.llm(prompt)
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
            query=question,
            timestamp=datetime.now().isoformat(),
            confidence=0.9,  # High confidence for paper summaries
            processing_time=0.0
        )
    
    def compare_papers(self, paper_ids: List[str], aspect: str = "methodology") -> Optional[RAGResponse]:
        """Compare multiple papers on a specific aspect."""
        all_chunks = []
        for paper_id in paper_ids:
            chunks = self.retriever.get_paper_chunks(paper_id)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return None
        
        question = f"Compare the {aspect} used in papers {', '.join(paper_ids)}. Highlight similarities and differences."
        
        # Limit chunks to most relevant ones
        all_chunks = all_chunks[:self.max_chunks * 2]  # Allow more chunks for comparison
        
        context, citations = self._prepare_context(all_chunks)
        
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        answer = self.llm(prompt)
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=all_chunks,
            query=question,
            timestamp=datetime.now().isoformat(),
            confidence=0.8,
            processing_time=0.0
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        retriever_stats = self.retriever.get_statistics()
        
        return {
            "llm_model": self.llm_model_name,
            "max_chunks": self.max_chunks,
            "max_context_length": self.max_context_length,
            "papers_metadata_loaded": len(self.papers_metadata),
            "retriever_stats": retriever_stats
        }

def create_rag_pipeline(
    faiss_index_path: str,
    bm25_index_path: str,
    chunk_metadata_path: str,
    papers_db_path: Optional[str] = None,
    llm_model: str = "deepseek-coder-v2"
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline with all components.
    
    Args:
        faiss_index_path: Path to FAISS index
        bm25_index_path: Path to BM25 index
        chunk_metadata_path: Path to chunk metadata database
        papers_db_path: Path to papers metadata database
        llm_model: Ollama model name
        
    Returns:
        Configured RAG pipeline
    """
    # Create retriever
    retriever = HybridRetriever(
        faiss_index_path=faiss_index_path,
        bm25_index_path=bm25_index_path,
        chunk_metadata_path=chunk_metadata_path
    )
    
    # Create RAG pipeline
    rag_pipeline = RAGPipeline(
        retriever=retriever,
        llm_model_name=llm_model,
        papers_db_path=papers_db_path
    )
    
    return rag_pipeline

if __name__ == "__main__":
    # Example usage
    pipeline = create_rag_pipeline(
        faiss_index_path="data/processed/faiss_index.bin",
        bm25_index_path="data/processed/bm25_index.pkl",
        chunk_metadata_path="data/processed/chunks.db",
        papers_db_path="data/processed/papers.db"
    )
    
    # Test query
    response = pipeline.query("What are the main advantages of transformer architectures?")
    
    print(f"Question: {response.query}")
    print(f"Answer: {response.answer}")
    print(f"Citations: {response.citations}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Processing time: {response.processing_time:.2f}s")
