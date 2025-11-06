"""
Advanced Q&A System - Enterprise RAG Pipeline
============================================

Enterprise-grade Question-Answering system with:
- Hybrid retrieval combining dense and sparse search
- Advanced query processing and expansion
- Multi-modal answer generation with citations
- Performance optimization and caching
- Comprehensive monitoring and metrics

Author: Research Copilot System
Version: 2.0 Enterprise
"""

import os
import sys
import time
import json
import logging
import sqlite3
import threading
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import hashlib
from collections import defaultdict, Counter, deque

# Import Ollama configuration
try:
    from config.ollama_config import OllamaConfigManager
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    print("Warning: Ollama configuration not available")
import re

# Optional dependencies with fallbacks
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryContext:
    """Context information for a query"""
    original_query: str
    expanded_query: str
    query_type: str  # 'factual', 'analytical', 'comparative', 'summary'
    intent: str  # 'find', 'compare', 'explain', 'summarize'
    entities: List[str]
    keywords: List[str]
    timestamp: datetime

@dataclass
class RetrievedChunk:
    """Represents a retrieved text chunk with metadata"""
    paper_id: str
    chunk_id: str
    text: str
    score: float
    retrieval_method: str  # 'dense', 'sparse', 'hybrid'
    metadata: Dict[str, Any]
    context_window: Optional[str] = None

@dataclass
class Citation:
    """Citation information for an answer"""
    paper_id: str
    paper_title: str
    authors: List[str]
    chunk_text: str
    relevance_score: float
    page_number: Optional[int] = None

@dataclass
class QAResponse:
    """Complete Q&A response with citations and metadata"""
    query: str
    answer: str
    confidence: float
    citations: List[Citation]
    retrieval_time: float
    generation_time: float
    total_time: float
    retrieved_chunks: int
    method_used: str
    metadata: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    items_processed: int
    throughput: float
    cache_hits: int
    cache_misses: int
    error_count: int
    memory_usage: float

class EnterpriseCache:
    """High-performance caching with TTL and analytics"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with hit/miss tracking"""
        with self.lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            # Check expiration
            if time.time() - self.creation_times[key] > self.ttl_seconds:
                self._remove_key(key)
                self.miss_count += 1
                return None
            
            self.access_times[key] = time.time()
            self.hit_count += 1
            return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache with LRU eviction"""
        with self.lock:
            # Evict if necessary
            if len(self.cache) >= self.max_size:
                lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                self._remove_key(lru_key)
            
            current_time = time.time()
            self.cache[key] = value
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
    
    def _remove_key(self, key: str):
        """Remove key from all caches"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl_seconds
        }

class AdvancedQueryProcessor:
    """Advanced query processing with expansion and classification"""
    
    def __init__(self):
        self.query_cache = EnterpriseCache(max_size=5000)
        self.metrics = []
        
        # Query expansion dictionary
        self.expansion_terms = {
            'machine learning': ['ML', 'artificial intelligence', 'AI', 'neural networks', 'deep learning'],
            'natural language processing': ['NLP', 'text mining', 'computational linguistics', 'language modeling'],
            'computer vision': ['CV', 'image processing', 'visual recognition', 'pattern recognition'],
            'deep learning': ['neural networks', 'CNN', 'RNN', 'transformer', 'attention mechanism']
        }
        
        # Query type patterns
        self.query_patterns = {
            'factual': [r'\bwhat is\b', r'\bwho is\b', r'\bwhen did\b', r'\bwhere is\b', r'\bhow many\b'],
            'analytical': [r'\bwhy\b', r'\bhow does\b', r'\bexplain\b', r'\banalyze\b'],
            'comparative': [r'\bcompare\b', r'\bdifference\b', r'\bversus\b', r'\bbetter\b', r'\bsimilar\b'],
            'summary': [r'\bsummarize\b', r'\boverview\b', r'\bmain points\b', r'\bkey findings\b']
        }
    
    def process_query(self, query: str) -> QueryContext:
        """Process query with expansion and classification"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"query_{hashlib.md5(query.encode()).hexdigest()}"
        cached_result = self.query_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Classify query type
            query_type = self._classify_query_type(query)
            intent = self._extract_intent(query)
            
            # Extract entities and keywords
            entities = self._extract_entities(query)
            keywords = self._extract_keywords(query)
            
            # Expand query
            expanded_query = self._expand_query(query, keywords)
            
            context = QueryContext(
                original_query=query,
                expanded_query=expanded_query,
                query_type=query_type,
                intent=intent,
                entities=entities,
                keywords=keywords,
                timestamp=datetime.now()
            )
            
            # Cache result
            self.query_cache.set(cache_key, context)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.append(PerformanceMetrics(
                operation="query_processing",
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                items_processed=1,
                throughput=1 / duration if duration > 0 else 0,
                cache_hits=0,
                cache_misses=0,
                error_count=0,
                memory_usage=sys.getsizeof(context)
            ))
            
            return context
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            # Return basic context
            return QueryContext(
                original_query=query,
                expanded_query=query,
                query_type='unknown',
                intent='find',
                entities=[],
                keywords=query.split(),
                timestamp=datetime.now()
            )
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query into types"""
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        return 'factual'  # Default
    
    def _extract_intent(self, query: str) -> str:
        """Extract user intent from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'compare'
        elif any(word in query_lower for word in ['explain', 'why', 'how']):
            return 'explain'
        elif any(word in query_lower for word in ['summarize', 'overview', 'main']):
            return 'summarize'
        else:
            return 'find'
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query"""
        # Simple entity extraction based on capitalization and common patterns
        entities = []
        words = query.split()
        
        for word in words:
            # Capitalized words (potential proper nouns)
            if word[0].isupper() and len(word) > 1:
                entities.append(word)
            
            # Known technical terms
            if word.lower() in ['bert', 'gpt', 'transformer', 'cnn', 'rnn', 'lstm']:
                entities.append(word.upper())
        
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove stop words and extract meaningful terms
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were', 
                     'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                     'could', 'should', 'may', 'might', 'must', 'can', 'of', 'in', 'to', 'for', 
                     'with', 'by', 'from', 'about', 'what', 'how', 'why', 'where', 'when'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _expand_query(self, query: str, keywords: List[str]) -> str:
        """Expand query with related terms"""
        expanded_terms = set()
        query_lower = query.lower()
        
        # Add original query
        expanded_terms.add(query)
        
        # Add expansion terms for matched concepts
        for concept, expansions in self.expansion_terms.items():
            if concept in query_lower:
                expanded_terms.update(expansions)
        
        # Add synonyms for keywords
        synonym_map = {
            'method': ['approach', 'technique', 'algorithm'],
            'result': ['finding', 'outcome', 'conclusion'],
            'model': ['framework', 'architecture', 'system'],
            'performance': ['accuracy', 'effectiveness', 'efficiency']
        }
        
        for keyword in keywords:
            if keyword in synonym_map:
                expanded_terms.update(synonym_map[keyword])
        
        return ' '.join(expanded_terms)

class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods"""
    
    def __init__(self, db_path: str = "papers.db"):
        self.db_path = db_path
        self.cache = EnterpriseCache(max_size=10000)
        self.metrics = []
        
        # Initialize dense retrieval (embeddings)
        self.dense_model = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.dense_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded dense retrieval model")
            except Exception as e:
                logger.warning(f"Failed to load dense model: {e}")
        
        # Initialize sparse retrieval (TF-IDF)
        self.sparse_model = None
        self.document_texts = {}
        self.document_embeddings = {}
        
        if HAS_SKLEARN:
            self.sparse_model = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            logger.info("Initialized sparse retrieval model")
        
        # Initialize FAISS index
        self.faiss_index = None
        self.paper_ids = []
        
        # Load and index documents
        self._load_and_index_documents()
    
    def _load_and_index_documents(self):
        """Load documents and create search indices"""
        start_time = time.time()
        
        try:
            # Load documents from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, abstract 
                    FROM papers 
                    WHERE title IS NOT NULL
                """)
                
                documents = []
                for row in cursor.fetchall():
                    paper_id, title, abstract = row
                    # Combine title and abstract
                    text = f"{title}. {abstract or ''}"
                    self.document_texts[paper_id] = text
                    documents.append(text)
                    self.paper_ids.append(paper_id)
            
            logger.info(f"Loaded {len(documents)} documents for indexing")
            
            # Create sparse index
            if self.sparse_model and documents:
                self.sparse_vectors = self.sparse_model.fit_transform(documents)
                logger.info("Created sparse TF-IDF index")
            
            # Create dense index
            if self.dense_model and documents:
                embeddings = self.dense_model.encode(documents, batch_size=32, show_progress_bar=True)
                
                # Store embeddings
                for paper_id, embedding in zip(self.paper_ids, embeddings):
                    self.document_embeddings[paper_id] = embedding
                
                # Create FAISS index
                if HAS_FAISS:
                    dimension = embeddings.shape[1]
                    self.faiss_index = faiss.IndexFlatIP(dimension)
                    
                    # Normalize embeddings for cosine similarity
                    normalized_embeddings = embeddings.astype('float32')
                    faiss.normalize_L2(normalized_embeddings)
                    
                    self.faiss_index.add(normalized_embeddings)
                    logger.info(f"Created FAISS index with {len(embeddings)} embeddings")
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.append(PerformanceMetrics(
                operation="document_indexing",
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                items_processed=len(documents),
                throughput=len(documents) / duration if duration > 0 else 0,
                cache_hits=0,
                cache_misses=0,
                error_count=0,
                memory_usage=0
            ))
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
    
    def retrieve(self, query_context: QueryContext, top_k: int = 10) -> List[RetrievedChunk]:
        """Retrieve relevant chunks using hybrid approach"""
        start_time = time.time()
        
        # Check cache
        cache_key = f"retrieve_{hashlib.md5(query_context.expanded_query.encode()).hexdigest()}_{top_k}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            all_chunks = []
            
            # Dense retrieval
            if self.faiss_index and self.dense_model:
                dense_chunks = self._dense_retrieve(query_context.expanded_query, top_k)
                all_chunks.extend(dense_chunks)
            
            # Sparse retrieval
            if self.sparse_model and self.sparse_vectors is not None:
                sparse_chunks = self._sparse_retrieve(query_context.expanded_query, top_k)
                all_chunks.extend(sparse_chunks)
            
            # Hybrid scoring and re-ranking
            final_chunks = self._hybrid_rerank(all_chunks, query_context, top_k)
            
            # Cache result
            self.cache.set(cache_key, final_chunks)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.append(PerformanceMetrics(
                operation="retrieval",
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                items_processed=len(final_chunks),
                throughput=len(final_chunks) / duration if duration > 0 else 0,
                cache_hits=0,
                cache_misses=0,
                error_count=0,
                memory_usage=sys.getsizeof(final_chunks)
            ))
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def retrieve_from_papers(self, query_context: QueryContext, paper_ids: List[str], top_k: int = 10) -> List[RetrievedChunk]:
        """Retrieve relevant chunks from specific papers"""
        start_time = time.time()
        
        try:
            # Filter paper indices for specified papers
            target_indices = []
            for i, paper_id in enumerate(self.paper_ids):
                if paper_id in paper_ids:
                    target_indices.append(i)
            
            if not target_indices:
                logger.warning(f"No specified papers found in index: {paper_ids}")
                return []
            
            all_chunks = []
            
            # Dense retrieval from specific papers
            if self.faiss_index and self.dense_model:
                query_embedding = self.dense_model.encode([query_context.expanded_query])
                query_embedding = query_embedding.astype('float32')
                faiss.normalize_L2(query_embedding)
                
                # Search full index but filter results
                scores, indices = self.faiss_index.search(query_embedding, len(self.paper_ids))
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx in target_indices and len(all_chunks) < top_k:
                        paper_id = self.paper_ids[idx]
                        text = self.document_texts.get(paper_id, '')
                        
                        all_chunks.append(RetrievedChunk(
                            paper_id=paper_id,
                            chunk_id=f"filtered_dense_{idx}",
                            text=text,
                            score=float(score),
                            retrieval_method="filtered_dense",
                            metadata={'index': int(idx)}
                        ))
            
            # If we don't have enough results, try sparse retrieval
            if len(all_chunks) < top_k and self.sparse_model and self.sparse_vectors is not None:
                query_vector = self.sparse_model.transform([query_context.expanded_query])
                similarities = cosine_similarity(query_vector, self.sparse_vectors).flatten()
                
                # Create (score, index) pairs for target papers only
                target_scores = [(similarities[i], i) for i in target_indices]
                target_scores.sort(reverse=True)
                
                for score, idx in target_scores:
                    if len(all_chunks) >= top_k:
                        break
                    if score > 0:  # Only include relevant results
                        paper_id = self.paper_ids[idx]
                        text = self.document_texts.get(paper_id, '')
                        
                        all_chunks.append(RetrievedChunk(
                            paper_id=paper_id,
                            chunk_id=f"filtered_sparse_{idx}",
                            text=text,
                            score=float(score),
                            retrieval_method="filtered_sparse",
                            metadata={'index': int(idx)}
                        ))
            
            # Sort by score and limit to top_k
            all_chunks.sort(key=lambda x: x.score, reverse=True)
            final_chunks = all_chunks[:top_k]
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.append(PerformanceMetrics(
                operation="filtered_retrieval",
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                items_processed=len(final_chunks),
                throughput=len(final_chunks) / duration if duration > 0 else 0,
                cache_hits=0,
                cache_misses=0,
                error_count=0,
                memory_usage=sys.getsizeof(final_chunks)
            ))
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Filtered retrieval failed: {e}")
            return []
    
    def _dense_retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Dense retrieval using embeddings"""
        try:
            # Encode query
            query_embedding = self.dense_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            chunks = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.paper_ids):
                    paper_id = self.paper_ids[idx]
                    text = self.document_texts.get(paper_id, '')
                    
                    chunks.append(RetrievedChunk(
                        paper_id=paper_id,
                        chunk_id=f"dense_{idx}",
                        text=text,
                        score=float(score),
                        retrieval_method="dense",
                        metadata={'index': int(idx)}
                    ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []
    
    def _sparse_retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Sparse retrieval using TF-IDF"""
        try:
            # Transform query
            query_vector = self.sparse_model.transform([query])
            
            # Compute similarities
            similarities = cosine_similarity(query_vector, self.sparse_vectors).flatten()
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k] if HAS_NUMPY else sorted(
                range(len(similarities)), key=lambda i: similarities[i], reverse=True
            )[:top_k]
            
            chunks = []
            for idx in top_indices:
                if idx < len(self.paper_ids):
                    paper_id = self.paper_ids[idx]
                    text = self.document_texts.get(paper_id, '')
                    score = similarities[idx]
                    
                    if score > 0:  # Only include relevant results
                        chunks.append(RetrievedChunk(
                            paper_id=paper_id,
                            chunk_id=f"sparse_{idx}",
                            text=text,
                            score=float(score),
                            retrieval_method="sparse",
                            metadata={'index': int(idx)}
                        ))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return []
    
    def _hybrid_rerank(self, chunks: List[RetrievedChunk], query_context: QueryContext, top_k: int) -> List[RetrievedChunk]:
        """Hybrid re-ranking combining multiple signals"""
        if not chunks:
            return []
        
        # Group by paper_id and combine scores
        paper_scores = defaultdict(list)
        paper_chunks = {}
        
        for chunk in chunks:
            paper_scores[chunk.paper_id].append(chunk.score)
            paper_chunks[chunk.paper_id] = chunk
        
        # Calculate hybrid scores
        hybrid_chunks = []
        for paper_id, scores in paper_scores.items():
            chunk = paper_chunks[paper_id]
            
            # Combine dense and sparse scores
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            
            # Apply query type weighting
            type_weight = {
                'factual': 1.0,
                'analytical': 1.2,
                'comparative': 1.1,
                'summary': 0.9
            }.get(query_context.query_type, 1.0)
            
            # Apply recency bonus (if we have dates)
            recency_weight = 1.0  # Could be implemented with publication dates
            
            # Final hybrid score
            hybrid_score = (0.6 * max_score + 0.4 * avg_score) * type_weight * recency_weight
            
            # Create new chunk with hybrid score
            hybrid_chunk = RetrievedChunk(
                paper_id=chunk.paper_id,
                chunk_id=f"hybrid_{paper_id}",
                text=chunk.text,
                score=hybrid_score,
                retrieval_method="hybrid",
                metadata={
                    'dense_score': max([c.score for c in chunks if c.paper_id == paper_id and c.retrieval_method == 'dense'], default=0),
                    'sparse_score': max([c.score for c in chunks if c.paper_id == paper_id and c.retrieval_method == 'sparse'], default=0),
                    'type_weight': type_weight,
                    'recency_weight': recency_weight
                }
            )
            hybrid_chunks.append(hybrid_chunk)
        
        # Sort by hybrid score and return top-k
        hybrid_chunks.sort(key=lambda x: x.score, reverse=True)
        return hybrid_chunks[:top_k]

class AdvancedAnswerGenerator:
    """Advanced answer generation with citation tracking"""
    
    def __init__(self):
        self.cache = EnterpriseCache(max_size=1000)
        self.metrics = []
        
        # Initialize Ollama for local LLM
        self.ollama = None
        if HAS_OLLAMA:
            try:
                self.ollama = OllamaConfigManager()
                logger.info("✅ Ollama DeepSeek-Coder-V2:16B initialized for answer generation")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Ollama: {e}")
        
        # Answer templates for different query types
        self.answer_templates = {
            'factual': "Based on the research papers, {answer}. This is supported by {citations}.",
            'analytical': "Analysis of the research shows that {answer}. The key evidence includes {citations}.",
            'comparative': "Comparing the approaches in the literature: {answer}. This comparison is based on {citations}.",
            'summary': "The main findings from the research papers are: {answer}. These insights are drawn from {citations}."
        }
    
    def generate_answer(self, query_context: QueryContext, retrieved_chunks: List[RetrievedChunk]) -> QAResponse:
        """Generate comprehensive answer with citations"""
        start_time = time.time()
        
        # Check cache
        cache_key = f"answer_{hashlib.md5(query_context.original_query.encode()).hexdigest()}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Generate answer based on query type
            if query_context.query_type == 'factual':
                answer, citations = self._generate_factual_answer(query_context, retrieved_chunks)
            elif query_context.query_type == 'analytical':
                answer, citations = self._generate_analytical_answer(query_context, retrieved_chunks)
            elif query_context.query_type == 'comparative':
                answer, citations = self._generate_comparative_answer(query_context, retrieved_chunks)
            elif query_context.query_type == 'summary':
                answer, citations = self._generate_summary_answer(query_context, retrieved_chunks)
            else:
                answer, citations = self._generate_factual_answer(query_context, retrieved_chunks)
            
            # Calculate confidence based on retrieval scores and coverage
            confidence = self._calculate_confidence(retrieved_chunks, citations)
            
            generation_time = time.time() - start_time
            
            # Create response
            response = QAResponse(
                query=query_context.original_query,
                answer=answer,
                confidence=confidence,
                citations=citations,
                retrieval_time=0,  # Set by caller
                generation_time=generation_time,
                total_time=generation_time,
                retrieved_chunks=len(retrieved_chunks),
                method_used=query_context.query_type,
                metadata={
                    'query_type': query_context.query_type,
                    'intent': query_context.intent,
                    'entities': query_context.entities,
                    'keywords': query_context.keywords
                }
            )
            
            # Cache response
            self.cache.set(cache_key, response)
            
            # Record metrics
            self.metrics.append(PerformanceMetrics(
                operation="answer_generation",
                start_time=start_time,
                end_time=time.time(),
                duration=generation_time,
                items_processed=1,
                throughput=1 / generation_time if generation_time > 0 else 0,
                cache_hits=0,
                cache_misses=0,
                error_count=0,
                memory_usage=sys.getsizeof(response)
            ))
            
            return response
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # Return fallback response
            return QAResponse(
                query=query_context.original_query,
                answer="I couldn't generate a comprehensive answer based on the available research papers.",
                confidence=0.0,
                citations=[],
                retrieval_time=0,
                generation_time=time.time() - start_time,
                total_time=time.time() - start_time,
                retrieved_chunks=len(retrieved_chunks),
                method_used="fallback",
                metadata={}
            )
    
    def _generate_factual_answer(self, query_context: QueryContext, chunks: List[RetrievedChunk]) -> Tuple[str, List[Citation]]:
        """Generate factual answer from retrieved chunks using Ollama"""
        if not chunks:
            return "No relevant information found in the research papers.", []
        
        citations = []
        context_texts = []
        
        # Prepare context from top chunks
        for chunk in chunks[:5]:  # Use top 5 chunks
            context_texts.append(f"Paper {chunk.paper_id}: {chunk.text[:500]}...")
            
            # Create citation
            citation = Citation(
                paper_id=chunk.paper_id,
                paper_title=self._get_paper_title(chunk.paper_id),
                authors=[],  # Could be loaded from database
                chunk_text=chunk.text[:200] + "...",
                relevance_score=chunk.score
            )
            citations.append(citation)
        
        # Generate answer using Ollama if available
        if self.ollama:
            context = "\n\n".join(context_texts)
            
            prompt = f"""Based on the following research paper excerpts, provide a factual answer to the question.

Question: {query_context.original_query}

Research Context:
{context}

Instructions:
- Provide a clear, factual answer based only on the information provided
- Be specific and accurate
- If information is limited, acknowledge that
- Keep the answer concise but informative
- Do not make up information not present in the context

Answer:"""

            result = self.ollama.generate_completion(prompt, max_tokens=500, temperature=0.3)
            
            if result["success"]:
                answer = result["response"].strip()
                return answer, citations
            else:
                logger.warning(f"Ollama generation failed: {result['error']}")
        
        # Fallback to template-based generation
        key_facts = []
        for chunk in chunks[:3]:
            sentences = chunk.text.split('.')[:2]  # First 2 sentences
            key_facts.extend([s.strip() for s in sentences if s.strip()])
        
        if key_facts:
            answer = '. '.join(key_facts[:5]) + '.'
        else:
            answer = f"The research papers discuss {', '.join(query_context.keywords)} but specific details vary across studies."
        
        return answer, citations
    
    def _generate_analytical_answer(self, query_context: QueryContext, chunks: List[RetrievedChunk]) -> Tuple[str, List[Citation]]:
        """Generate analytical answer exploring relationships and causation"""
        if not chunks:
            return "Insufficient information for detailed analysis.", []
        
        # Analyze patterns and relationships
        concepts = defaultdict(list)
        citations = []
        
        for chunk in chunks[:5]:
            chunk_text = chunk.text
            
            # Look for analytical keywords
            analytical_keywords = ['because', 'therefore', 'thus', 'consequently', 'as a result', 
                                 'due to', 'leads to', 'causes', 'results in', 'shows that']
            
            sentences = chunk_text.split('.')
            analytical_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in analytical_keywords):
                    analytical_sentences.append(sentence)
            
            if analytical_sentences:
                concepts['analysis'].extend(analytical_sentences[:2])
                
                citation = Citation(
                    paper_id=chunk.paper_id,
                    paper_title=self._get_paper_title(chunk.paper_id),
                    authors=[],
                    chunk_text='. '.join(analytical_sentences[:2]),
                    relevance_score=chunk.score
                )
                citations.append(citation)
        
        # Build analytical answer
        if concepts['analysis']:
            answer = f"Analysis of the research reveals several key relationships: {' '.join(concepts['analysis'][:3])}"
        else:
            answer = f"The research on {', '.join(query_context.keywords)} presents various analytical perspectives that warrant further investigation."
        
        return answer, citations
    
    def _generate_comparative_answer(self, query_context: QueryContext, chunks: List[RetrievedChunk]) -> Tuple[str, List[Citation]]:
        """Generate comparative answer highlighting differences and similarities"""
        if not chunks:
            return "No comparative information available.", []
        
        comparisons = []
        citations = []
        
        for chunk in chunks[:5]:
            chunk_text = chunk.text
            
            # Look for comparative language
            comparative_keywords = ['compared to', 'versus', 'while', 'whereas', 'however', 
                                  'in contrast', 'on the other hand', 'different from', 'similar to']
            
            sentences = chunk_text.split('.')
            comparative_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence.lower() for keyword in comparative_keywords):
                    comparative_sentences.append(sentence)
            
            if comparative_sentences:
                comparisons.extend(comparative_sentences[:2])
                
                citation = Citation(
                    paper_id=chunk.paper_id,
                    paper_title=self._get_paper_title(chunk.paper_id),
                    authors=[],
                    chunk_text='. '.join(comparative_sentences[:2]),
                    relevance_score=chunk.score
                )
                citations.append(citation)
        
        # Build comparative answer
        if comparisons:
            answer = f"Comparative analysis shows: {' '.join(comparisons[:3])}"
        else:
            answer = f"The research presents different approaches to {', '.join(query_context.keywords)} with varying methodologies and results."
        
        return answer, citations
    
    def _generate_summary_answer(self, query_context: QueryContext, chunks: List[RetrievedChunk]) -> Tuple[str, List[Citation]]:
        """Generate summary answer covering main points"""
        if not chunks:
            return "No information available for summary.", []
        
        main_points = []
        citations = []
        
        for i, chunk in enumerate(chunks[:5]):
            # Extract first few sentences as main points
            sentences = chunk.text.split('.')[:3]  # First 3 sentences
            clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            if clean_sentences:
                main_points.extend(clean_sentences[:2])
                
                citation = Citation(
                    paper_id=chunk.paper_id,
                    paper_title=self._get_paper_title(chunk.paper_id),
                    authors=[],
                    chunk_text='. '.join(clean_sentences[:2]),
                    relevance_score=chunk.score
                )
                citations.append(citation)
        
        # Build summary answer
        if main_points:
            # Group and deduplicate similar points
            unique_points = []
            for point in main_points[:5]:  # Top 5 points
                if not any(self._text_similarity(point, existing) > 0.7 for existing in unique_points):
                    unique_points.append(point)
            
            answer = f"Key findings from the research: {' '.join(unique_points)}"
        else:
            answer = f"The research on {', '.join(query_context.keywords)} covers multiple aspects but specific summaries are limited."
        
        return answer, citations
    
    def _get_paper_title(self, paper_id: str) -> str:
        """Get paper title from database"""
        # This would typically query the database
        # For now, return a placeholder
        return f"Research Paper {paper_id}"
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_confidence(self, chunks: List[RetrievedChunk], citations: List[Citation]) -> float:
        """Calculate confidence score for the answer"""
        if not chunks or not citations:
            return 0.0
        
        # Base confidence on retrieval scores
        avg_retrieval_score = sum(chunk.score for chunk in chunks) / len(chunks)
        
        # Adjust based on number of supporting citations
        citation_factor = min(len(citations) / 3, 1.0)  # Max benefit from 3 citations
        
        # Adjust based on score distribution (higher confidence if scores are consistently high)
        score_std = np.std([chunk.score for chunk in chunks]) if HAS_NUMPY and len(chunks) > 1 else 0
        consistency_factor = max(0.5, 1.0 - score_std)
        
        confidence = avg_retrieval_score * citation_factor * consistency_factor
        return min(confidence, 1.0)

class EnterpriseQAPipeline:
    """Main Q&A pipeline orchestrating all components"""
    
    def __init__(self, db_path: str = "papers.db"):
        self.db_path = db_path
        self.query_processor = AdvancedQueryProcessor()
        self.retriever = HybridRetriever(db_path)
        self.answer_generator = AdvancedAnswerGenerator()
        self.metrics = []
        
        # Performance monitoring
        self.query_count = 0
        self.total_response_time = 0
        self.recent_queries = deque(maxlen=100)
        
        logger.info("Enterprise Q&A Pipeline initialized")
    
    def answer_question(self, query: str, paper_ids: List[str] = None, top_k: int = 10) -> QAResponse:
        """Process question and generate comprehensive answer"""
        start_time = time.time()
        self.query_count += 1
        
        try:
            # Step 1: Process query
            query_context = self.query_processor.process_query(query)
            
            # Step 2: Retrieve relevant chunks (optionally filtered by paper IDs)
            retrieval_start = time.time()
            if paper_ids:
                # Filter retrieval to specific papers if provided
                retrieved_chunks = self.retriever.retrieve_from_papers(query_context, paper_ids, top_k)
            else:
                retrieved_chunks = self.retriever.retrieve(query_context, top_k)
            retrieval_time = time.time() - retrieval_start
            
            # Step 3: Generate answer
            generation_start = time.time()
            response = self.answer_generator.generate_answer(query_context, retrieved_chunks)
            generation_time = time.time() - generation_start
            
            # Update response timing
            total_time = time.time() - start_time
            response.retrieval_time = retrieval_time
            response.total_time = total_time
            
            # Track performance
            self.total_response_time += total_time
            self.recent_queries.append({
                'query': query,
                'response_time': total_time,
                'confidence': response.confidence,
                'retrieved_chunks': len(retrieved_chunks),
                'timestamp': datetime.now()
            })
            
            # Record metrics
            self.metrics.append(PerformanceMetrics(
                operation="qa_pipeline",
                start_time=start_time,
                end_time=time.time(),
                duration=total_time,
                items_processed=1,
                throughput=1 / total_time if total_time > 0 else 0,
                cache_hits=0,
                cache_misses=0,
                error_count=0,
                memory_usage=sys.getsizeof(response)
            ))
            
            logger.info(f"Q&A completed in {total_time:.2f}s with confidence {response.confidence:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"Q&A pipeline failed: {e}")
            
            # Return error response
            return QAResponse(
                query=query,
                answer="I apologize, but I encountered an error while processing your question.",
                confidence=0.0,
                citations=[],
                retrieval_time=0,
                generation_time=0,
                total_time=time.time() - start_time,
                retrieved_chunks=0,
                method_used="error",
                metadata={'error': str(e)}
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        # Collect metrics from all components
        all_metrics = []
        all_metrics.extend(self.metrics)
        all_metrics.extend(self.query_processor.metrics)
        all_metrics.extend(self.retriever.metrics)
        all_metrics.extend(self.answer_generator.metrics)
        
        # Aggregate by operation
        operation_stats = defaultdict(list)
        for metric in all_metrics:
            operation_stats[metric.operation].append(metric)
        
        aggregated_metrics = {}
        for operation, metrics_list in operation_stats.items():
            durations = [m.duration for m in metrics_list]
            throughputs = [m.throughput for m in metrics_list if m.throughput > 0]
            
            aggregated_metrics[operation] = {
                'count': len(metrics_list),
                'avg_duration': np.mean(durations) if HAS_NUMPY and durations else (sum(durations) / len(durations) if durations else 0),
                'max_duration': max(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'avg_throughput': np.mean(throughputs) if HAS_NUMPY and throughputs else (sum(throughputs) / len(throughputs) if throughputs else 0)
            }
        
        # Overall pipeline stats
        avg_response_time = self.total_response_time / self.query_count if self.query_count > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_queries': self.query_count,
            'avg_response_time': avg_response_time,
            'operation_metrics': aggregated_metrics,
            'cache_stats': {
                'query_processor': self.query_processor.query_cache.get_stats(),
                'retriever': self.retriever.cache.get_stats(),
                'answer_generator': self.answer_generator.cache.get_stats()
            },
            'recent_query_stats': self._get_recent_query_stats()
        }
    
    def _get_recent_query_stats(self) -> Dict[str, Any]:
        """Get statistics for recent queries"""
        if not self.recent_queries:
            return {}
        
        response_times = [q['response_time'] for q in self.recent_queries]
        confidences = [q['confidence'] for q in self.recent_queries]
        
        return {
            'count': len(self.recent_queries),
            'avg_response_time': np.mean(response_times) if HAS_NUMPY else sum(response_times) / len(response_times),
            'max_response_time': max(response_times),
            'min_response_time': min(response_times),
            'avg_confidence': np.mean(confidences) if HAS_NUMPY else sum(confidences) / len(confidences),
            'high_confidence_queries': len([q for q in self.recent_queries if q['confidence'] > 0.7])
        }

def main():
    """Main function for testing the advanced Q&A system"""
    print("❓ Advanced Q&A System Test")
    print("=" * 50)
    
    # Initialize pipeline
    qa_pipeline = EnterpriseQAPipeline()
    
    # Test queries
    test_queries = [
        "What are the main contributions of machine learning research?",
        "How does deep learning compare to traditional methods?",
        "Explain the role of attention mechanisms in transformers",
        "Summarize the key findings in computer vision research",
        "What are the limitations of current NLP models?"
    ]
    
    print("\n🔍 Processing Test Queries:")
    print("-" * 30)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        
        response = qa_pipeline.answer_question(query)
        
        print(f"Answer: {response.answer[:200]}...")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Response Time: {response.total_time:.2f}s")
        print(f"Citations: {len(response.citations)}")
        print(f"Method: {response.method_used}")
    
    # Get performance metrics
    print("\n📊 Performance Metrics:")
    print("-" * 30)
    
    metrics = qa_pipeline.get_performance_metrics()
    
    print(f"Total Queries: {metrics['total_queries']}")
    print(f"Avg Response Time: {metrics['avg_response_time']:.2f}s")
    
    print("\nCache Performance:")
    for component, stats in metrics['cache_stats'].items():
        print(f"  {component}: {stats['hit_rate']:.2f} hit rate ({stats['hit_count']} hits, {stats['miss_count']} misses)")
    
    print("\nOperation Performance:")
    for operation, stats in metrics['operation_metrics'].items():
        print(f"  {operation}: {stats['count']} ops, {stats['avg_duration']:.4f}s avg")
    
    print("\n✅ Advanced Q&A System Test Complete!")

if __name__ == "__main__":
    main()
