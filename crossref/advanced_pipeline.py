"""
Advanced Cross-Referencing Pipeline
==================================

Enterprise-grade cross-referencing system with:
- Advanced citation analysis and extraction
- Semantic similarity using state-of-the-art embeddings
- Knowledge graph construction with relationship mapping
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
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import hashlib
from collections import defaultdict, Counter
import networkx as nx

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CitationMatch:
    """Represents a matched citation with confidence score"""
    source_paper_id: str
    target_paper_id: str
    citation_text: str
    confidence: float
    match_type: str  # 'exact', 'fuzzy', 'semantic'
    context: str
    page_number: Optional[int] = None

@dataclass
class SimilarityResult:
    """Represents similarity between two papers"""
    paper1_id: str
    paper2_id: str
    similarity_score: float
    similarity_type: str  # 'title', 'abstract', 'content', 'combined'
    matched_concepts: List[str]
    timestamp: datetime

@dataclass
class KnowledgeGraphEdge:
    """Represents an edge in the knowledge graph"""
    source: str
    target: str
    edge_type: str  # 'cites', 'similar_to', 'related_to', 'builds_on'
    weight: float
    metadata: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """Performance tracking for monitoring"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    items_processed: int
    throughput: float
    memory_usage: float
    error_count: int

class EnterpriseCache:
    """High-performance caching system with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.lock = threading.RLock()
        
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.creation_times:
            return True
        return time.time() - self.creation_times[key] > self.ttl_seconds
    
    def _evict_lru(self):
        """Evict least recently used items"""
        if len(self.cache) < self.max_size:
            return
            
        # Remove expired items first
        expired_keys = [k for k in self.cache.keys() if self._is_expired(k)]
        for key in expired_keys:
            self._remove_key(key)
            
        # If still over limit, remove LRU items
        while len(self.cache) >= self.max_size:
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self._remove_key(lru_key)
    
    def _remove_key(self, key: str):
        """Remove key from all caches"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache or self._is_expired(key):
                return None
            
            self.access_times[key] = time.time()
            return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        with self.lock:
            self._evict_lru()
            current_time = time.time()
            self.cache[key] = value
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
    
    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()

class AdvancedCitationExtractor:
    """Advanced citation extraction with multiple extraction strategies"""
    
    def __init__(self, cache_size: int = 5000):
        self.cache = EnterpriseCache(max_size=cache_size)
        self.metrics = []
        
        # Citation patterns for different formats
        self.citation_patterns = [
            # IEEE format: [1], [2], [1-3], [1,2,3]
            r'\[(\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)\]',
            # Nature format: (Smith et al., 2023)
            r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?(?:,\s*\d{4})?)\)',
            # APA format: (Smith, 2023; Jones, 2024)
            r'\(([A-Z][a-z]+(?:,\s*\d{4})?(?:;\s*[A-Z][a-z]+(?:,\s*\d{4})?)*)\)',
            # Vancouver format: (1,2,3)
            r'\((\d+(?:,\s*\d+)*)\)',
        ]
        
    def extract_citations_from_text(self, text: str, paper_id: str) -> List[CitationMatch]:
        """Extract citations from text using multiple strategies"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"citations_{hashlib.md5(text.encode()).hexdigest()}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        citations = []
        
        try:
            # Strategy 1: Pattern-based extraction
            citations.extend(self._extract_pattern_citations(text, paper_id))
            
            # Strategy 2: Context-based extraction
            citations.extend(self._extract_context_citations(text, paper_id))
            
            # Strategy 3: Reference section extraction
            citations.extend(self._extract_reference_citations(text, paper_id))
            
            # Deduplicate and score
            citations = self._deduplicate_citations(citations)
            
            # Cache results
            self.cache.set(cache_key, citations)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.append(PerformanceMetrics(
                operation="citation_extraction",
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                items_processed=len(citations),
                throughput=len(citations) / duration if duration > 0 else 0,
                memory_usage=sys.getsizeof(citations),
                error_count=0
            ))
            
            return citations
            
        except Exception as e:
            logger.error(f"Citation extraction failed: {e}")
            return []
    
    def _extract_pattern_citations(self, text: str, paper_id: str) -> List[CitationMatch]:
        """Extract citations using regex patterns"""
        import re
        citations = []
        
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                citations.append(CitationMatch(
                    source_paper_id=paper_id,
                    target_paper_id=f"ref_{match.group(1)}",
                    citation_text=match.group(0),
                    confidence=0.8,
                    match_type="pattern",
                    context=text[max(0, match.start()-50):match.end()+50]
                ))
        
        return citations
    
    def _extract_context_citations(self, text: str, paper_id: str) -> List[CitationMatch]:
        """Extract citations based on context clues"""
        citations = []
        
        # Look for common citation contexts
        context_indicators = [
            "according to", "as shown by", "following", "based on",
            "references", "cites", "mentions", "discusses"
        ]
        
        sentences = text.split('.')
        for sentence in sentences:
            for indicator in context_indicators:
                if indicator.lower() in sentence.lower():
                    # Extract potential citations from this sentence
                    words = sentence.split()
                    for i, word in enumerate(words):
                        if word.lower() == indicator.lower() and i < len(words) - 1:
                            context = ' '.join(words[max(0, i-5):i+10])
                            citations.append(CitationMatch(
                                source_paper_id=paper_id,
                                target_paper_id=f"context_{i}",
                                citation_text=context,
                                confidence=0.6,
                                match_type="context",
                                context=sentence
                            ))
        
        return citations
    
    def _extract_reference_citations(self, text: str, paper_id: str) -> List[CitationMatch]:
        """Extract citations from reference sections"""
        citations = []
        
        # Look for reference sections
        ref_section_start = -1
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in ['references', 'bibliography', 'works cited']):
                ref_section_start = i
                break
        
        if ref_section_start >= 0:
            ref_lines = lines[ref_section_start:ref_section_start + 100]  # Limit to 100 lines
            
            for i, line in enumerate(ref_lines):
                line = line.strip()
                if len(line) > 20 and not line.startswith('Figure') and not line.startswith('Table'):
                    citations.append(CitationMatch(
                        source_paper_id=paper_id,
                        target_paper_id=f"ref_line_{i}",
                        citation_text=line,
                        confidence=0.9,
                        match_type="reference",
                        context=line
                    ))
        
        return citations
    
    def _deduplicate_citations(self, citations: List[CitationMatch]) -> List[CitationMatch]:
        """Remove duplicate citations and merge similar ones"""
        unique_citations = {}
        
        for citation in citations:
            key = f"{citation.citation_text}_{citation.match_type}"
            if key not in unique_citations or citation.confidence > unique_citations[key].confidence:
                unique_citations[key] = citation
        
        return list(unique_citations.values())

class SemanticSimilarityEngine:
    """Advanced semantic similarity with multiple embedding models"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 10000):
        self.model_name = model_name
        self.model = None
        self.cache = EnterpriseCache(max_size=cache_size)
        self.metrics = []
        
        # Initialize model if available
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded semantic similarity model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
                self.model = None
        
        # Initialize FAISS index if available
        self.faiss_index = None
        self.paper_embeddings = {}
        if HAS_FAISS and self.model:
            self._initialize_faiss_index()
    
    def _initialize_faiss_index(self):
        """Initialize FAISS index for fast similarity search"""
        try:
            # Use inner product for cosine similarity
            dimension = self.model.get_sentence_embedding_dimension()
            self.faiss_index = faiss.IndexFlatIP(dimension)
            logger.info(f"Initialized FAISS index with dimension {dimension}")
        except Exception as e:
            logger.warning(f"Failed to initialize FAISS index: {e}")
    
    def compute_paper_embeddings(self, papers: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """Compute embeddings for all papers"""
        start_time = time.time()
        embeddings = {}
        
        if not self.model:
            logger.warning("No embedding model available")
            return embeddings
        
        try:
            paper_texts = []
            paper_ids = []
            
            for paper_id, paper_data in papers.items():
                # Combine title and abstract for embedding
                text_parts = []
                if paper_data.get('title'):
                    text_parts.append(paper_data['title'])
                if paper_data.get('abstract'):
                    text_parts.append(paper_data['abstract'])
                
                if text_parts:
                    combined_text = '. '.join(text_parts)
                    paper_texts.append(combined_text)
                    paper_ids.append(paper_id)
            
            if paper_texts:
                # Batch embedding computation
                batch_embeddings = self.model.encode(paper_texts, 
                                                   batch_size=32,
                                                   show_progress_bar=True)
                
                for paper_id, embedding in zip(paper_ids, batch_embeddings):
                    embeddings[paper_id] = embedding
                    self.paper_embeddings[paper_id] = embedding
                
                # Add to FAISS index
                if self.faiss_index is not None:
                    embeddings_array = np.array(list(embeddings.values())).astype('float32')
                    # Normalize for cosine similarity
                    faiss.normalize_L2(embeddings_array)
                    self.faiss_index.add(embeddings_array)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.append(PerformanceMetrics(
                operation="embedding_computation",
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                items_processed=len(embeddings),
                throughput=len(embeddings) / duration if duration > 0 else 0,
                memory_usage=sum(sys.getsizeof(emb) for emb in embeddings.values()),
                error_count=0
            ))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            return {}
    
    def find_similar_papers(self, query_paper_id: str, top_k: int = 10) -> List[SimilarityResult]:
        """Find most similar papers using FAISS index"""
        start_time = time.time()
        
        if not self.faiss_index or query_paper_id not in self.paper_embeddings:
            return []
        
        try:
            query_embedding = self.paper_embeddings[query_paper_id].reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search for similar papers
            similarities, indices = self.faiss_index.search(query_embedding, top_k + 1)
            
            results = []
            paper_ids = list(self.paper_embeddings.keys())
            
            for sim, idx in zip(similarities[0], indices[0]):
                if idx < len(paper_ids):
                    similar_paper_id = paper_ids[idx]
                    if similar_paper_id != query_paper_id:  # Exclude self
                        results.append(SimilarityResult(
                            paper1_id=query_paper_id,
                            paper2_id=similar_paper_id,
                            similarity_score=float(sim),
                            similarity_type="semantic",
                            matched_concepts=[],
                            timestamp=datetime.now()
                        ))
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.append(PerformanceMetrics(
                operation="similarity_search",
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                items_processed=len(results),
                throughput=len(results) / duration if duration > 0 else 0,
                memory_usage=sys.getsizeof(results),
                error_count=0
            ))
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def compute_pairwise_similarity(self, paper1_id: str, paper2_id: str) -> Optional[SimilarityResult]:
        """Compute similarity between two specific papers"""
        cache_key = f"sim_{paper1_id}_{paper2_id}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        if paper1_id not in self.paper_embeddings or paper2_id not in self.paper_embeddings:
            return None
        
        try:
            emb1 = self.paper_embeddings[paper1_id]
            emb2 = self.paper_embeddings[paper2_id]
            
            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            result = SimilarityResult(
                paper1_id=paper1_id,
                paper2_id=paper2_id,
                similarity_score=float(similarity),
                similarity_type="semantic",
                matched_concepts=[],
                timestamp=datetime.now()
            )
            
            self.cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Pairwise similarity computation failed: {e}")
            return None
    
    def compute_similarity(self, text1: str, text2: str, paper1_id: str, paper2_id: str) -> SimilarityResult:
        """Compute similarity between two text contents"""
        if not self.model:
            # Fallback to simple keyword-based similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if words1 and words2:
                similarity = len(words1.intersection(words2)) / len(words1.union(words2))
            else:
                similarity = 0.0
        else:
            # Use embedding-based similarity
            embeddings = self.model.encode([text1, text2])
            emb1, emb2 = embeddings[0], embeddings[1]
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return SimilarityResult(
            paper1_id=paper1_id,
            paper2_id=paper2_id,
            similarity_score=float(similarity),
            similarity_type="content_based",
            matched_concepts=[],
            timestamp=datetime.now()
        )

class EnterpriseKnowledgeGraph:
    """Advanced knowledge graph with enterprise features"""
    
    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.db_path = db_path
        self.graph = nx.MultiDiGraph()
        self.cache = EnterpriseCache(max_size=5000)
        self.metrics = []
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS papers (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        abstract TEXT,
                        authors TEXT,
                        published_date TEXT,
                        venue TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS edges (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_id TEXT,
                        target_id TEXT,
                        edge_type TEXT,
                        weight REAL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (source_id) REFERENCES papers (id),
                        FOREIGN KEY (target_id) REFERENCES papers (id)
                    )
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target_id)
                """)
                
                conn.commit()
                logger.info("Knowledge graph database initialized")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def add_paper(self, paper_id: str, paper_data: Dict[str, Any]):
        """Add paper to knowledge graph"""
        start_time = time.time()
        
        try:
            # Add to NetworkX graph
            self.graph.add_node(paper_id, **paper_data)
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO papers 
                    (id, title, abstract, authors, published_date, venue, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    paper_id,
                    paper_data.get('title', ''),
                    paper_data.get('abstract', ''),
                    json.dumps(paper_data.get('authors', [])),
                    paper_data.get('published_date', ''),
                    paper_data.get('venue', '')
                ))
                conn.commit()
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.append(PerformanceMetrics(
                operation="add_paper",
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                items_processed=1,
                throughput=1 / duration if duration > 0 else 0,
                memory_usage=sys.getsizeof(paper_data),
                error_count=0
            ))
            
        except Exception as e:
            logger.error(f"Failed to add paper {paper_id}: {e}")
    
    def add_edge(self, edge: KnowledgeGraphEdge):
        """Add edge to knowledge graph"""
        start_time = time.time()
        
        try:
            # Add to NetworkX graph
            self.graph.add_edge(
                edge.source,
                edge.target,
                edge_type=edge.edge_type,
                weight=edge.weight,
                **edge.metadata
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO edges (source_id, target_id, edge_type, weight, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    edge.source,
                    edge.target,
                    edge.edge_type,
                    edge.weight,
                    json.dumps(edge.metadata)
                ))
                conn.commit()
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.append(PerformanceMetrics(
                operation="add_edge",
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                items_processed=1,
                throughput=1 / duration if duration > 0 else 0,
                memory_usage=sys.getsizeof(edge),
                error_count=0
            ))
            
        except Exception as e:
            logger.error(f"Failed to add edge: {e}")
    
    def find_related_papers(self, paper_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Find papers related to given paper within max_depth"""
        cache_key = f"related_{paper_id}_{max_depth}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        start_time = time.time()
        
        try:
            if paper_id not in self.graph:
                return {}
            
            # Find neighbors within max_depth
            related_papers = {}
            
            # Direct neighbors (depth 1)
            direct_neighbors = list(self.graph.neighbors(paper_id))
            related_papers[1] = direct_neighbors
            
            # Extended neighbors (depth 2+)
            if max_depth > 1:
                extended_neighbors = set()
                for neighbor in direct_neighbors:
                    if neighbor in self.graph:
                        extended_neighbors.update(self.graph.neighbors(neighbor))
                
                extended_neighbors.discard(paper_id)  # Remove self
                related_papers[2] = list(extended_neighbors - set(direct_neighbors))
            
            # Get relationship details
            relationships = []
            for depth, papers in related_papers.items():
                for related_paper in papers:
                    if self.graph.has_edge(paper_id, related_paper):
                        edge_data = self.graph[paper_id][related_paper]
                        relationships.append({
                            'target': related_paper,
                            'depth': depth,
                            'relationship': edge_data.get(0, {}).get('edge_type', 'unknown'),
                            'weight': edge_data.get(0, {}).get('weight', 0.0)
                        })
            
            result = {
                'paper_id': paper_id,
                'related_papers': related_papers,
                'relationships': relationships,
                'total_related': sum(len(papers) for papers in related_papers.values())
            }
            
            self.cache.set(cache_key, result)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.append(PerformanceMetrics(
                operation="find_related_papers",
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                items_processed=result['total_related'],
                throughput=result['total_related'] / duration if duration > 0 else 0,
                memory_usage=sys.getsizeof(result),
                error_count=0
            ))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to find related papers for {paper_id}: {e}")
            return {}
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        try:
            stats = {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'avg_clustering': nx.average_clustering(self.graph.to_undirected()),
                'connected_components': nx.number_connected_components(self.graph.to_undirected()),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add degree statistics
            degrees = [d for n, d in self.graph.degree()]
            if degrees:
                stats.update({
                    'avg_degree': np.mean(degrees) if HAS_NUMPY else sum(degrees) / len(degrees),
                    'max_degree': max(degrees),
                    'min_degree': min(degrees)
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compute graph statistics: {e}")
            return {}

class AdvancedCrossReferencePipeline:
    """Main pipeline orchestrating all cross-referencing operations"""
    
    def __init__(self, 
                 db_path: str = "papers.db",
                 kg_db_path: str = "knowledge_graph.db",
                 cache_size: int = 10000):
        
        self.db_path = db_path
        self.citation_extractor = AdvancedCitationExtractor(cache_size=cache_size)
        self.similarity_engine = SemanticSimilarityEngine(cache_size=cache_size)
        self.knowledge_graph = EnterpriseKnowledgeGraph(kg_db_path)
        self.metrics = []
        
        # Load papers from database
        self.papers = self._load_papers()
        logger.info(f"Loaded {len(self.papers)} papers for cross-referencing")
    
    def _load_papers(self) -> Dict[str, Dict]:
        """Load papers from database"""
        papers = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, title, abstract, authors, published_date, venue
                    FROM papers
                    WHERE title IS NOT NULL
                """)
                
                for row in cursor.fetchall():
                    paper_id, title, abstract, authors, published_date, venue = row
                    papers[paper_id] = {
                        'id': paper_id,
                        'title': title,
                        'abstract': abstract or '',
                        'authors': json.loads(authors) if authors else [],
                        'published_date': published_date,
                        'venue': venue or ''
                    }
                    
        except Exception as e:
            logger.error(f"Failed to load papers: {e}")
            
        return papers
    
    def process_all_papers(self) -> Dict[str, Any]:
        """Process all papers for cross-referencing"""
        start_time = time.time()
        results = {
            'papers_processed': 0,
            'citations_extracted': 0,
            'similarities_computed': 0,
            'graph_edges_added': 0,
            'processing_time': 0,
            'errors': []
        }
        
        try:
            # Step 1: Add all papers to knowledge graph
            logger.info("Adding papers to knowledge graph...")
            for paper_id, paper_data in self.papers.items():
                self.knowledge_graph.add_paper(paper_id, paper_data)
                results['papers_processed'] += 1
            
            # Step 2: Compute embeddings for all papers
            logger.info("Computing semantic embeddings...")
            embeddings = self.similarity_engine.compute_paper_embeddings(self.papers)
            logger.info(f"Computed embeddings for {len(embeddings)} papers")
            
            # Step 3: Extract citations (if PDFs are available)
            logger.info("Extracting citations...")
            all_citations = []
            for paper_id, paper_data in self.papers.items():
                pdf_path = paper_data.get('pdf_path')
                if pdf_path and os.path.exists(pdf_path):
                    # For this implementation, we'll skip actual PDF processing
                    # In production, this would use PyMuPDF or similar
                    pass
                
                # For demo, create some mock citations based on abstracts
                if paper_data.get('abstract'):
                    citations = self.citation_extractor.extract_citations_from_text(
                        paper_data['abstract'], paper_id
                    )
                    all_citations.extend(citations)
                    results['citations_extracted'] += len(citations)
            
            # Step 4: Find similar papers and add edges
            logger.info("Computing similarities and building graph...")
            paper_ids = list(self.papers.keys())
            for i, paper_id in enumerate(paper_ids):
                # Find similar papers
                similar_papers = self.similarity_engine.find_similar_papers(paper_id, top_k=5)
                
                for similarity in similar_papers:
                    if similarity.similarity_score > 0.3:  # Threshold for meaningful similarity
                        edge = KnowledgeGraphEdge(
                            source=similarity.paper1_id,
                            target=similarity.paper2_id,
                            edge_type="similar_to",
                            weight=similarity.similarity_score,
                            metadata={
                                'similarity_type': similarity.similarity_type,
                                'timestamp': similarity.timestamp.isoformat()
                            }
                        )
                        self.knowledge_graph.add_edge(edge)
                        results['similarities_computed'] += 1
                        results['graph_edges_added'] += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(paper_ids)} papers")
            
            # Step 5: Add citation edges
            for citation in all_citations:
                if citation.confidence > 0.5:  # Only add high-confidence citations
                    edge = KnowledgeGraphEdge(
                        source=citation.source_paper_id,
                        target=citation.target_paper_id,
                        edge_type="cites",
                        weight=citation.confidence,
                        metadata={
                            'citation_text': citation.citation_text,
                            'match_type': citation.match_type,
                            'context': citation.context
                        }
                    )
                    self.knowledge_graph.add_edge(edge)
                    results['graph_edges_added'] += 1
            
            results['processing_time'] = time.time() - start_time
            
            # Record overall metrics
            self.metrics.append(PerformanceMetrics(
                operation="process_all_papers",
                start_time=start_time,
                end_time=time.time(),
                duration=results['processing_time'],
                items_processed=results['papers_processed'],
                throughput=results['papers_processed'] / results['processing_time'],
                memory_usage=0,  # Could be computed if needed
                error_count=len(results['errors'])
            ))
            
            logger.info(f"Cross-referencing complete: {results}")
            return results
            
        except Exception as e:
            error_msg = f"Cross-referencing pipeline failed: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            return results
    
    def extract_citations(self, text: str) -> List[CitationMatch]:
        """Extract citations from text (wrapper for consistency)"""
        return self.citation_extractor.extract_citations_from_text(text, "pipeline_extraction")
    
    def analyze_similarity(self, papers: List[Dict]) -> List[SimilarityResult]:
        """Analyze similarity between papers"""
        results = []
        paper_ids = [p.get('id', str(i)) for i, p in enumerate(papers)]
        
        for i, paper1 in enumerate(papers):
            for j, paper2 in enumerate(papers[i+1:], i+1):
                similarity = self.similarity_engine.compute_similarity(
                    paper1.get('content', paper1.get('abstract', '')),
                    paper2.get('content', paper2.get('abstract', '')),
                    paper_ids[i],
                    paper_ids[j]
                )
                results.append(similarity)
        
        return results
    
    def build_knowledge_graph(self, papers: List[Dict]) -> Dict[str, Any]:
        """Build knowledge graph from papers"""
        # Add papers to graph
        for paper in papers:
            paper_id = paper.get('id', str(hash(paper.get('title', ''))))
            self.knowledge_graph.add_paper(paper_id, paper)
        
        # Extract citations and add edges
        all_citations = []
        for paper in papers:
            paper_id = paper.get('id', str(hash(paper.get('title', ''))))
            content = paper.get('content', paper.get('abstract', ''))
            if content:
                citations = self.citation_extractor.extract_citations_from_text(content, paper_id)
                all_citations.extend(citations)
        
        # Add citation edges
        for citation in all_citations:
            if citation.confidence > 0.5:
                edge = KnowledgeGraphEdge(
                    source=citation.source_paper_id,
                    target=citation.target_paper_id,
                    edge_type="cites",
                    weight=citation.confidence,
                    metadata={'citation_text': citation.citation_text}
                )
                self.knowledge_graph.add_edge(edge)
        
        return self.knowledge_graph.get_graph_statistics()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        all_metrics = []
        all_metrics.extend(self.metrics)
        all_metrics.extend(self.citation_extractor.metrics)
        all_metrics.extend(self.similarity_engine.metrics)
        all_metrics.extend(self.knowledge_graph.metrics)
        
        if not all_metrics:
            return {}
        
        # Aggregate metrics by operation
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
                'avg_throughput': np.mean(throughputs) if HAS_NUMPY and throughputs else (sum(throughputs) / len(throughputs) if throughputs else 0),
                'total_items_processed': sum(m.items_processed for m in metrics_list),
                'total_errors': sum(m.error_count for m in metrics_list)
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_operations': len(all_metrics),
            'operation_metrics': aggregated_metrics,
            'graph_statistics': self.knowledge_graph.get_graph_statistics()
        }

def main():
    """Main function for testing the advanced cross-referencing pipeline"""
    print("🔗 Advanced Cross-Referencing Pipeline Test")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = AdvancedCrossReferencePipeline()
    
    # Process all papers
    results = pipeline.process_all_papers()
    
    # Get performance metrics
    metrics = pipeline.get_performance_metrics()
    
    # Display results
    print("\n📊 Processing Results:")
    print(f"Papers Processed: {results['papers_processed']}")
    print(f"Citations Extracted: {results['citations_extracted']}")
    print(f"Similarities Computed: {results['similarities_computed']}")
    print(f"Graph Edges Added: {results['graph_edges_added']}")
    print(f"Processing Time: {results['processing_time']:.2f}s")
    
    if results['errors']:
        print(f"\n⚠️ Errors: {len(results['errors'])}")
        for error in results['errors'][:3]:
            print(f"  - {error}")
    
    print("\n📈 Performance Metrics:")
    for operation, stats in metrics.get('operation_metrics', {}).items():
        print(f"{operation}:")
        print(f"  Count: {stats['count']}")
        print(f"  Avg Duration: {stats['avg_duration']:.4f}s")
        print(f"  Avg Throughput: {stats['avg_throughput']:.2f} items/s")
    
    print("\n🕸️ Graph Statistics:")
    graph_stats = metrics.get('graph_statistics', {})
    print(f"Nodes: {graph_stats.get('nodes', 0)}")
    print(f"Edges: {graph_stats.get('edges', 0)}")
    print(f"Density: {graph_stats.get('density', 0):.4f}")
    print(f"Average Clustering: {graph_stats.get('avg_clustering', 0):.4f}")
    
    print("\n✅ Advanced Cross-Referencing Pipeline Complete!")

if __name__ == "__main__":
    main()
