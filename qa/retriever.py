"""
Hybrid Retriever for RAG Pipeline

Combines FAISS semantic similarity search with BM25 keyword search
for comprehensive retrieval of relevant paper chunks.
"""

import os
import logging
import sqlite3
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievedChunk:
    """Data class for retrieved document chunks."""
    chunk_id: str
    paper_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str  # 'faiss' or 'bm25' or 'hybrid'

class HybridRetriever:
    """
    Hybrid retriever combining FAISS semantic search with BM25 keyword search.
    
    This retriever uses both semantic similarity (via sentence embeddings) and
    keyword matching (via BM25) to find the most relevant chunks for a query.
    """
    
    def __init__(
        self,
        faiss_index_path: str,
        bm25_index_path: str,
        chunk_metadata_path: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        alpha: float = 0.7  # Weight for FAISS vs BM25 (0.7 = 70% FAISS, 30% BM25)
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            faiss_index_path: Path to FAISS index file
            bm25_index_path: Path to BM25 index pickle file
            chunk_metadata_path: Path to chunk metadata SQLite database
            embedding_model_name: Name of sentence transformer model
            alpha: Weight for combining FAISS and BM25 scores (0-1)
        """
        self.faiss_index_path = faiss_index_path
        self.bm25_index_path = bm25_index_path
        self.chunk_metadata_path = chunk_metadata_path
        self.embedding_model_name = embedding_model_name
        self.alpha = alpha
        
        # Initialize components
        self.embedding_model = None
        self.faiss_index = None
        self.bm25_index = None
        self.chunk_metadata = {}
        self.chunk_texts = []
        
        # Load models and indexes
        self._load_embedding_model()
        self._load_faiss_index()
        self._load_bm25_index()
        self._load_chunk_metadata()
        
        logger.info(f"Hybrid retriever initialized with {len(self.chunk_metadata)} chunks")
    
    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings."""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _load_faiss_index(self):
        """Load FAISS index from file."""
        if os.path.exists(self.faiss_index_path):
            try:
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
                self.faiss_index = None
        else:
            logger.warning(f"FAISS index not found at {self.faiss_index_path}")
            self.faiss_index = None
    
    def _load_bm25_index(self):
        """Load BM25 index from pickle file."""
        if os.path.exists(self.bm25_index_path):
            try:
                with open(self.bm25_index_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                    self.bm25_index = bm25_data['index']
                    self.chunk_texts = bm25_data['texts']
                logger.info(f"Loaded BM25 index with {len(self.chunk_texts)} documents")
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")
                self.bm25_index = None
                self.chunk_texts = []
        else:
            logger.warning(f"BM25 index not found at {self.bm25_index_path}")
            self.bm25_index = None
            self.chunk_texts = []
    
    def _load_chunk_metadata(self):
        """Load chunk metadata from SQLite database."""
        if os.path.exists(self.chunk_metadata_path):
            try:
                conn = sqlite3.connect(self.chunk_metadata_path)
                cursor = conn.cursor()
                
                # Get all chunk metadata
                cursor.execute("""
                    SELECT chunk_id, paper_id, content, section, token_count, metadata
                    FROM chunks
                """)
                
                for row in cursor.fetchall():
                    chunk_id, paper_id, content, section, token_count, metadata_json = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    self.chunk_metadata[chunk_id] = {
                        'paper_id': paper_id,
                        'content': content,
                        'section': section,
                        'token_count': token_count,
                        'metadata': metadata
                    }
                
                conn.close()
                logger.info(f"Loaded metadata for {len(self.chunk_metadata)} chunks")
                
            except Exception as e:
                logger.warning(f"Failed to load chunk metadata: {e}")
                self.chunk_metadata = {}
        else:
            logger.warning(f"Chunk metadata database not found at {self.chunk_metadata_path}")
            self.chunk_metadata = {}
    
    def _embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query text."""
        if not self.embedding_model:
            raise ValueError("Embedding model not loaded")
        
        embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        return embedding.astype(np.float32)
    
    def _faiss_search(self, query: str, k: int) -> List[RetrievedChunk]:
        """Perform FAISS semantic similarity search."""
        if not self.faiss_index:
            logger.warning("FAISS index not available")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._embed_query(query).reshape(1, -1)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, k)
            
            chunks = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunk_metadata):
                    chunk_id = list(self.chunk_metadata.keys())[idx]
                    chunk_data = self.chunk_metadata[chunk_id]
                    
                    chunk = RetrievedChunk(
                        chunk_id=chunk_id,
                        paper_id=chunk_data['paper_id'],
                        content=chunk_data['content'],
                        score=float(score),
                        metadata=chunk_data['metadata'],
                        source='faiss'
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _bm25_search(self, query: str, k: int) -> List[RetrievedChunk]:
        """Perform BM25 keyword search."""
        if not self.bm25_index or not self.chunk_texts:
            logger.warning("BM25 index not available")
            return []
        
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:k]
            
            chunks = []
            for idx in top_indices:
                if idx < len(self.chunk_texts) and scores[idx] > 0:
                    # Find corresponding chunk ID
                    chunk_id = list(self.chunk_metadata.keys())[idx]
                    chunk_data = self.chunk_metadata[chunk_id]
                    
                    chunk = RetrievedChunk(
                        chunk_id=chunk_id,
                        paper_id=chunk_data['paper_id'],
                        content=chunk_data['content'],
                        score=float(scores[idx]),
                        metadata=chunk_data['metadata'],
                        source='bm25'
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _combine_results(
        self,
        faiss_chunks: List[RetrievedChunk],
        bm25_chunks: List[RetrievedChunk],
        k: int
    ) -> List[RetrievedChunk]:
        """Combine and rank results from FAISS and BM25."""
        
        # Normalize scores
        def normalize_scores(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
            if not chunks:
                return chunks
            
            scores = [chunk.score for chunk in chunks]
            min_score, max_score = min(scores), max(scores)
            
            if max_score == min_score:
                for chunk in chunks:
                    chunk.score = 1.0
            else:
                for chunk in chunks:
                    chunk.score = (chunk.score - min_score) / (max_score - min_score)
            
            return chunks
        
        # Normalize scores for both result sets
        faiss_chunks = normalize_scores(faiss_chunks)
        bm25_chunks = normalize_scores(bm25_chunks)
        
        # Combine results with weighting
        combined_scores = {}
        
        # Add FAISS results
        for chunk in faiss_chunks:
            combined_scores[chunk.chunk_id] = {
                'chunk': chunk,
                'faiss_score': chunk.score,
                'bm25_score': 0.0
            }
        
        # Add/update with BM25 results
        for chunk in bm25_chunks:
            if chunk.chunk_id in combined_scores:
                combined_scores[chunk.chunk_id]['bm25_score'] = chunk.score
            else:
                combined_scores[chunk.chunk_id] = {
                    'chunk': chunk,
                    'faiss_score': 0.0,
                    'bm25_score': chunk.score
                }
        
        # Calculate hybrid scores and create final results
        final_chunks = []
        for chunk_id, data in combined_scores.items():
            hybrid_score = (self.alpha * data['faiss_score'] + 
                          (1 - self.alpha) * data['bm25_score'])
            
            chunk = data['chunk']
            chunk.score = hybrid_score
            chunk.source = 'hybrid'
            final_chunks.append(chunk)
        
        # Sort by hybrid score and return top-k
        final_chunks.sort(key=lambda x: x.score, reverse=True)
        return final_chunks[:k]
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        use_hybrid: bool = True
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
            use_hybrid: Whether to use hybrid search (FAISS + BM25)
            
        Returns:
            List of retrieved chunks ranked by relevance
        """
        logger.info(f"Retrieving chunks for query: {query[:100]}...")
        
        if use_hybrid:
            # Perform both searches
            faiss_chunks = self._faiss_search(query, k * 2)  # Get more for better combination
            bm25_chunks = self._bm25_search(query, k * 2)
            
            # Combine results
            chunks = self._combine_results(faiss_chunks, bm25_chunks, k)
        else:
            # Use only FAISS search
            chunks = self._faiss_search(query, k)
        
        logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[RetrievedChunk]:
        """Get a specific chunk by ID."""
        if chunk_id not in self.chunk_metadata:
            return None
        
        chunk_data = self.chunk_metadata[chunk_id]
        return RetrievedChunk(
            chunk_id=chunk_id,
            paper_id=chunk_data['paper_id'],
            content=chunk_data['content'],
            score=1.0,
            metadata=chunk_data['metadata'],
            source='direct'
        )
    
    def get_paper_chunks(self, paper_id: str) -> List[RetrievedChunk]:
        """Get all chunks for a specific paper."""
        chunks = []
        for chunk_id, chunk_data in self.chunk_metadata.items():
            if chunk_data['paper_id'] == paper_id:
                chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    paper_id=paper_id,
                    content=chunk_data['content'],
                    score=1.0,
                    metadata=chunk_data['metadata'],
                    source='paper'
                )
                chunks.append(chunk)
        
        return chunks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            'total_chunks': len(self.chunk_metadata),
            'faiss_available': self.faiss_index is not None,
            'bm25_available': self.bm25_index is not None,
            'embedding_model': self.embedding_model_name,
            'hybrid_weight_alpha': self.alpha
        }

def create_bm25_index(
    chunk_texts: List[str],
    output_path: str
) -> None:
    """
    Create and save a BM25 index from chunk texts.
    
    Args:
        chunk_texts: List of text chunks
        output_path: Path to save the BM25 index
    """
    logger.info(f"Creating BM25 index from {len(chunk_texts)} chunks...")
    
    # Tokenize texts
    tokenized_texts = [text.lower().split() for text in chunk_texts]
    
    # Create BM25 index
    bm25_index = BM25Okapi(tokenized_texts)
    
    # Save index and texts
    bm25_data = {
        'index': bm25_index,
        'texts': chunk_texts
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(bm25_data, f)
    
    logger.info(f"BM25 index saved to {output_path}")

def create_chunk_database(
    chunks: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Create SQLite database for chunk metadata.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to save the database
    """
    logger.info(f"Creating chunk database with {len(chunks)} chunks...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()
    
    # Create chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            paper_id TEXT NOT NULL,
            content TEXT NOT NULL,
            section TEXT,
            token_count INTEGER,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert chunks
    for chunk in chunks:
        cursor.execute("""
            INSERT OR REPLACE INTO chunks 
            (chunk_id, paper_id, content, section, token_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            chunk['chunk_id'],
            chunk['paper_id'],
            chunk['content'],
            chunk.get('section', 'unknown'),
            chunk.get('token_count', 0),
            json.dumps(chunk.get('metadata', {}))
        ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Chunk database saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    retriever = HybridRetriever(
        faiss_index_path="data/processed/faiss_index.bin",
        bm25_index_path="data/processed/bm25_index.pkl",
        chunk_metadata_path="data/processed/chunks.db"
    )
    
    # Test retrieval
    results = retriever.retrieve("transformer architecture attention mechanism", k=5)
    
    for i, chunk in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {chunk.score:.3f}, Source: {chunk.source}) ---")
        print(f"Paper: {chunk.paper_id}")
        print(f"Content: {chunk.content[:200]}...")
