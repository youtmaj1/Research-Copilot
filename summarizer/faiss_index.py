"""
FAISS Vector Index

Vector database for storing and retrieving paper embeddings with metadata filtering.
Supports similarity search, clustering, and efficient storage/retrieval.
"""

import logging
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import uuid
from datetime import datetime

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .chunker import TextChunk
from .summarizer import SummaryResult

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingMetadata:
    """Metadata for embeddings stored in the vector index."""
    embedding_id: str
    source_file: str
    chunk_id: str
    section_title: str
    chunk_type: str
    paper_title: str
    authors: List[str]
    timestamp: str
    token_count: int
    char_count: int
    page_numbers: List[int]
    summary: Optional[str] = None
    key_points: Optional[List[str]] = None


@dataclass
class SearchResult:
    """Result from similarity search."""
    chunk: TextChunk
    similarity_score: float
    embedding_metadata: EmbeddingMetadata
    distance: float


class FaissVectorIndex:
    """
    FAISS-based vector index for research paper embeddings.
    
    Provides efficient similarity search, metadata filtering,
    and persistent storage capabilities.
    """
    
    def __init__(
        self,
        index_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        index_type: str = "flat",
        metric: str = "cosine"
    ):
        """
        Initialize the FAISS vector index.
        
        Args:
            index_path: Path to store the index files
            embedding_model: Sentence transformer model name
            dimension: Embedding dimension
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
            metric: Distance metric ('cosine', 'l2', 'inner_product')
        """
        if faiss is None:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")
        
        if SentenceTransformer is None:
            raise ImportError("SentenceTransformers not installed. Run: pip install sentence-transformers")
        
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model_name = embedding_model
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Update dimension based on actual model
        test_embedding = self.embedding_model.encode(["test"])
        self.dimension = test_embedding.shape[1]
        logger.info(f"Embedding dimension: {self.dimension}")
        
        # Initialize FAISS index
        self.index = None
        self.metadata_store = {}  # embedding_id -> EmbeddingMetadata
        self.id_to_index = {}     # embedding_id -> faiss_index
        self.index_to_id = {}     # faiss_index -> embedding_id
        
        self._init_index()
        self._load_existing_index()
    
    def _init_index(self):
        """Initialize the FAISS index based on configuration."""
        if self.index_type == "flat":
            if self.metric == "cosine":
                # Normalize vectors for cosine similarity
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.metric == "l2":
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                self.index = faiss.IndexFlatIP(self.dimension)  # Default to inner product
                
        elif self.index_type == "ivf":
            # IVF (Inverted File) index for faster search on large datasets
            quantizer = faiss.IndexFlatIP(self.dimension)
            n_centroids = min(100, max(10, 2 ** int(np.log2(1000))))  # Adaptive centroids
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_centroids)
            
        elif self.index_type == "hnsw":
            # HNSW (Hierarchical Navigable Small World) for very fast approximate search
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
            
        else:
            # Default to flat index
            self.index = faiss.IndexFlatIP(self.dimension)
        
        logger.info(f"Initialized {self.index_type} FAISS index with {self.metric} metric")
    
    def _load_existing_index(self):
        """Load existing index and metadata if available."""
        index_file = self.index_path / "faiss.index"
        metadata_file = self.index_path / "metadata.pkl"
        mappings_file = self.index_path / "mappings.pkl"
        
        if index_file.exists() and metadata_file.exists() and mappings_file.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_file))
                
                # Load metadata
                with open(metadata_file, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                
                # Load ID mappings
                with open(mappings_file, 'rb') as f:
                    mappings = pickle.load(f)
                    self.id_to_index = mappings['id_to_index']
                    self.index_to_id = mappings['index_to_id']
                
                logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
                
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}")
                # Reinitialize if loading fails
                self._init_index()
                self.metadata_store = {}
                self.id_to_index = {}
                self.index_to_id = {}
    
    def add_chunks(self, chunks: List[TextChunk], paper_title: str = "", authors: List[str] = None) -> List[str]:
        """
        Add text chunks to the vector index.
        
        Args:
            chunks: List of text chunks to add
            paper_title: Title of the paper
            authors: List of authors
            
        Returns:
            List of embedding IDs for the added chunks
        """
        if not chunks:
            return []
        
        logger.info(f"Adding {len(chunks)} chunks to vector index")
        
        # Extract text content for embedding
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Normalize embeddings if using cosine similarity
        if self.metric == "cosine":
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Generate embedding IDs and metadata
        embedding_ids = []
        current_time = datetime.now().isoformat()
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            embedding_id = str(uuid.uuid4())
            embedding_ids.append(embedding_id)
            
            # Create metadata
            metadata = EmbeddingMetadata(
                embedding_id=embedding_id,
                source_file=chunk.metadata.source_file,
                chunk_id=chunk.metadata.chunk_id,
                section_title=chunk.metadata.section_title,
                chunk_type=chunk.metadata.chunk_type,
                paper_title=paper_title,
                authors=authors or [],
                timestamp=current_time,
                token_count=chunk.metadata.token_count,
                char_count=chunk.metadata.char_count,
                page_numbers=chunk.metadata.page_numbers
            )
            
            # Store metadata
            self.metadata_store[embedding_id] = metadata
            
            # Update ID mappings
            faiss_index = self.index.ntotal + i
            self.id_to_index[embedding_id] = faiss_index
            self.index_to_id[faiss_index] = embedding_id
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            if self.index.ntotal >= self.index.nlist:
                logger.info("Training IVF index...")
                self.index.train(embeddings.astype(np.float32))
        
        logger.info(f"Added {len(chunks)} chunks. Total vectors: {self.index.ntotal}")
        return embedding_ids
    
    def add_summary(self, summary_result: SummaryResult, source_file: str):
        """
        Add summary information to existing embeddings.
        
        Args:
            summary_result: Summary result to add
            source_file: Source file path
        """
        # Find embeddings for this source file
        matching_ids = [eid for eid, metadata in self.metadata_store.items() 
                       if metadata.source_file == source_file]
        
        if not matching_ids:
            logger.warning(f"No embeddings found for source file: {source_file}")
            return
        
        # Add summary info to metadata
        for embedding_id in matching_ids:
            metadata = self.metadata_store[embedding_id]
            metadata.summary = summary_result.summary
            metadata.key_points = summary_result.key_points
        
        logger.info(f"Added summary information to {len(matching_ids)} embeddings")
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Metadata filters to apply
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of search results sorted by similarity
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Normalize if using cosine similarity
        if self.metric == "cosine":
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding.astype(np.float32), min(k * 2, self.index.ntotal))
        
        # Process results
        results = []
        for distance, faiss_idx in zip(distances[0], indices[0]):
            if faiss_idx == -1:  # Invalid index
                continue
            
            # Get embedding ID
            embedding_id = self.index_to_id.get(faiss_idx)
            if not embedding_id:
                continue
            
            # Get metadata
            metadata = self.metadata_store.get(embedding_id)
            if not metadata:
                continue
            
            # Apply metadata filters
            if filter_metadata and not self._matches_filter(metadata, filter_metadata):
                continue
            
            # Convert distance to similarity score
            if self.metric == "cosine":
                similarity_score = float(1 - distance)
            elif self.metric == "l2":
                similarity_score = float(1 / (1 + distance))
            else:
                similarity_score = float(distance)
            
            # Apply similarity threshold
            if similarity_score < min_similarity:
                continue
            
            # Reconstruct text chunk (simplified)
            chunk = TextChunk(
                content=f"[Content for {metadata.chunk_id}]",  # Would need to store actual content
                metadata=type('ChunkMetadata', (), {
                    'chunk_id': metadata.chunk_id,
                    'source_file': metadata.source_file,
                    'section_title': metadata.section_title,
                    'chunk_type': metadata.chunk_type,
                    'token_count': metadata.token_count,
                    'char_count': metadata.char_count,
                    'page_numbers': metadata.page_numbers
                })()
            )
            
            result = SearchResult(
                chunk=chunk,
                similarity_score=similarity_score,
                embedding_metadata=metadata,
                distance=float(distance)
            )
            results.append(result)
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results[:k]
    
    def _matches_filter(self, metadata: EmbeddingMetadata, filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters."""
        for key, value in filters.items():
            if not hasattr(metadata, key):
                continue
            
            metadata_value = getattr(metadata, key)
            
            if isinstance(value, list):
                if metadata_value not in value:
                    return False
            elif isinstance(value, str):
                if value.lower() not in str(metadata_value).lower():
                    return False
            else:
                if metadata_value != value:
                    return False
        
        return True
    
    def get_similar_papers(self, source_file: str, k: int = 5) -> List[SearchResult]:
        """
        Find papers similar to a given paper.
        
        Args:
            source_file: Source file to find similar papers for
            k: Number of similar papers to return
            
        Returns:
            List of similar papers
        """
        # Get embeddings for the source file
        source_embeddings = [eid for eid, metadata in self.metadata_store.items() 
                           if metadata.source_file == source_file]
        
        if not source_embeddings:
            return []
        
        # Use the first chunk as representative (could be improved)
        representative_id = source_embeddings[0]
        faiss_idx = self.id_to_index.get(representative_id)
        
        if faiss_idx is None:
            return []
        
        # Get the embedding vector
        embedding = self.index.reconstruct(faiss_idx).reshape(1, -1)
        
        # Search for similar vectors
        distances, indices = self.index.search(embedding, k + len(source_embeddings))
        
        # Filter out the source paper itself
        results = []
        seen_papers = set()
        
        for distance, faiss_idx in zip(distances[0], indices[0]):
            if faiss_idx == -1:
                continue
            
            embedding_id = self.index_to_id.get(faiss_idx)
            if not embedding_id:
                continue
            
            metadata = self.metadata_store.get(embedding_id)
            if not metadata or metadata.source_file == source_file:
                continue
            
            # Avoid duplicate papers
            if metadata.source_file in seen_papers:
                continue
            seen_papers.add(metadata.source_file)
            
            # Convert distance to similarity
            if self.metric == "cosine":
                similarity_score = float(1 - distance)
            else:
                similarity_score = float(1 / (1 + distance))
            
            # Create dummy chunk
            chunk = TextChunk(
                content=f"[Paper: {metadata.paper_title}]",
                metadata=type('ChunkMetadata', (), {
                    'source_file': metadata.source_file,
                    'section_title': metadata.section_title,
                    'chunk_type': metadata.chunk_type
                })()
            )
            
            result = SearchResult(
                chunk=chunk,
                similarity_score=similarity_score,
                embedding_metadata=metadata,
                distance=float(distance)
            )
            results.append(result)
        
        return results[:k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        if not self.metadata_store:
            return {}
        
        # Count by type
        chunk_types = {}
        papers = set()
        
        for metadata in self.metadata_store.values():
            chunk_types[metadata.chunk_type] = chunk_types.get(metadata.chunk_type, 0) + 1
            papers.add(metadata.source_file)
        
        return {
            'total_embeddings': len(self.metadata_store),
            'total_papers': len(papers),
            'chunk_types': chunk_types,
            'index_size': self.index.ntotal,
            'embedding_dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric
        }
    
    def save_index(self):
        """Save the index and metadata to disk."""
        try:
            # Save FAISS index
            index_file = self.index_path / "faiss.index"
            faiss.write_index(self.index, str(index_file))
            
            # Save metadata
            metadata_file = self.index_path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            # Save ID mappings
            mappings_file = self.index_path / "mappings.pkl"
            with open(mappings_file, 'wb') as f:
                pickle.dump({
                    'id_to_index': self.id_to_index,
                    'index_to_id': self.index_to_id
                }, f)
            
            # Save configuration
            config_file = self.index_path / "config.json"
            with open(config_file, 'w') as f:
                json.dump({
                    'embedding_model': self.embedding_model_name,
                    'dimension': self.dimension,
                    'index_type': self.index_type,
                    'metric': self.metric,
                    'created': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Saved index with {self.index.ntotal} vectors to {self.index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def delete_paper(self, source_file: str) -> int:
        """
        Delete all embeddings for a specific paper.
        
        Args:
            source_file: Source file to delete
            
        Returns:
            Number of embeddings deleted
        """
        # Find embeddings to delete
        to_delete = [eid for eid, metadata in self.metadata_store.items() 
                    if metadata.source_file == source_file]
        
        if not to_delete:
            return 0
        
        # Remove from metadata store
        for embedding_id in to_delete:
            del self.metadata_store[embedding_id]
        
        # Remove from ID mappings
        for embedding_id in to_delete:
            if embedding_id in self.id_to_index:
                faiss_idx = self.id_to_index[embedding_id]
                del self.id_to_index[embedding_id]
                if faiss_idx in self.index_to_id:
                    del self.index_to_id[faiss_idx]
        
        # Note: FAISS doesn't support deletion, so we'd need to rebuild the index
        # For now, we just remove from metadata
        logger.warning(f"Deleted {len(to_delete)} embeddings from metadata. "
                      "Index rebuild required for complete removal.")
        
        return len(to_delete)
    
    def rebuild_index(self):
        """Rebuild the FAISS index from metadata (useful after deletions)."""
        if not self.metadata_store:
            logger.info("No metadata to rebuild index from")
            return
        
        logger.info("Rebuilding FAISS index...")
        
        # This would require re-embedding all content
        # For now, we just reinitialize an empty index
        self._init_index()
        self.id_to_index = {}
        self.index_to_id = {}
        
        logger.warning("Index rebuilt as empty. Re-add content to populate.")


def create_vector_index(
    index_path: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    index_type: str = "flat"
) -> FaissVectorIndex:
    """
    Convenience function to create a vector index.
    
    Args:
        index_path: Path to store the index
        embedding_model: Embedding model to use
        index_type: Type of FAISS index
        
    Returns:
        Initialized vector index
    """
    return FaissVectorIndex(
        index_path=index_path,
        embedding_model=embedding_model,
        index_type=index_type
    )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python faiss_index.py <index_path> [action]")
        print("Actions: create, stats, search")
        sys.exit(1)
    
    index_path = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else "stats"
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        if action == "create":
            # Create new index
            index = create_vector_index(index_path)
            print(f"Created vector index at: {index_path}")
            
        elif action == "stats":
            # Show statistics
            index = FaissVectorIndex(index_path)
            stats = index.get_statistics()
            print("Index Statistics:")
            print(json.dumps(stats, indent=2))
            
        elif action == "search":
            # Interactive search
            index = FaissVectorIndex(index_path)
            
            while True:
                query = input("\nEnter search query (or 'quit'): ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                results = index.search(query, k=5)
                print(f"\nFound {len(results)} results:")
                
                for i, result in enumerate(results, 1):
                    print(f"\n{i}. Similarity: {result.similarity_score:.3f}")
                    print(f"   Paper: {result.embedding_metadata.paper_title}")
                    print(f"   Section: {result.embedding_metadata.section_title}")
                    print(f"   Type: {result.embedding_metadata.chunk_type}")
        
        else:
            print(f"Unknown action: {action}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
