"""
Semantic Similarity Engine

Finds semantically similar papers using FAISS embeddings.
Computes cosine similarity between paper embeddings for topic overlap detection.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import json
import pickle

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Represents a similarity match between papers."""
    source_paper_id: str
    target_paper_id: str
    similarity_score: float
    match_type: str = "semantic"  # 'semantic', 'title', 'abstract'
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SimilarityEngine:
    """
    Semantic similarity engine using FAISS for efficient similarity search.
    
    Supports multiple similarity types:
    - Full paper semantic similarity
    - Abstract-only similarity
    - Title similarity
    - Combined similarity scores
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        faiss_index_path: Optional[str] = None,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize similarity engine.
        
        Args:
            embedding_model_name: Name of sentence transformer model
            faiss_index_path: Path to existing FAISS index
            similarity_threshold: Minimum similarity score to consider
        """
        self.embedding_model_name = embedding_model_name
        self.faiss_index_path = faiss_index_path
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.embedding_model = None
        self.faiss_index = None
        self.paper_metadata = {}  # paper_id -> metadata
        self.paper_embeddings = {}  # paper_id -> embedding
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        # Load existing index if provided
        if faiss_index_path and Path(faiss_index_path).exists():
            self.load_index(faiss_index_path)
        
        logger.info(f"Similarity engine initialized with model: {embedding_model_name}")
    
    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model."""
        if SentenceTransformer is None:
            logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
            return
        
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.embedding_model_name}: {e}")
    
    def add_papers(
        self, 
        papers: Dict[str, Dict],
        text_field: str = "full_text"
    ):
        """
        Add papers to the similarity index.
        
        Args:
            papers: Dict of paper_id -> paper data
            text_field: Field containing text to embed
        """
        if self.embedding_model is None:
            logger.error("Embedding model not available")
            return
        
        new_papers = []
        new_embeddings = []
        
        for paper_id, paper_data in papers.items():
            if paper_id in self.paper_metadata:
                continue  # Skip already indexed papers
            
            # Get text to embed
            text = self._get_text_for_embedding(paper_data, text_field)
            if not text:
                logger.warning(f"No text found for paper {paper_id}")
                continue
            
            try:
                # Generate embedding
                embedding = self.embedding_model.encode(text, normalize_embeddings=True)
                
                # Store paper data
                self.paper_metadata[paper_id] = paper_data
                self.paper_embeddings[paper_id] = embedding
                
                new_papers.append(paper_id)
                new_embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Failed to process paper {paper_id}: {e}")
                continue
        
        if new_embeddings:
            self._add_embeddings_to_index(new_papers, new_embeddings)
            logger.info(f"Added {len(new_embeddings)} papers to similarity index")
    
    def _get_text_for_embedding(self, paper_data: Dict, text_field: str) -> str:
        """Get text content from paper data for embedding."""
        # Try different text sources in order of preference
        text_sources = [
            text_field,
            "summary",
            "abstract", 
            "title",
            "content",
            "text"
        ]
        
        for source in text_sources:
            if source in paper_data and paper_data[source]:
                text = paper_data[source]
                
                # If it's a summary object with multiple fields
                if isinstance(text, dict):
                    if "summary" in text:
                        return text["summary"]
                    elif "abstract" in text:
                        return text["abstract"]
                    else:
                        # Combine available text fields
                        combined_text = ""
                        for key, value in text.items():
                            if isinstance(value, str) and value:
                                combined_text += f"{value} "
                        return combined_text.strip()
                
                # If it's a list of chunks
                elif isinstance(text, list):
                    if all(isinstance(item, dict) for item in text):
                        # List of chunk objects
                        combined_text = ""
                        for chunk in text:
                            if "content" in chunk:
                                combined_text += f"{chunk['content']} "
                            elif "text" in chunk:
                                combined_text += f"{chunk['text']} "
                        return combined_text.strip()
                    else:
                        # List of strings
                        return " ".join(str(item) for item in text)
                
                # Regular string
                elif isinstance(text, str):
                    return text
        
        # Fallback: combine title and abstract
        title = paper_data.get("title", "")
        abstract = paper_data.get("abstract", "")
        
        if title or abstract:
            return f"{title} {abstract}".strip()
        
        return ""
    
    def _add_embeddings_to_index(self, paper_ids: List[str], embeddings: List[np.ndarray]):
        """Add embeddings to FAISS index."""
        if faiss is None:
            logger.error("FAISS not available. Install with: pip install faiss-cpu")
            return
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        if self.faiss_index is None:
            # Create new index
            dimension = embeddings_array.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            logger.info(f"Created new FAISS index with dimension {dimension}")
        
        # Add embeddings to index
        self.faiss_index.add(embeddings_array)
    
    def find_similar_papers(
        self, 
        query_paper_id: str,
        k: int = 10,
        exclude_self: bool = True
    ) -> List[SimilarityResult]:
        """
        Find papers similar to a given paper.
        
        Args:
            query_paper_id: ID of query paper
            k: Number of similar papers to return
            exclude_self: Whether to exclude the query paper from results
            
        Returns:
            List of similarity results
        """
        if query_paper_id not in self.paper_embeddings:
            logger.warning(f"Paper {query_paper_id} not in index")
            return []
        
        query_embedding = self.paper_embeddings[query_paper_id]
        return self._search_by_embedding(
            query_embedding, 
            k=k + (1 if exclude_self else 0),
            exclude_paper_id=query_paper_id if exclude_self else None
        )
    
    def find_similar_by_text(
        self,
        query_text: str,
        k: int = 10
    ) -> List[SimilarityResult]:
        """
        Find papers similar to given text.
        
        Args:
            query_text: Text to search for
            k: Number of results to return
            
        Returns:
            List of similarity results
        """
        if self.embedding_model is None:
            logger.error("Embedding model not available")
            return []
        
        try:
            query_embedding = self.embedding_model.encode(query_text, normalize_embeddings=True)
            return self._search_by_embedding(query_embedding, k=k)
        except Exception as e:
            logger.error(f"Failed to search by text: {e}")
            return []
    
    def _search_by_embedding(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        exclude_paper_id: Optional[str] = None
    ) -> List[SimilarityResult]:
        """Search for similar papers by embedding."""
        if self.faiss_index is None:
            logger.warning("FAISS index not initialized")
            return []
        
        try:
            # Reshape for FAISS
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Search
            scores, indices = self.faiss_index.search(query_vector, k)
            
            results = []
            paper_ids = list(self.paper_metadata.keys())
            
            for score, idx in zip(scores[0], indices[0]):
                if idx >= len(paper_ids):
                    continue
                
                paper_id = paper_ids[idx]
                
                # Skip excluded paper
                if exclude_paper_id and paper_id == exclude_paper_id:
                    continue
                
                # Apply similarity threshold
                if score < self.similarity_threshold:
                    continue
                
                result = SimilarityResult(
                    source_paper_id="query",
                    target_paper_id=paper_id,
                    similarity_score=float(score),
                    match_type="semantic",
                    metadata=self.paper_metadata[paper_id].copy()
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search by embedding: {e}")
            return []
    
    def compute_pairwise_similarities(
        self, 
        paper_ids: Optional[List[str]] = None,
        min_similarity: Optional[float] = None
    ) -> List[SimilarityResult]:
        """
        Compute pairwise similarities between all papers or given subset.
        
        Args:
            paper_ids: Specific papers to compare (None for all)
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similarity results
        """
        if min_similarity is None:
            min_similarity = self.similarity_threshold
        
        if paper_ids is None:
            paper_ids = list(self.paper_metadata.keys())
        
        results = []
        
        for i, paper_id_1 in enumerate(paper_ids):
            if paper_id_1 not in self.paper_embeddings:
                continue
            
            # Find similar papers
            similar_papers = self.find_similar_papers(
                paper_id_1, 
                k=len(paper_ids),
                exclude_self=True
            )
            
            # Filter by paper_ids and minimum similarity
            for result in similar_papers:
                if (result.target_paper_id in paper_ids and 
                    result.similarity_score >= min_similarity):
                    
                    result.source_paper_id = paper_id_1
                    results.append(result)
        
        # Remove duplicates (A->B and B->A)
        seen_pairs = set()
        unique_results = []
        
        for result in results:
            pair = tuple(sorted([result.source_paper_id, result.target_paper_id]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_results.append(result)
        
        logger.info(f"Computed {len(unique_results)} unique similarity pairs")
        return unique_results
    
    def find_topic_clusters(
        self,
        similarity_threshold: float = 0.8,
        min_cluster_size: int = 2
    ) -> List[List[str]]:
        """
        Find clusters of similar papers (topic groups).
        
        Args:
            similarity_threshold: Minimum similarity for clustering
            min_cluster_size: Minimum papers per cluster
            
        Returns:
            List of clusters (each cluster is a list of paper IDs)
        """
        # Get all pairwise similarities above threshold
        similarities = self.compute_pairwise_similarities(
            min_similarity=similarity_threshold
        )
        
        # Build adjacency graph
        adjacency = {}
        for result in similarities:
            source = result.source_paper_id
            target = result.target_paper_id
            
            if source not in adjacency:
                adjacency[source] = set()
            if target not in adjacency:
                adjacency[target] = set()
            
            adjacency[source].add(target)
            adjacency[target].add(source)
        
        # Find connected components (clusters)
        visited = set()
        clusters = []
        
        for paper_id in adjacency:
            if paper_id not in visited:
                cluster = self._find_connected_component(paper_id, adjacency, visited)
                if len(cluster) >= min_cluster_size:
                    clusters.append(list(cluster))
        
        logger.info(f"Found {len(clusters)} topic clusters")
        return clusters
    
    def _find_connected_component(
        self, 
        start_node: str, 
        adjacency: Dict[str, Set[str]], 
        visited: Set[str]
    ) -> Set[str]:
        """Find connected component starting from given node."""
        component = set()
        stack = [start_node]
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.add(node)
                
                for neighbor in adjacency.get(node, []):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return component
    
    def get_paper_similarities_matrix(
        self, 
        paper_ids: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get similarity matrix for given papers.
        
        Args:
            paper_ids: List of paper IDs (None for all)
            
        Returns:
            Tuple of (similarity_matrix, paper_ids_ordered)
        """
        if paper_ids is None:
            paper_ids = list(self.paper_metadata.keys())
        
        # Filter to papers we have embeddings for
        valid_paper_ids = [pid for pid in paper_ids if pid in self.paper_embeddings]
        
        if not valid_paper_ids:
            return np.array([]), []
        
        # Get embeddings matrix
        embeddings = np.array([
            self.paper_embeddings[pid] 
            for pid in valid_paper_ids
        ])
        
        # Compute cosine similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        return similarity_matrix, valid_paper_ids
    
    def save_index(self, output_path: str):
        """Save FAISS index and metadata to disk."""
        if self.faiss_index is None:
            logger.warning("No FAISS index to save")
            return
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_file = output_path / "similarity.index"
        faiss.write_index(self.faiss_index, str(faiss_file))
        
        # Save metadata
        metadata_file = output_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'paper_metadata': self.paper_metadata,
                'paper_embeddings': self.paper_embeddings,
                'embedding_model_name': self.embedding_model_name,
                'similarity_threshold': self.similarity_threshold
            }, f)
        
        # Save readable metadata
        readable_metadata_file = output_path / "metadata.json"
        readable_metadata = {
            'paper_count': len(self.paper_metadata),
            'embedding_model': self.embedding_model_name,
            'similarity_threshold': self.similarity_threshold,
            'papers': {
                pid: {
                    'title': info.get('title', ''),
                    'authors': info.get('authors', []),
                    'year': info.get('published_date', '')[:4] if info.get('published_date') else ''
                }
                for pid, info in self.paper_metadata.items()
            }
        }
        
        with open(readable_metadata_file, 'w') as f:
            json.dump(readable_metadata, f, indent=2)
        
        logger.info(f"Saved similarity index to {output_path}")
    
    def load_index(self, input_path: str):
        """Load FAISS index and metadata from disk."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            logger.error(f"Index path does not exist: {input_path}")
            return
        
        try:
            # Load FAISS index
            faiss_file = input_path / "similarity.index"
            if faiss_file.exists():
                self.faiss_index = faiss.read_index(str(faiss_file))
            
            # Load metadata
            metadata_file = input_path / "metadata.pkl"
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.paper_metadata = data['paper_metadata']
                    self.paper_embeddings = data['paper_embeddings']
                    self.embedding_model_name = data.get('embedding_model_name', self.embedding_model_name)
                    self.similarity_threshold = data.get('similarity_threshold', self.similarity_threshold)
            
            logger.info(f"Loaded similarity index from {input_path}")
            logger.info(f"Loaded {len(self.paper_metadata)} papers")
            
        except Exception as e:
            logger.error(f"Failed to load similarity index: {e}")
    
    def cluster_papers(self, n_clusters: int = 5) -> Dict[str, int]:
        """
        Cluster papers based on similarity.
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary mapping paper_id to cluster_id
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.error("scikit-learn not available for clustering")
            return {}
        
        if not self.paper_metadata or self.faiss_index is None:
            logger.warning("No papers or index available for clustering")
            return {}
        
        # Get all embeddings
        paper_ids = list(self.paper_metadata.keys())
        embeddings = []
        
        for paper_id in paper_ids:
            if paper_id in self.paper_metadata:
                # Get embedding from index
                embedding = self._get_paper_embedding(paper_id)
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    # Generate embedding if not found
                    text = self._get_paper_text(paper_id)
                    embedding = self._generate_embedding(text)
                    embeddings.append(embedding)
        
        if not embeddings:
            return {}
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Create mapping
        clusters = {}
        for paper_id, cluster_id in zip(paper_ids, cluster_labels):
            clusters[paper_id] = int(cluster_id)
        
        return clusters
    
    def _get_paper_embedding(self, paper_id: str):
        """Get embedding for a specific paper from the index."""
        if self.faiss_index is None or paper_id not in self.paper_metadata:
            return None
        
        # For simplicity, regenerate embedding
        # In a full implementation, you'd store paper_id to index mappings
        text = self._get_paper_text(paper_id)
        return self._generate_embedding(text)
    
    def _get_paper_text(self, paper_id: str) -> str:
        """Get text content for a paper."""
        if paper_id not in self.paper_metadata:
            return ""
        
        paper = self.paper_metadata[paper_id]
        
        # Try different text fields in order of preference
        text_fields = ['full_text', 'abstract', 'title']
        
        for field in text_fields:
            if field in paper and paper[field]:
                return str(paper[field])
        
        # Fallback to title + abstract combination
        text_parts = []
        if paper.get('title'):
            text_parts.append(paper['title'])
        if paper.get('abstract'):
            text_parts.append(paper['abstract'])
        
        return ' '.join(text_parts) if text_parts else ""
    
    def _generate_embedding(self, text: str):
        """Generate embedding for given text."""
        if not text or not self.embedding_model:
            return None
        
        try:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """Get statistics about the similarity index."""
        stats = {
            'total_papers': len(self.paper_metadata),
            'embedding_model': self.embedding_model_name,
            'similarity_threshold': self.similarity_threshold,
            'index_initialized': self.faiss_index is not None
        }
        
        if self.faiss_index:
            stats['index_size'] = self.faiss_index.ntotal
            stats['index_dimension'] = self.faiss_index.d
        
        return stats


def compute_similarities_batch(
    papers: Dict[str, Dict],
    output_path: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.7,
    text_field: str = "full_text"
) -> List[SimilarityResult]:
    """
    Convenience function to compute similarities for a batch of papers.
    
    Args:
        papers: Dict of paper_id -> paper data
        output_path: Path to save similarity index
        embedding_model: Embedding model name
        similarity_threshold: Minimum similarity threshold
        text_field: Field containing text to embed
        
    Returns:
        List of similarity results
    """
    # Initialize similarity engine
    engine = SimilarityEngine(
        embedding_model_name=embedding_model,
        similarity_threshold=similarity_threshold
    )
    
    # Add papers
    engine.add_papers(papers, text_field=text_field)
    
    # Compute pairwise similarities
    similarities = engine.compute_pairwise_similarities()
    
    # Save index
    engine.save_index(output_path)
    
    return similarities


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python similarity.py <command> [args...]")
        print("Commands:")
        print("  create <papers.json> <output_dir> - Create similarity index")
        print("  search <index_dir> <query_text> - Search for similar papers")
        print("  stats <index_dir> - Show index statistics")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "create" and len(sys.argv) >= 4:
        papers_file = sys.argv[2]
        output_dir = sys.argv[3]
        
        # Load papers
        with open(papers_file, 'r') as f:
            papers = json.load(f)
        
        # Compute similarities
        similarities = compute_similarities_batch(papers, output_dir)
        print(f"Created similarity index with {len(similarities)} relationships")
    
    elif command == "search" and len(sys.argv) >= 4:
        index_dir = sys.argv[2]
        query_text = " ".join(sys.argv[3:])
        
        # Load index
        engine = SimilarityEngine(faiss_index_path=index_dir)
        
        # Search
        results = engine.find_similar_by_text(query_text, k=5)
        
        print(f"Found {len(results)} similar papers:")
        for i, result in enumerate(results, 1):
            paper_info = result.metadata
            print(f"{i}. {paper_info.get('title', 'Unknown Title')}")
            print(f"   Similarity: {result.similarity_score:.3f}")
            print(f"   Authors: {', '.join(paper_info.get('authors', []))}")
            print()
    
    elif command == "stats" and len(sys.argv) >= 3:
        index_dir = sys.argv[2]
        
        # Load index
        engine = SimilarityEngine(faiss_index_path=index_dir)
        stats = engine.get_statistics()
        
        print("Similarity Index Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    else:
        print("Invalid command or arguments")
        sys.exit(1)
