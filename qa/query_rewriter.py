"""
Query Rewriter for RAG Pipeline

Expands and rewrites user queries to improve retrieval performance
by generating synonyms, alternative phrasings, and domain-specific terms.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryExpansion:
    """Represents an expanded query with metadata."""
    original_query: str
    expanded_query: str
    expansion_terms: List[str]
    confidence: float
    method: str  # 'llm', 'synonym', 'domain_specific'

class QueryRewriter:
    """
    Query rewriter that expands user queries for better retrieval.
    
    Uses multiple strategies:
    1. LLM-based query expansion
    2. Domain-specific term mapping
    3. Synonym expansion
    4. Academic phrase normalization
    """
    
    def __init__(
        self,
        ollama_model: str = "deepseek-coder-v2",
        ollama_base_url: str = "http://localhost:11434",
        use_llm_expansion: bool = True,
        use_domain_mapping: bool = True,
        use_synonyms: bool = True
    ):
        """
        Initialize query rewriter.
        
        Args:
            ollama_model: Ollama model for LLM-based expansion
            ollama_base_url: Ollama API base URL
            use_llm_expansion: Whether to use LLM for query expansion
            use_domain_mapping: Whether to use domain-specific term mapping
            use_synonyms: Whether to use synonym expansion
        """
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.use_llm_expansion = use_llm_expansion
        self.use_domain_mapping = use_domain_mapping
        self.use_synonyms = use_synonyms
        
        # Initialize domain-specific term mappings
        self.domain_mappings = self._initialize_domain_mappings()
        
        # Initialize synonym mappings
        self.synonym_mappings = self._initialize_synonyms()
        
        # Academic phrase patterns
        self.academic_patterns = self._initialize_academic_patterns()
        
        logger.info("Query rewriter initialized")
    
    def _initialize_domain_mappings(self) -> Dict[str, List[str]]:
        """Initialize domain-specific term mappings for common research terms."""
        return {
            # Machine Learning / AI
            "neural network": ["deep learning", "artificial neural network", "multilayer perceptron", "neural model"],
            "transformer": ["attention mechanism", "self-attention", "multi-head attention", "BERT", "GPT"],
            "deep learning": ["neural network", "backpropagation", "gradient descent", "convolutional neural network"],
            "attention": ["self-attention", "multi-head attention", "attention mechanism", "attention weights"],
            "embeddings": ["word embeddings", "sentence embeddings", "vector representations", "latent representations"],
            
            # Computer Vision
            "cnn": ["convolutional neural network", "convolution", "pooling", "feature maps"],
            "object detection": ["YOLO", "R-CNN", "bounding box", "computer vision"],
            "image classification": ["computer vision", "feature extraction", "visual recognition"],
            
            # Natural Language Processing
            "nlp": ["natural language processing", "text processing", "language model", "computational linguistics"],
            "language model": ["LLM", "large language model", "GPT", "BERT", "transformer"],
            "tokenization": ["text preprocessing", "word segmentation", "subword units"],
            
            # Research Methods
            "evaluation": ["assessment", "validation", "testing", "performance measurement", "metrics"],
            "methodology": ["approach", "method", "technique", "framework", "algorithm"],
            "dataset": ["data", "corpus", "benchmark", "training data", "test set"],
            "baseline": ["benchmark", "comparison method", "reference model", "state-of-the-art"],
            
            # Performance Metrics
            "accuracy": ["precision", "recall", "F1-score", "performance metrics"],
            "loss function": ["objective function", "cost function", "optimization objective"],
            
            # Technical Terms
            "optimization": ["gradient descent", "SGD", "Adam", "learning rate", "training"],
            "regularization": ["dropout", "weight decay", "batch normalization", "overfitting prevention"],
            "hyperparameters": ["learning rate", "batch size", "model parameters", "tuning"],
        }
    
    def _initialize_synonyms(self) -> Dict[str, List[str]]:
        """Initialize general synonym mappings."""
        return {
            "improve": ["enhance", "boost", "increase", "optimize", "better"],
            "method": ["approach", "technique", "algorithm", "strategy", "procedure"],
            "result": ["outcome", "finding", "conclusion", "performance", "achievement"],
            "analyze": ["examine", "investigate", "study", "evaluate", "assess"],
            "propose": ["present", "introduce", "suggest", "offer", "develop"],
            "novel": ["new", "innovative", "original", "creative", "unique"],
            "effective": ["efficient", "successful", "powerful", "robust", "reliable"],
            "problem": ["issue", "challenge", "task", "question", "difficulty"],
            "solution": ["answer", "resolution", "approach", "method", "technique"],
            "comparison": ["comparison", "evaluation", "analysis", "assessment", "study"],
        }
    
    def _initialize_academic_patterns(self) -> Dict[str, str]:
        """Initialize academic phrase normalization patterns."""
        return {
            r"\bstate of the art\b": "state-of-the-art",
            r"\bstate-of-art\b": "state-of-the-art",
            r"\bcutting edge\b": "cutting-edge",
            r"\breal world\b": "real-world",
            r"\bdata set\b": "dataset",
            r"\bdata base\b": "database",
            r"\bmachine learning\b": "machine learning",
            r"\bartificial intelligence\b": "AI",
            r"\bnatural language processing\b": "NLP",
            r"\bcomputer vision\b": "computer vision",
        }
    
    def _normalize_query(self, query: str) -> str:
        """Normalize academic phrases in query."""
        normalized = query.lower()
        
        for pattern, replacement in self.academic_patterns.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def _expand_with_domain_terms(self, query: str) -> List[str]:
        """Expand query with domain-specific terms."""
        expanded_terms = []
        query_lower = query.lower()
        
        for term, expansions in self.domain_mappings.items():
            if term in query_lower:
                expanded_terms.extend(expansions)
        
        # Remove duplicates and terms already in query
        query_words = set(query_lower.split())
        unique_terms = []
        for term in expanded_terms:
            if term.lower() not in query_lower and not any(word in query_words for word in term.split()):
                unique_terms.append(term)
        
        return unique_terms
    
    def _expand_with_synonyms(self, query: str) -> List[str]:
        """Expand query with synonyms."""
        expanded_terms = []
        query_words = query.lower().split()
        
        for word in query_words:
            if word in self.synonym_mappings:
                expanded_terms.extend(self.synonym_mappings[word])
        
        # Remove duplicates and original words
        unique_terms = list(set(expanded_terms) - set(query_words))
        return unique_terms
    
    def _llm_expand_query(self, query: str) -> Optional[str]:
        """Use LLM to expand query with related terms."""
        if not self.use_llm_expansion:
            return None
        
        try:
            import requests
            
            prompt = f"""You are a research assistant helping to expand search queries for academic papers.

Original query: "{query}"

Please provide an expanded version of this query that includes:
1. Related technical terms and synonyms
2. Alternative phrasings common in academic literature
3. Specific methodologies or approaches related to the topic
4. Key terms that researchers in this field would use

Expanded query (keep it concise but comprehensive):"""

            data = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 150,
                }
            }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                expanded = result.get("response", "").strip()
                
                # Clean up the response
                expanded = re.sub(r'^["\']|["\']$', '', expanded)  # Remove quotes
                expanded = expanded.replace('\n', ' ')  # Remove newlines
                
                return expanded if expanded and expanded != query else None
            
        except Exception as e:
            logger.warning(f"LLM expansion failed: {e}")
        
        return None
    
    def rewrite(self, query: str, method: str = "hybrid") -> str:
        """
        Rewrite and expand a query.
        
        Args:
            query: Original query
            method: Expansion method ('llm', 'domain', 'synonym', 'hybrid')
            
        Returns:
            Expanded query string
        """
        logger.info(f"Rewriting query: {query}")
        
        # Start with normalized query
        normalized_query = self._normalize_query(query)
        expanded_terms = []
        
        # Apply different expansion methods
        if method in ["domain", "hybrid"] and self.use_domain_mapping:
            domain_terms = self._expand_with_domain_terms(query)
            expanded_terms.extend(domain_terms)
            logger.debug(f"Domain terms: {domain_terms}")
        
        if method in ["synonym", "hybrid"] and self.use_synonyms:
            synonym_terms = self._expand_with_synonyms(query)
            expanded_terms.extend(synonym_terms)
            logger.debug(f"Synonym terms: {synonym_terms}")
        
        if method in ["llm", "hybrid"] and self.use_llm_expansion:
            llm_expanded = self._llm_expand_query(query)
            if llm_expanded:
                logger.debug(f"LLM expanded: {llm_expanded}")
                return llm_expanded
        
        # Combine original query with expansion terms
        if expanded_terms:
            # Limit number of expansion terms to avoid overly long queries
            limited_terms = expanded_terms[:8]  # Max 8 additional terms
            expanded_query = f"{normalized_query} {' '.join(limited_terms)}"
            
            logger.info(f"Expanded query: {expanded_query}")
            return expanded_query
        
        return normalized_query
    
    def get_expansion_details(self, query: str) -> QueryExpansion:
        """
        Get detailed information about query expansion.
        
        Args:
            query: Original query
            
        Returns:
            QueryExpansion object with details
        """
        expanded_query = self.rewrite(query, method="hybrid")
        
        # Extract expansion terms
        original_words = set(query.lower().split())
        expanded_words = set(expanded_query.lower().split())
        expansion_terms = list(expanded_words - original_words)
        
        # Calculate confidence based on number of expansions
        confidence = min(len(expansion_terms) / 5, 1.0)  # Max confidence with 5+ terms
        
        return QueryExpansion(
            original_query=query,
            expanded_query=expanded_query,
            expansion_terms=expansion_terms,
            confidence=confidence,
            method="hybrid"
        )
    
    def batch_rewrite(self, queries: List[str]) -> List[str]:
        """Rewrite multiple queries in batch."""
        return [self.rewrite(query) for query in queries]
    
    def add_domain_mapping(self, term: str, expansions: List[str]):
        """Add a custom domain mapping."""
        self.domain_mappings[term.lower()] = expansions
        logger.info(f"Added domain mapping: {term} -> {expansions}")
    
    def add_synonym_mapping(self, word: str, synonyms: List[str]):
        """Add a custom synonym mapping."""
        self.synonym_mappings[word.lower()] = synonyms
        logger.info(f"Added synonym mapping: {word} -> {synonyms}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get query rewriter statistics."""
        return {
            "domain_mappings": len(self.domain_mappings),
            "synonym_mappings": len(self.synonym_mappings),
            "academic_patterns": len(self.academic_patterns),
            "llm_model": self.ollama_model,
            "features_enabled": {
                "llm_expansion": self.use_llm_expansion,
                "domain_mapping": self.use_domain_mapping,
                "synonyms": self.use_synonyms
            }
        }

# Utility functions
def create_academic_query_rewriter(
    ollama_model: str = "phi4-mini:3.8b",
    enable_llm: bool = True
) -> QueryRewriter:
    """
    Create a query rewriter optimized for academic research.
    
    Args:
        ollama_model: Ollama model name
        enable_llm: Whether to enable LLM-based expansion
        
    Returns:
        Configured QueryRewriter
    """
    return QueryRewriter(
        ollama_model=ollama_model,
        use_llm_expansion=enable_llm,
        use_domain_mapping=True,
        use_synonyms=True
    )

if __name__ == "__main__":
    # Example usage
    rewriter = create_academic_query_rewriter()
    
    # Test queries
    test_queries = [
        "transformer architecture attention mechanism",
        "deep learning CNN image classification",
        "NLP language model evaluation",
        "machine learning optimization methods"
    ]
    
    for query in test_queries:
        expansion = rewriter.get_expansion_details(query)
        print(f"\nOriginal: {expansion.original_query}")
        print(f"Expanded: {expansion.expanded_query}")
        print(f"Added terms: {expansion.expansion_terms}")
        print(f"Confidence: {expansion.confidence:.2f}")
