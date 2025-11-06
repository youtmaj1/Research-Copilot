"""
Microservices Architecture: Embedding Service
============================================

Dedicated service for text embeddings and vector operations
Handles embedding generation, similarity calculations, and vector management
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional
from pydantic import BaseModel
import structlog
from fastapi import FastAPI, HTTPException
import uvicorn
from sentence_transformers import SentenceTransformer
import torch
import aioredis
import json
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor

logger = structlog.get_logger()

# Models
class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: str = "all-MiniLM-L6-v2"
    normalize: bool = True

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int
    processing_time: float
    cached_count: int = 0

class SimilarityRequest(BaseModel):
    query_embedding: List[float]
    candidate_embeddings: List[List[float]]
    top_k: int = 10

class SimilarityResponse(BaseModel):
    similarities: List[float]
    indices: List[int]
    processing_time: float

class EmbeddingService:
    """Text Embedding Service"""
    
    def __init__(self):
        self.models = {}
        self.redis = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    async def initialize(self):
        """Initialize service"""
        logger.info("Initializing Embedding Service", device=self.device)
        
        # Initialize Redis for caching
        self.redis = await aioredis.from_url("redis://localhost:6379")
        
        # Load default models
        await self._load_model("all-MiniLM-L6-v2")
        await self._load_model("all-mpnet-base-v2")  # Higher quality model
        
        logger.info("Embedding Service initialized successfully", 
                   models=list(self.models.keys()))
    
    async def _load_model(self, model_name: str):
        """Load embedding model"""
        logger.info("Loading embedding model", model=model_name)
        
        try:
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self.executor,
                lambda: SentenceTransformer(model_name, device=self.device)
            )
            self.models[model_name] = model
            logger.info("Model loaded successfully", 
                       model=model_name, 
                       dimension=model.get_sentence_embedding_dimension())
        except Exception as e:
            logger.error("Failed to load model", model=model_name, error=str(e))
            raise
    
    async def generate_embeddings(
        self, 
        texts: List[str], 
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True
    ) -> EmbeddingResponse:
        """Generate embeddings for texts"""
        start_time = time.time()
        
        # Validate inputs
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(texts) > 100:
            raise HTTPException(status_code=400, detail="Too many texts (max 100)")
        
        # Load model if not available
        if model_name not in self.models:
            await self._load_model(model_name)
        
        model = self.models[model_name]
        
        # Check cache for each text
        cached_embeddings = {}
        texts_to_process = []
        text_indices = {}
        
        for i, text in enumerate(texts):
            cache_key = self._generate_cache_key(text, model_name, normalize)
            cached = await self.redis.get(cache_key)
            
            if cached:
                cached_embeddings[i] = json.loads(cached)
            else:
                texts_to_process.append(text)
                text_indices[len(texts_to_process) - 1] = i
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if texts_to_process:
            logger.info("Generating embeddings", 
                       model=model_name, 
                       texts=len(texts_to_process),
                       cached=len(cached_embeddings))
            
            # Generate embeddings in thread pool
            loop = asyncio.get_event_loop()
            embeddings_array = await loop.run_in_executor(
                self.executor,
                lambda: model.encode(texts_to_process, normalize=normalize)
            )
            
            new_embeddings = embeddings_array.tolist()
            
            # Cache new embeddings
            for i, embedding in enumerate(new_embeddings):
                original_index = text_indices[i]
                cache_key = self._generate_cache_key(texts[original_index], model_name, normalize)
                await self.redis.setex(cache_key, 3600, json.dumps(embedding))  # Cache for 1 hour
        
        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for i, embedding in cached_embeddings.items():
            all_embeddings[i] = embedding
        
        # Place new embeddings
        for i, embedding in enumerate(new_embeddings):
            original_index = text_indices[i]
            all_embeddings[original_index] = embedding
        
        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=model_name,
            dimension=len(all_embeddings[0]) if all_embeddings else 0,
            processing_time=time.time() - start_time,
            cached_count=len(cached_embeddings)
        )
    
    async def calculate_similarity(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 10
    ) -> SimilarityResponse:
        """Calculate cosine similarity between query and candidates"""
        start_time = time.time()
        
        if not candidate_embeddings:
            raise HTTPException(status_code=400, detail="No candidate embeddings provided")
        
        # Convert to numpy arrays
        query_vec = np.array(query_embedding, dtype=np.float32)
        candidate_vecs = np.array(candidate_embeddings, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        query_norm = query_vec / np.linalg.norm(query_vec)
        candidate_norms = candidate_vecs / np.linalg.norm(candidate_vecs, axis=1, keepdims=True)
        
        # Calculate similarities
        similarities = np.dot(candidate_norms, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        return SimilarityResponse(
            similarities=top_similarities.tolist(),
            indices=top_indices.tolist(),
            processing_time=time.time() - start_time
        )
    
    def _generate_cache_key(self, text: str, model: str, normalize: bool) -> str:
        """Generate cache key for embedding"""
        key_data = f"{text}:{model}:{normalize}"
        return f"embedding:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    async def get_model_info(self, model_name: str) -> Dict:
        """Get information about a model"""
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model = self.models[model_name]
        return {
            "name": model_name,
            "dimension": model.get_sentence_embedding_dimension(),
            "max_seq_length": getattr(model, 'max_seq_length', 'unknown'),
            "device": str(model.device)
        }
    
    async def list_models(self) -> Dict:
        """List available models"""
        return {
            "models": list(self.models.keys()),
            "device": self.device,
            "available_models": [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "sentence-transformers/all-roberta-large-v1"
            ]
        }

# FastAPI app
app = FastAPI(title="Embedding Service", version="1.0.0")
embedding_service = EmbeddingService()

@app.on_event("startup")
async def startup():
    await embedding_service.initialize()

@app.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings_endpoint(request: EmbeddingRequest):
    """Generate embeddings for texts"""
    try:
        return await embedding_service.generate_embeddings(
            request.texts,
            request.model,
            request.normalize
        )
    except Exception as e:
        logger.error("Embedding generation error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity_endpoint(request: SimilarityRequest):
    """Calculate similarity between embeddings"""
    try:
        return await embedding_service.calculate_similarity(
            request.query_embedding,
            request.candidate_embeddings,
            request.top_k
        )
    except Exception as e:
        logger.error("Similarity calculation error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    return await embedding_service.list_models()

@app.get("/models/{model_name}")
async def model_info(model_name: str):
    """Get model information"""
    return await embedding_service.get_model_info(model_name)

@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a new model"""
    try:
        await embedding_service._load_model(model_name)
        return {"message": f"Model {model_name} loaded successfully"}
    except Exception as e:
        logger.error("Model loading error", model=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "models_loaded": len(embedding_service.models),
        "device": embedding_service.device,
        "gpu_available": torch.cuda.is_available()
    }

@app.get("/metrics")
async def metrics():
    """Service metrics"""
    # Get Redis info
    redis_info = {}
    try:
        info = await embedding_service.redis.info()
        redis_info = {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory": info.get("used_memory_human", "unknown")
        }
    except:
        redis_info = {"status": "unavailable"}
    
    return {
        "models_loaded": len(embedding_service.models),
        "device": embedding_service.device,
        "gpu_available": torch.cuda.is_available(),
        "redis": redis_info,
        "service": "embedding"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
