"""
Microservices Architecture: RAG Service
======================================

Dedicated service for Retrieval-Augmented Generation
Handles document retrieval, embedding search, and context building
"""

import asyncio
import asyncpg
import numpy as np
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
import structlog
from fastapi import FastAPI, HTTPException
import uvicorn
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import aioredis
import json

logger = structlog.get_logger()

# Models
class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 10
    paper_ids: Optional[List[str]] = None
    min_score: float = 0.5

class DocumentChunk(BaseModel):
    id: str
    content: str
    paper_id: str
    title: str
    authors: List[str]
    score: float
    metadata: Dict

class RetrievalResponse(BaseModel):
    chunks: List[DocumentChunk]
    total_found: int
    query_time: float

class RAGService:
    """Retrieval-Augmented Generation Service"""
    
    def __init__(self):
        self.embedding_model = None
        self.faiss_index = None
        self.chunk_metadata = {}
        self.pg_pool = None
        self.redis = None
    
    async def initialize(self):
        """Initialize service components"""
        logger.info("Initializing RAG Service")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize PostgreSQL connection
        self.pg_pool = await asyncpg.create_pool(
            "postgresql://user:password@localhost/research_copilot",
            min_size=2,
            max_size=10
        )
        
        # Initialize Redis
        self.redis = await aioredis.from_url("redis://localhost:6379")
        
        # Load FAISS index
        await self.load_vector_index()
        
        logger.info("RAG Service initialized successfully")
    
    async def load_vector_index(self):
        """Load or create FAISS vector index"""
        try:
            # Try to load existing index
            self.faiss_index = faiss.read_index("vector_index.faiss")
            with open("chunk_metadata.pkl", "rb") as f:
                self.chunk_metadata = pickle.load(f)
            logger.info("Loaded existing vector index", chunks=len(self.chunk_metadata))
        except FileNotFoundError:
            logger.info("No existing index found, will create new one")
            await self.build_vector_index()
    
    async def build_vector_index(self):
        """Build vector index from database"""
        logger.info("Building vector index from database")
        
        # Fetch all document chunks from database
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT c.id, c.content, c.paper_id, p.title, p.authors, c.metadata
                FROM document_chunks c
                JOIN papers p ON c.paper_id = p.id
            """)
        
        if not rows:
            logger.warning("No document chunks found in database")
            return
        
        # Generate embeddings
        texts = [row['content'] for row in rows]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.chunk_metadata = {
            i: {
                'id': row['id'],
                'content': row['content'],
                'paper_id': row['paper_id'],
                'title': row['title'],
                'authors': row['authors'],
                'metadata': row['metadata']
            }
            for i, row in enumerate(rows)
        }
        
        # Save index and metadata
        faiss.write_index(self.faiss_index, "vector_index.faiss")
        with open("chunk_metadata.pkl", "wb") as f:
            pickle.dump(self.chunk_metadata, f)
        
        logger.info("Vector index built successfully", 
                   chunks=len(self.chunk_metadata),
                   dimension=dimension)
    
    async def retrieve_documents(
        self, 
        query: str, 
        top_k: int = 10,
        paper_ids: Optional[List[str]] = None,
        min_score: float = 0.5
    ) -> List[DocumentChunk]:
        """Retrieve relevant documents using vector similarity"""
        
        # Check cache first
        cache_key = f"retrieval:{hash(query)}:{top_k}:{':'.join(paper_ids or [])}"
        cached = await self.redis.get(cache_key)
        if cached:
            logger.info("Retrieved from cache", query=query[:50])
            return [DocumentChunk(**chunk) for chunk in json.loads(cached)]
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search vector index
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            min(top_k * 2, len(self.chunk_metadata))  # Get more to filter
        )
        
        # Filter and format results
        chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if score < min_score:
                continue
                
            metadata = self.chunk_metadata[idx]
            
            # Filter by paper_ids if specified
            if paper_ids and metadata['paper_id'] not in paper_ids:
                continue
            
            chunk = DocumentChunk(
                id=metadata['id'],
                content=metadata['content'],
                paper_id=metadata['paper_id'],
                title=metadata['title'],
                authors=metadata['authors'],
                score=float(score),
                metadata=metadata['metadata']
            )
            chunks.append(chunk)
            
            if len(chunks) >= top_k:
                break
        
        # Cache results
        await self.redis.setex(
            cache_key, 
            3600,  # 1 hour
            json.dumps([chunk.dict() for chunk in chunks])
        )
        
        logger.info("Retrieved documents", 
                   query=query[:50],
                   found=len(chunks),
                   top_score=chunks[0].score if chunks else 0)
        
        return chunks
    
    async def add_documents(self, paper_id: str, chunks: List[Dict]):
        """Add new documents to the index"""
        logger.info("Adding documents to index", paper_id=paper_id, chunks=len(chunks))
        
        # Store in database first
        async with self.pg_pool.acquire() as conn:
            for chunk in chunks:
                await conn.execute("""
                    INSERT INTO document_chunks (id, content, paper_id, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata
                """, chunk['id'], chunk['content'], paper_id, json.dumps(chunk.get('metadata', {})))
        
        # Rebuild index (in production, you'd do incremental updates)
        await self.build_vector_index()
        
        # Clear cache
        keys = await self.redis.keys("retrieval:*")
        if keys:
            await self.redis.delete(*keys)
        
        logger.info("Documents added successfully", paper_id=paper_id)

# FastAPI app
app = FastAPI(title="RAG Service", version="1.0.0")
rag = RAGService()

@app.on_event("startup")
async def startup():
    await rag.initialize()

@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_endpoint(request: RetrievalRequest):
    """Retrieve relevant documents"""
    import time
    start_time = time.time()
    
    try:
        chunks = await rag.retrieve_documents(
            request.query,
            request.top_k,
            request.paper_ids,
            request.min_score
        )
        
        return RetrievalResponse(
            chunks=chunks,
            total_found=len(chunks),
            query_time=time.time() - start_time
        )
    except Exception as e:
        logger.error("Retrieval error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-documents")
async def add_documents_endpoint(paper_id: str, chunks: List[Dict]):
    """Add new documents to the index"""
    try:
        await rag.add_documents(paper_id, chunks)
        return {"message": "Documents added successfully"}
    except Exception as e:
        logger.error("Add documents error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "index_size": len(rag.chunk_metadata) if rag.chunk_metadata else 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
