"""
Microservices Architecture: Storage Service
==========================================

Dedicated service for data storage and management
Handles PostgreSQL operations, document management, and data consistency
"""

import asyncio
import asyncpg
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import structlog
from fastapi import FastAPI, HTTPException
import uvicorn
import json
from datetime import datetime
import uuid

logger = structlog.get_logger()

# Models
class Paper(BaseModel):
    id: Optional[str] = None
    title: str
    authors: List[str]
    abstract: str
    content: str
    arxiv_id: Optional[str] = None
    published_date: Optional[datetime] = None
    metadata: Dict[str, Any] = {}

class DocumentChunk(BaseModel):
    id: Optional[str] = None
    content: str
    paper_id: str
    chunk_index: int
    metadata: Dict[str, Any] = {}

class QueryHistory(BaseModel):
    id: Optional[str] = None
    user_id: str
    query: str
    response: str
    confidence: float
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = {}

class StorageService:
    """Database Storage Service"""
    
    def __init__(self):
        self.pool = None
        self.db_config = {
            "host": "localhost",
            "port": 5432,
            "user": "research_user",
            "password": "research_password",
            "database": "research_copilot"
        }
    
    async def initialize(self):
        """Initialize database connection and schema"""
        logger.info("Initializing Storage Service")
        
        # Create connection pool
        self.pool = await asyncpg.create_pool(
            host=self.db_config["host"],
            port=self.db_config["port"],
            user=self.db_config["user"],
            password=self.db_config["password"],
            database=self.db_config["database"],
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # Initialize schema
        await self._initialize_schema()
        
        logger.info("Storage Service initialized successfully")
    
    async def _initialize_schema(self):
        """Create database tables if they don't exist"""
        async with self.pool.acquire() as conn:
            # Enable UUID extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
            
            # Papers table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    title TEXT NOT NULL,
                    authors JSONB NOT NULL,
                    abstract TEXT,
                    content TEXT,
                    arxiv_id TEXT UNIQUE,
                    published_date TIMESTAMP,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Document chunks table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    content TEXT NOT NULL,
                    paper_id UUID REFERENCES papers(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(paper_id, chunk_index)
                );
            """)
            
            # Query history table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    confidence FLOAT,
                    metadata JSONB DEFAULT '{}',
                    timestamp TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id ON papers(arxiv_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_title ON papers USING GIN(to_tsvector('english', title));")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_paper_id ON document_chunks(paper_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_query_history_user_id ON query_history(user_id);")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_query_history_timestamp ON query_history(timestamp);")
            
            logger.info("Database schema initialized")
    
    # Paper operations
    async def create_paper(self, paper: Paper) -> str:
        """Create a new paper"""
        async with self.pool.acquire() as conn:
            paper_id = await conn.fetchval("""
                INSERT INTO papers (title, authors, abstract, content, arxiv_id, published_date, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            """, paper.title, json.dumps(paper.authors), paper.abstract, 
                paper.content, paper.arxiv_id, paper.published_date, json.dumps(paper.metadata))
            
            logger.info("Paper created", paper_id=str(paper_id), title=paper.title[:50])
            return str(paper_id)
    
    async def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get paper by ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM papers WHERE id = $1", uuid.UUID(paper_id))
            
            if row:
                return Paper(
                    id=str(row['id']),
                    title=row['title'],
                    authors=json.loads(row['authors']),
                    abstract=row['abstract'],
                    content=row['content'],
                    arxiv_id=row['arxiv_id'],
                    published_date=row['published_date'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
            return None
    
    async def update_paper(self, paper_id: str, paper: Paper) -> bool:
        """Update paper"""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE papers 
                SET title = $2, authors = $3, abstract = $4, content = $5, 
                    arxiv_id = $6, published_date = $7, metadata = $8, updated_at = NOW()
                WHERE id = $1
            """, uuid.UUID(paper_id), paper.title, json.dumps(paper.authors), 
                paper.abstract, paper.content, paper.arxiv_id, 
                paper.published_date, json.dumps(paper.metadata))
            
            updated = result.split()[-1] == '1'
            if updated:
                logger.info("Paper updated", paper_id=paper_id)
            return updated
    
    async def delete_paper(self, paper_id: str) -> bool:
        """Delete paper and associated chunks"""
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM papers WHERE id = $1", uuid.UUID(paper_id))
            deleted = result.split()[-1] == '1'
            if deleted:
                logger.info("Paper deleted", paper_id=paper_id)
            return deleted
    
    async def list_papers(self, limit: int = 100, offset: int = 0) -> List[Paper]:
        """List papers with pagination"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM papers 
                ORDER BY created_at DESC 
                LIMIT $1 OFFSET $2
            """, limit, offset)
            
            return [
                Paper(
                    id=str(row['id']),
                    title=row['title'],
                    authors=json.loads(row['authors']),
                    abstract=row['abstract'],
                    content=row['content'],
                    arxiv_id=row['arxiv_id'],
                    published_date=row['published_date'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                for row in rows
            ]
    
    async def search_papers(self, query: str, limit: int = 20) -> List[Paper]:
        """Full-text search papers"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT *, ts_rank(to_tsvector('english', title || ' ' || COALESCE(abstract, '')), 
                                 plainto_tsquery('english', $1)) as rank
                FROM papers 
                WHERE to_tsvector('english', title || ' ' || COALESCE(abstract, '')) 
                      @@ plainto_tsquery('english', $1)
                ORDER BY rank DESC
                LIMIT $2
            """, query, limit)
            
            return [
                Paper(
                    id=str(row['id']),
                    title=row['title'],
                    authors=json.loads(row['authors']),
                    abstract=row['abstract'],
                    content=row['content'],
                    arxiv_id=row['arxiv_id'],
                    published_date=row['published_date'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                for row in rows
            ]
    
    # Document chunk operations
    async def create_document_chunks(self, paper_id: str, chunks: List[DocumentChunk]) -> List[str]:
        """Create document chunks for a paper"""
        chunk_ids = []
        async with self.pool.acquire() as conn:
            for chunk in chunks:
                chunk_id = await conn.fetchval("""
                    INSERT INTO document_chunks (content, paper_id, chunk_index, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                """, chunk.content, uuid.UUID(paper_id), chunk.chunk_index, json.dumps(chunk.metadata))
                chunk_ids.append(str(chunk_id))
        
        logger.info("Document chunks created", paper_id=paper_id, count=len(chunk_ids))
        return chunk_ids
    
    async def get_document_chunks(self, paper_id: str) -> List[DocumentChunk]:
        """Get all chunks for a paper"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM document_chunks 
                WHERE paper_id = $1 
                ORDER BY chunk_index
            """, uuid.UUID(paper_id))
            
            return [
                DocumentChunk(
                    id=str(row['id']),
                    content=row['content'],
                    paper_id=str(row['paper_id']),
                    chunk_index=row['chunk_index'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                for row in rows
            ]
    
    async def delete_document_chunks(self, paper_id: str) -> int:
        """Delete all chunks for a paper"""
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM document_chunks WHERE paper_id = $1", uuid.UUID(paper_id))
            count = int(result.split()[-1])
            logger.info("Document chunks deleted", paper_id=paper_id, count=count)
            return count
    
    # Query history operations
    async def create_query_history(self, query_history: QueryHistory) -> str:
        """Create query history entry"""
        async with self.pool.acquire() as conn:
            history_id = await conn.fetchval("""
                INSERT INTO query_history (user_id, query, response, confidence, metadata)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """, query_history.user_id, query_history.query, query_history.response,
                query_history.confidence, json.dumps(query_history.metadata))
            
            return str(history_id)
    
    async def get_query_history(self, user_id: str, limit: int = 50) -> List[QueryHistory]:
        """Get query history for user"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM query_history 
                WHERE user_id = $1 
                ORDER BY timestamp DESC 
                LIMIT $2
            """, user_id, limit)
            
            return [
                QueryHistory(
                    id=str(row['id']),
                    user_id=row['user_id'],
                    query=row['query'],
                    response=row['response'],
                    confidence=row['confidence'],
                    timestamp=row['timestamp'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                for row in rows
            ]
    
    # Analytics and metrics
    async def get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage metrics"""
        async with self.pool.acquire() as conn:
            paper_count = await conn.fetchval("SELECT COUNT(*) FROM papers")
            chunk_count = await conn.fetchval("SELECT COUNT(*) FROM document_chunks")
            query_count = await conn.fetchval("SELECT COUNT(*) FROM query_history")
            
            # Recent activity
            recent_papers = await conn.fetchval("""
                SELECT COUNT(*) FROM papers WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            recent_queries = await conn.fetchval("""
                SELECT COUNT(*) FROM query_history WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
            
            return {
                "papers": {
                    "total": paper_count,
                    "recent_24h": recent_papers
                },
                "chunks": {
                    "total": chunk_count
                },
                "queries": {
                    "total": query_count,
                    "recent_24h": recent_queries
                }
            }

# FastAPI app
app = FastAPI(title="Storage Service", version="1.0.0")
storage_service = StorageService()

@app.on_event("startup")
async def startup():
    await storage_service.initialize()

# Paper endpoints
@app.post("/papers", response_model=dict)
async def create_paper(paper: Paper):
    """Create a new paper"""
    try:
        paper_id = await storage_service.create_paper(paper)
        return {"id": paper_id, "message": "Paper created successfully"}
    except Exception as e:
        logger.error("Paper creation error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/{paper_id}", response_model=Paper)
async def get_paper(paper_id: str):
    """Get paper by ID"""
    paper = await storage_service.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper

@app.put("/papers/{paper_id}")
async def update_paper(paper_id: str, paper: Paper):
    """Update paper"""
    updated = await storage_service.update_paper(paper_id, paper)
    if not updated:
        raise HTTPException(status_code=404, detail="Paper not found")
    return {"message": "Paper updated successfully"}

@app.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):
    """Delete paper"""
    deleted = await storage_service.delete_paper(paper_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Paper not found")
    return {"message": "Paper deleted successfully"}

@app.get("/papers", response_model=List[Paper])
async def list_papers(limit: int = 100, offset: int = 0):
    """List papers"""
    return await storage_service.list_papers(limit, offset)

@app.get("/papers/search/{query}", response_model=List[Paper])
async def search_papers(query: str, limit: int = 20):
    """Search papers"""
    return await storage_service.search_papers(query, limit)

# Document chunk endpoints
@app.post("/papers/{paper_id}/chunks")
async def create_document_chunks(paper_id: str, chunks: List[DocumentChunk]):
    """Create document chunks"""
    try:
        chunk_ids = await storage_service.create_document_chunks(paper_id, chunks)
        return {"chunk_ids": chunk_ids, "message": "Chunks created successfully"}
    except Exception as e:
        logger.error("Chunk creation error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/{paper_id}/chunks", response_model=List[DocumentChunk])
async def get_document_chunks(paper_id: str):
    """Get document chunks for paper"""
    return await storage_service.get_document_chunks(paper_id)

@app.delete("/papers/{paper_id}/chunks")
async def delete_document_chunks(paper_id: str):
    """Delete document chunks for paper"""
    count = await storage_service.delete_document_chunks(paper_id)
    return {"deleted_count": count, "message": "Chunks deleted successfully"}

# Query history endpoints
@app.post("/query-history")
async def create_query_history(query_history: QueryHistory):
    """Create query history entry"""
    try:
        history_id = await storage_service.create_query_history(query_history)
        return {"id": history_id, "message": "Query history created successfully"}
    except Exception as e:
        logger.error("Query history creation error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query-history/{user_id}", response_model=List[QueryHistory])
async def get_query_history(user_id: str, limit: int = 50):
    """Get query history for user"""
    return await storage_service.get_query_history(user_id, limit)

@app.get("/health")
async def health_check():
    """Health check"""
    try:
        async with storage_service.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

@app.get("/metrics")
async def metrics():
    """Storage metrics"""
    try:
        metrics = await storage_service.get_storage_metrics()
        return {**metrics, "service": "storage"}
    except Exception as e:
        logger.error("Metrics error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicoru.run(app, host="0.0.0.0", port=8004)
