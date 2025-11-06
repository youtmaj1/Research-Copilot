#!/usr/bin/env python3
"""
Simple Demo: Production Research Copilot
=======================================
Demonstrates working microservices architecture with:
- FastAPI async API
- PostgreSQL database 
- Redis caching
- Health checks
- Basic RAG functionality using local Ollama
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import asyncpg
import redis.asyncio as aioredis
import httpx
import json
import time
from datetime import datetime
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Copilot - Production Demo",
    description="Microservices-based Research Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global connections
redis_client = None
db_pool = None

@app.on_event("startup")
async def startup_event():
    global redis_client, db_pool
    
    # Connect to Redis
    try:
        redis_client = aioredis.from_url("redis://:research_redis_pass@localhost:6379")
        await redis_client.ping()
        logger.info("✅ Redis connected successfully")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
    
    # Connect to PostgreSQL
    try:
        db_pool = await asyncpg.create_pool(
            "postgresql://research_user:research_password@localhost:5432/research_copilot",
            min_size=1,
            max_size=10
        )
        logger.info("✅ PostgreSQL connected successfully")
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global redis_client, db_pool
    if redis_client:
        await redis_client.close()
    if db_pool:
        await db_pool.close()

@app.get("/health")
async def health_check():
    """System health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    # Test Redis
    try:
        await redis_client.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Test PostgreSQL
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            health_status["services"]["postgresql"] = "healthy"
    except Exception as e:
        health_status["services"]["postgresql"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Test Ollama
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                health_status["services"]["ollama"] = f"healthy ({len(models)} models)"
            else:
                health_status["services"]["ollama"] = "unhealthy"
    except Exception as e:
        health_status["services"]["ollama"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/api/v1/stats")
async def get_system_stats():
    """Get system statistics"""
    stats = {
        "timestamp": datetime.now().isoformat(),
        "uptime": "demo",
        "cache_stats": {},
        "database_stats": {}
    }
    
    # Redis stats
    try:
        info = await redis_client.info()
        stats["cache_stats"] = {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "total_commands_processed": info.get("total_commands_processed", 0)
        }
    except Exception as e:
        stats["cache_stats"]["error"] = str(e)
    
    # Database stats
    try:
        async with db_pool.acquire() as conn:
            papers_count = await conn.fetchval("SELECT COUNT(*) FROM papers")
            stats["database_stats"] = {
                "papers_count": papers_count,
                "status": "connected"
            }
    except Exception as e:
        stats["database_stats"]["error"] = str(e)
    
    return stats

@app.post("/api/v1/query")
async def process_query(request: Dict[str, Any]):
    """Process research query with caching"""
    query = request.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    start_time = time.time()
    cache_key = f"query:{hash(query)}"
    
    # Check cache first
    try:
        cached_result = await redis_client.get(cache_key)
        if cached_result:
            result = json.loads(cached_result)
            result["cached"] = True
            result["response_time"] = time.time() - start_time
            return result
    except Exception as e:
        logger.warning(f"Cache read failed: {e}")
    
    # Process with Ollama
    try:
        async with httpx.AsyncClient() as client:
            ollama_request = {
                "model": "deepseek-coder-v2:16b",
                "prompt": f"Research Query: {query}\n\nProvide a helpful research-focused response:",
                "stream": False
            }
            
            response = await client.post(
                "http://localhost:11434/api/generate",
                json=ollama_request,
                timeout=60.0
            )
            
            if response.status_code == 200:
                ollama_response = response.json()
                
                result = {
                    "query": query,
                    "response": ollama_response.get("response", "No response generated"),
                    "model": "deepseek-coder-v2:16b",
                    "cached": False,
                    "response_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache result
                try:
                    await redis_client.setex(
                        cache_key, 
                        3600,  # 1 hour cache
                        json.dumps(result, default=str)
                    )
                except Exception as e:
                    logger.warning(f"Cache write failed: {e}")
                
                return result
            else:
                raise HTTPException(status_code=500, detail="LLM service unavailable")
                
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Query processing failed: {e}")
        logger.error(f"Full traceback: {error_details}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
