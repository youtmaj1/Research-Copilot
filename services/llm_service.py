"""
Microservices Architecture: LLM Service
=======================================

Dedicated service for Large Language Model interactions
Handles model management, prompt optimization, and response generation
"""

import asyncio
from typing import Dict, Optional, AsyncGenerator
from pydantic import BaseModel
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import httpx
from circuitbreaker import CircuitBreaker
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import time
from datetime import datetime
import aioredis

logger = structlog.get_logger()

# Models
class GenerationRequest(BaseModel):
    prompt: str
    model: str = "phi4-mini:3.8b"
    temperature: float = 0.3
    max_tokens: int = 500
    stream: bool = False
    system_prompt: Optional[str] = None

class GenerationResponse(BaseModel):
    response: str
    model: str
    tokens_used: int
    generation_time: float
    cached: bool = False

class LLMService:
    """Large Language Model Service"""
    
    def __init__(self):
        self.ollama_client = httpx.AsyncClient(timeout=60.0)
        self.ollama_base_url = "http://localhost:11434"
        self.redis = None
        
        # Circuit breaker for Ollama
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=Exception
        )
    
    async def initialize(self):
        """Initialize service"""
        logger.info("Initializing LLM Service")
        
        # Initialize Redis for caching
        self.redis = await aioredis.from_url("redis://localhost:6379")
        
        # Test Ollama connection
        await self.health_check()
        
        logger.info("LLM Service initialized successfully")
    
    async def health_check(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = await self.ollama_client.get(f"{self.ollama_base_url}/api/tags")
            response.raise_for_status()
            models = response.json()
            logger.info("Ollama health check passed", models=len(models.get("models", [])))
            return True
        except Exception as e:
            logger.error("Ollama health check failed", error=str(e))
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def generate_completion(self, request: GenerationRequest) -> GenerationResponse:
        """Generate completion with caching and retry logic"""
        start_time = time.time()
        
        # Check cache for non-streaming requests
        if not request.stream:
            cache_key = self._generate_cache_key(request)
            cached = await self.redis.get(cache_key)
            if cached:
                logger.info("LLM cache hit", model=request.model)
                data = json.loads(cached)
                data["cached"] = True
                data["generation_time"] = time.time() - start_time
                return GenerationResponse(**data)
        
        # Prepare prompt
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}"
        
        # Prepare Ollama request
        ollama_request = {
            "model": request.model,
            "prompt": full_prompt,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "stop": ["Human:", "User:", "Assistant:"]
            }
        }
        
        try:
            # Use circuit breaker
            response = await self._call_ollama_with_circuit_breaker(ollama_request)
            
            if request.stream:
                # Return streaming response generator
                return self._handle_streaming_response(response, request.model, start_time)
            else:
                # Handle non-streaming response
                result = response.json()
                
                response_obj = GenerationResponse(
                    response=result.get("response", ""),
                    model=request.model,
                    tokens_used=self._estimate_tokens(result.get("response", "")),
                    generation_time=time.time() - start_time
                )
                
                # Cache the response
                if not request.stream:
                    await self._cache_response(request, response_obj)
                
                return response_obj
                
        except httpx.TimeoutException:
            logger.error("LLM timeout", model=request.model, prompt_length=len(request.prompt))
            raise HTTPException(status_code=504, detail="LLM request timeout")
        except httpx.HTTPStatusError as e:
            logger.error("LLM HTTP error", status_code=e.response.status_code, model=request.model)
            raise HTTPException(status_code=502, detail=f"LLM service error: {e.response.status_code}")
        except Exception as e:
            logger.error("LLM unexpected error", error=str(e), model=request.model)
            raise HTTPException(status_code=500, detail="LLM service internal error")
    
    async def _call_ollama_with_circuit_breaker(self, request_data: Dict) -> httpx.Response:
        """Call Ollama with circuit breaker protection"""
        @self.circuit_breaker
        async def _call():
            response = await self.ollama_client.post(
                f"{self.ollama_base_url}/api/generate",
                json=request_data
            )
            response.raise_for_status()
            return response
        
        return await _call()
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        start_time = time.time()
        
        # Prepare prompt
        full_prompt = request.prompt
        if request.system_prompt:
            full_prompt = f"System: {request.system_prompt}\n\nUser: {request.prompt}"
        
        ollama_request = {
            "model": request.model,
            "prompt": full_prompt,
            "stream": True,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens
            }
        }
        
        try:
            response = await self._call_ollama_with_circuit_breaker(ollama_request)
            
            full_response = ""
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "response" in data and data["response"]:
                            chunk = data["response"]
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk, 'model': request.model})}\n\n"
                        
                        if data.get("done"):
                            # Send final metadata
                            metadata = {
                                "done": True,
                                "model": request.model,
                                "generation_time": time.time() - start_time,
                                "tokens_used": self._estimate_tokens(full_response)
                            }
                            yield f"data: {json.dumps(metadata)}\n\n"
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error("Streaming generation error", error=str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    def _generate_cache_key(self, request: GenerationRequest) -> str:
        """Generate cache key for request"""
        import hashlib
        key_data = f"{request.prompt}:{request.model}:{request.temperature}:{request.max_tokens}:{request.system_prompt}"
        return f"llm_cache:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    async def _cache_response(self, request: GenerationRequest, response: GenerationResponse):
        """Cache LLM response"""
        try:
            cache_key = self._generate_cache_key(request)
            cache_data = response.dict()
            cache_data.pop("generation_time", None)  # Don't cache timing info
            cache_data.pop("cached", None)
            
            # Cache for 1 hour
            await self.redis.setex(cache_key, 3600, json.dumps(cache_data))
            logger.info("Cached LLM response", model=request.model)
        except Exception as e:
            logger.error("Failed to cache LLM response", error=str(e))
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        return len(text.split()) * 4 // 3  # Rough approximation
    
    async def list_models(self) -> Dict:
        """List available models"""
        try:
            response = await self.ollama_client.get(f"{self.ollama_base_url}/api/tags")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to list models")
    
    async def model_info(self, model_name: str) -> Dict:
        """Get model information"""
        try:
            response = await self.ollama_client.post(
                f"{self.ollama_base_url}/api/show",
                json={"name": model_name}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get model info", model=model_name, error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to get info for model: {model_name}")

# FastAPI app
app = FastAPI(title="LLM Service", version="1.0.0")
llm_service = LLMService()

@app.on_event("startup")
async def startup():
    await llm_service.initialize()

@app.post("/generate", response_model=GenerationResponse)
async def generate_endpoint(request: GenerationRequest):
    """Generate completion"""
    if request.stream:
        # Return streaming response
        return StreamingResponse(
            llm_service.generate_stream(request),
            media_type="text/event-stream"
        )
    else:
        return await llm_service.generate_completion(request)

@app.post("/generate/stream")
async def generate_stream_endpoint(request: GenerationRequest):
    """Generate streaming completion"""
    return StreamingResponse(
        llm_service.generate_stream(request),
        media_type="text/event-stream"
    )

@app.get("/models")
async def list_models():
    """List available models"""
    return await llm_service.list_models()

@app.get("/models/{model_name}")
async def model_info(model_name: str):
    """Get model information"""
    return await llm_service.model_info(model_name)

@app.get("/health")
async def health_check():
    """Health check"""
    is_healthy = await llm_service.health_check()
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "ollama_available": is_healthy,
        "timestamp": datetime.utcnow()
    }

@app.get("/metrics")
async def metrics():
    """Service metrics"""
    return {
        "circuit_breaker_state": str(llm_service.circuit_breaker.current_state),
        "circuit_breaker_failures": llm_service.circuit_breaker.failure_count,
        "service": "llm",
        "timestamp": datetime.utcnow()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
