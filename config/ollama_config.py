"""
Ollama Configuration Manager
============================

Configuration manager for using local Ollama models with the Research Copilot system.
Specifically configured for DeepSeek-Coder-V2:16B model.
"""

import os
import requests
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

class OllamaConfigManager:
    """Manager for Ollama model configuration and communication"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "phi4-mini:3.8b"):
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                
                if self.model in available_models:
                    self.logger.info(f"✅ Connected to Ollama server. Model {self.model} is available.")
                    return True
                else:
                    self.logger.warning(f"⚠️ Model {self.model} not found. Available: {available_models}")
                    return False
            else:
                self.logger.error(f"❌ Failed to connect to Ollama server: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Ollama connection error: {e}")
            return False
    
    def generate_completion(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.3) -> Dict[str, Any]:
        """Generate completion using Ollama model"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": self.model,
                    "tokens_used": len(result.get("response", "").split())
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 2000, temperature: float = 0.3) -> Dict[str, Any]:
        """Generate chat completion using Ollama model"""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", {})
                return {
                    "success": True,
                    "response": message.get("content", ""),
                    "model": self.model,
                    "tokens_used": len(message.get("content", "").split())
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}

def configure_ollama_for_research_copilot():
    """Configure Ollama for Research Copilot system"""
    print("🦙 Configuring Ollama for Research Copilot...")
    
    # Initialize Ollama manager
    ollama = OllamaConfigManager()
    
    # Test basic functionality
    test_prompt = "Hello! Please confirm you're working correctly."
    result = ollama.generate_completion(test_prompt, max_tokens=100)
    
    if result["success"]:
        print(f"✅ Ollama DeepSeek-Coder-V2:16B is working!")
        print(f"📝 Test response: {result['response'][:100]}...")
        
        # Set environment variables for the system
        os.environ['OLLAMA_BASE_URL'] = ollama.base_url
        os.environ['OLLAMA_MODEL'] = ollama.model
        os.environ['LLM_PROVIDER'] = 'ollama'
        
        return ollama
    else:
        print(f"❌ Ollama test failed: {result['error']}")
        return None

if __name__ == "__main__":
    configure_ollama_for_research_copilot()
