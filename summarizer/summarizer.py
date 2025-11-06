"""
Research Paper Summarizer

LLM-based summarization system with local Ollama support and cloud API fallback.
Implements map-reduce pattern for long documents and structured output.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

try:
    import requests
except ImportError:
    requests = None

try:
    from langchain_ollama import OllamaLLM
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    try:
        # Fallback to older imports
        from langchain.llms import Ollama as OllamaLLM
        from langchain.chat_models import ChatOpenAI, ChatAnthropic
        from langchain.schema import HumanMessage, SystemMessage  
        from langchain.callbacks.base import BaseCallbackHandler
    except ImportError:
        OllamaLLM = None
        ChatOpenAI = None
        ChatAnthropic = None
        HumanMessage = None
        SystemMessage = None
        BaseCallbackHandler = None

from .chunker import TextChunk

logger = logging.getLogger(__name__)


class SummaryType(Enum):
    """Types of summaries to generate."""
    EXTRACTIVE = "extractive"      # Key sentences extraction
    ABSTRACTIVE = "abstractive"    # Generated summary
    STRUCTURED = "structured"      # Structured output with sections
    BULLET_POINTS = "bullet_points" # Bulleted key points


class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"


@dataclass
class SummaryConfig:
    """Configuration for summarization."""
    summary_type: SummaryType = SummaryType.STRUCTURED
    max_summary_tokens: int = 500
    temperature: float = 0.3
    include_key_points: bool = True
    include_methodology: bool = True
    include_results: bool = True
    include_conclusions: bool = True
    language: str = "english"


@dataclass
class SummaryResult:
    """Result of summarization process."""
    summary: str
    key_points: List[str]
    methodology: str
    results: str
    conclusions: str
    confidence_score: float
    processing_time: float
    token_usage: Dict[str, int]
    model_used: str
    chunks_processed: int


class ProgressCallback(BaseCallbackHandler if BaseCallbackHandler else object):
    """Callback for tracking summarization progress."""
    
    def __init__(self):
        self.steps_completed = 0
        self.total_steps = 0
        self.current_step = ""
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts."""
        self.current_step = f"Processing prompt {self.steps_completed + 1}/{self.total_steps}"
        logger.info(self.current_step)
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when LLM ends."""
        self.steps_completed += 1


class ResearchPaperSummarizer:
    """
    LLM-based research paper summarizer with multiple provider support.
    """
    
    SUMMARY_PROMPTS = {
        SummaryType.EXTRACTIVE: """
Extract the most important sentences from this research paper text that capture the key findings, methodology, and conclusions. Return only the selected sentences, preserving their original wording.

Text: {text}

Important sentences:""",
        
        SummaryType.ABSTRACTIVE: """
Write a concise summary of this research paper text. Focus on the main contributions, methodology, key findings, and conclusions. Keep it under {max_tokens} words.

Text: {text}

Summary:""",
        
        SummaryType.STRUCTURED: """
Analyze this research paper text and provide a structured summary with the following sections:

1. **Key Points**: 3-5 bullet points of main contributions
2. **Methodology**: Brief description of the approach used
3. **Results**: Key findings and outcomes
4. **Conclusions**: Main conclusions and implications

Text: {text}

Structured Summary:""",
        
        SummaryType.BULLET_POINTS: """
Create a bullet-point summary of this research paper text. Focus on:
- Main research question/problem
- Key methodology points
- Important findings
- Major conclusions

Text: {text}

Bullet Points:"""
    }
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OLLAMA,
        model_name: str = "llama2",
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        config: Optional[SummaryConfig] = None
    ):
        """
        Initialize the summarizer.
        
        Args:
            provider: LLM provider to use
            model_name: Model name for the provider
            api_key: API key for cloud providers
            base_url: Base URL for Ollama
            config: Summary configuration
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.config = config or SummaryConfig()
        
        # Initialize LLM
        self.llm = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize the LLM based on provider."""
        try:
            if self.provider == LLMProvider.OLLAMA:
                if OllamaLLM is None:
                    raise ImportError("LangChain Ollama not installed")
                
                self.llm = OllamaLLM(
                    model=self.model_name,
                    base_url=self.base_url,
                    temperature=self.config.temperature
                )
                
                # Test Ollama connection
                if not self._test_ollama_connection():
                    logger.warning("Ollama not available, will try fallback")
                    self._init_fallback_llm()
                    
            elif self.provider == LLMProvider.OPENAI:
                if ChatOpenAI is None:
                    raise ImportError("LangChain OpenAI not installed")
                
                self.llm = ChatOpenAI(
                    model_name=self.model_name,
                    openai_api_key=self.api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_summary_tokens
                )
                
            elif self.provider == LLMProvider.ANTHROPIC:
                if ChatAnthropic is None:
                    raise ImportError("LangChain Anthropic not installed")
                
                self.llm = ChatAnthropic(
                    model=self.model_name,
                    anthropic_api_key=self.api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_summary_tokens
                )
            
            logger.info(f"Initialized {self.provider.value} with model {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider.value}: {e}")
            self._init_fallback_llm()
    
    def _init_fallback_llm(self):
        """Initialize fallback LLM (simple API-based approach)."""
        logger.info("Using fallback summarization approach")
        self.llm = None  # Will use direct API calls
    
    def _test_ollama_connection(self) -> bool:
        """Test if Ollama is available."""
        try:
            if requests is None:
                return False
            
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def summarize_chunks(
        self,
        chunks: List[TextChunk],
        progress_callback: Optional[ProgressCallback] = None
    ) -> SummaryResult:
        """
        Summarize a list of text chunks using map-reduce approach.
        
        Args:
            chunks: List of text chunks to summarize
            progress_callback: Optional progress callback
            
        Returns:
            Summary result with structured output
        """
        if not chunks:
            raise ValueError("No chunks provided for summarization")
        
        start_time = time.time()
        logger.info(f"Starting summarization of {len(chunks)} chunks")
        
        if progress_callback:
            progress_callback.total_steps = len(chunks) + 1  # +1 for final reduction
        
        # Step 1: Map - Summarize individual chunks
        chunk_summaries = []
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_summary = self._summarize_single_chunk(chunk)
                chunk_summaries.append(chunk_summary)
                
                # Update token usage (approximate)
                token_usage["prompt_tokens"] += chunk.metadata.token_count
                token_usage["completion_tokens"] += len(chunk_summary.split()) * 1.3  # Rough estimate
                
                if progress_callback:
                    progress_callback.on_llm_end(None)
                    
            except Exception as e:
                logger.error(f"Failed to summarize chunk {i}: {e}")
                continue
        
        if not chunk_summaries:
            raise RuntimeError("Failed to summarize any chunks")
        
        # Step 2: Reduce - Combine chunk summaries into final summary
        final_summary = self._reduce_summaries(chunk_summaries)
        
        # Extract structured components
        key_points, methodology, results, conclusions = self._extract_structured_components(
            final_summary, chunks
        )
        
        processing_time = time.time() - start_time
        token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
        
        logger.info(f"Summarization completed in {processing_time:.2f}s")
        
        return SummaryResult(
            summary=final_summary,
            key_points=key_points,
            methodology=methodology,
            results=results,
            conclusions=conclusions,
            confidence_score=self._calculate_confidence_score(chunks, final_summary),
            processing_time=processing_time,
            token_usage=token_usage,
            model_used=f"{self.provider.value}:{self.model_name}",
            chunks_processed=len(chunks)
        )
    
    def _summarize_single_chunk(self, chunk: TextChunk) -> str:
        """Summarize a single text chunk."""
        prompt = self.SUMMARY_PROMPTS[self.config.summary_type].format(
            text=chunk.content,
            max_tokens=self.config.max_summary_tokens // 4  # Smaller for individual chunks
        )
        
        return self._generate_text(prompt)
    
    def _reduce_summaries(self, summaries: List[str]) -> str:
        """Combine individual summaries into a final summary."""
        combined_text = "\n\n".join(summaries)
        
        reduce_prompt = f"""
Combine these individual section summaries into a cohesive final summary of the research paper. 
Eliminate redundancy and create a flowing narrative that covers:
1. Main research contribution
2. Methodology used
3. Key findings
4. Conclusions and implications

Section summaries:
{combined_text}

Final Summary:"""
        
        return self._generate_text(reduce_prompt)
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the configured LLM."""
        try:
            if self.llm is not None:
                # Use LangChain LLM
                if hasattr(self.llm, 'invoke'):
                    # New LangChain API
                    response = self.llm.invoke(prompt)
                else:
                    # Old LangChain API
                    response = self.llm(prompt)
                
                return response.strip() if isinstance(response, str) else str(response).strip()
            else:
                # Use fallback approach
                return self._generate_text_fallback(prompt)
                
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return self._generate_text_fallback(prompt)
    
    def _generate_text_fallback(self, prompt: str) -> str:
        """Fallback text generation using direct API calls."""
        if self.provider == LLMProvider.OLLAMA:
            return self._call_ollama_api(prompt)
        else:
            # For other providers, return a simple extractive summary
            return self._extractive_fallback(prompt)
    
    def _call_ollama_api(self, prompt: str) -> str:
        """Call Ollama API directly."""
        try:
            if requests is None:
                raise ImportError("requests not available")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_summary_tokens
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            return ""
    
    def _extractive_fallback(self, prompt: str) -> str:
        """Simple extractive summarization fallback."""
        # Extract the text from the prompt
        text_start = prompt.find("Text: ") + 6
        text_end = prompt.find("\n\n", text_start)
        if text_end == -1:
            text_end = len(prompt)
        
        text = prompt[text_start:text_end]
        
        # Simple sentence extraction based on length and position
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        
        # Select first few sentences up to token limit
        selected = []
        total_words = 0
        max_words = self.config.max_summary_tokens
        
        for sentence in sentences:
            words = len(sentence.split())
            if total_words + words <= max_words:
                selected.append(sentence)
                total_words += words
            else:
                break
        
        return '. '.join(selected) + '.' if selected else "Summary not available."
    
    def _extract_structured_components(
        self,
        summary: str,
        chunks: List[TextChunk]
    ) -> tuple[List[str], str, str, str]:
        """Extract structured components from summary and chunks."""
        # Simple pattern-based extraction
        key_points = []
        methodology = ""
        results = ""
        conclusions = ""
        
        # Extract key points (look for bullet points or numbered lists)
        import re
        bullet_pattern = r'[•\-\*]\s*(.+?)(?=\n|$)'
        number_pattern = r'\d+\.\s*(.+?)(?=\n|$)'
        
        key_points.extend(re.findall(bullet_pattern, summary))
        key_points.extend(re.findall(number_pattern, summary))
        
        # Extract methodology from chunks or summary
        methodology_chunks = [c for c in chunks 
                            if any(term in c.metadata.section_title.lower() 
                                  for term in ['method', 'approach', 'implementation'])]
        if methodology_chunks:
            methodology = methodology_chunks[0].content[:500] + "..."
        
        # Extract results
        results_chunks = [c for c in chunks 
                         if any(term in c.metadata.section_title.lower() 
                               for term in ['result', 'finding', 'experiment', 'evaluation'])]
        if results_chunks:
            results = results_chunks[0].content[:500] + "..."
        
        # Extract conclusions
        conclusion_chunks = [c for c in chunks 
                           if any(term in c.metadata.section_title.lower() 
                                 for term in ['conclusion', 'discussion', 'future'])]
        if conclusion_chunks:
            conclusions = conclusion_chunks[0].content[:500] + "..."
        
        return key_points[:5], methodology, results, conclusions
    
    def _calculate_confidence_score(self, chunks: List[TextChunk], summary: str) -> float:
        """Calculate confidence score for the summary."""
        # Simple heuristic based on coverage and coherence
        total_content_length = sum(len(chunk.content) for chunk in chunks)
        summary_length = len(summary)
        
        # Coverage score (how much was summarized)
        coverage_score = min(1.0, summary_length / (total_content_length * 0.1))
        
        # Coherence score (based on summary structure)
        coherence_score = 0.8  # Placeholder
        if len(summary.split('.')) > 3:  # Has multiple sentences
            coherence_score += 0.1
        if any(word in summary.lower() for word in ['method', 'result', 'conclusion']):
            coherence_score += 0.1
        
        return min(1.0, (coverage_score + coherence_score) / 2)
    
    def summarize_paper_from_file(self, pdf_path: str) -> SummaryResult:
        """
        Convenience method to summarize a paper directly from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Summary result
        """
        from .pdf_extractor import extract_pdf_structure
        from .chunker import chunk_paper_structure
        
        # Extract and chunk the paper
        paper_structure = extract_pdf_structure(pdf_path)
        chunks = chunk_paper_structure(paper_structure)
        
        return self.summarize_chunks(chunks)


def summarize_paper(
    pdf_path: str,
    provider: LLMProvider = LLMProvider.OLLAMA,
    model_name: str = "llama2",
    summary_type: SummaryType = SummaryType.STRUCTURED,
    api_key: Optional[str] = None
) -> SummaryResult:
    """
    Convenience function to summarize a research paper.
    
    Args:
        pdf_path: Path to the PDF file
        provider: LLM provider to use
        model_name: Model name
        summary_type: Type of summary to generate
        api_key: API key for cloud providers
        
    Returns:
        Summary result
    """
    config = SummaryConfig(summary_type=summary_type)
    
    summarizer = ResearchPaperSummarizer(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        config=config
    )
    
    return summarizer.summarize_paper_from_file(pdf_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python summarizer.py <pdf_path> [provider] [model]")
        print("Providers: ollama, openai, anthropic")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    provider_name = sys.argv[2] if len(sys.argv) > 2 else "ollama"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "llama2"
    
    # Map provider name
    provider_map = {
        'ollama': LLMProvider.OLLAMA,
        'openai': LLMProvider.OPENAI,
        'anthropic': LLMProvider.ANTHROPIC
    }
    provider = provider_map.get(provider_name.lower(), LLMProvider.OLLAMA)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        result = summarize_paper(
            pdf_path=pdf_path,
            provider=provider,
            model_name=model_name,
            summary_type=SummaryType.STRUCTURED
        )
        
        print(f"Paper: {pdf_path}")
        print(f"Model: {result.model_used}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Chunks processed: {result.chunks_processed}")
        
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(result.summary)
        
        if result.key_points:
            print("\n" + "="*50)
            print("KEY POINTS")
            print("="*50)
            for i, point in enumerate(result.key_points, 1):
                print(f"{i}. {point}")
        
        if result.methodology:
            print("\n" + "="*50)
            print("METHODOLOGY")
            print("="*50)
            print(result.methodology)
        
        if result.results:
            print("\n" + "="*50)
            print("RESULTS")
            print("="*50)
            print(result.results)
        
        if result.conclusions:
            print("\n" + "="*50)
            print("CONCLUSIONS")
            print("="*50)
            print(result.conclusions)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
