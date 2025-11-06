"""
Summarization Pipeline

Main orchestrator that coordinates PDF extraction, chunking, summarization, 
and vector storage with progress tracking and error handling.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import uuid
from datetime import datetime

from .pdf_extractor import PDFExtractor, PaperStructure, extract_pdf_structure
from .chunker import ResearchPaperChunker, TextChunk, ChunkingStrategy, chunk_paper_structure
from .summarizer import ResearchPaperSummarizer, SummaryResult, SummaryConfig, LLMProvider, SummaryType
from .faiss_index import FaissVectorIndex, create_vector_index

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Processing status for pipeline operations."""
    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    SUMMARIZING = "summarizing"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Configuration for the summarization pipeline."""
    # PDF Extraction
    use_ocr: bool = True
    ocr_fallback_threshold: float = 0.1
    
    # Chunking
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    max_chunk_tokens: int = 1000
    overlap_tokens: int = 100
    min_chunk_tokens: int = 50
    
    # Summarization
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    model_name: str = "llama2"
    summary_type: SummaryType = SummaryType.STRUCTURED
    max_summary_tokens: int = 500
    temperature: float = 0.3
    api_key: Optional[str] = None
    base_url: str = "http://localhost:11434"
    
    # Vector Index
    embedding_model: str = "all-MiniLM-L6-v2"
    index_type: str = "flat"
    index_path: str = "./data/embeddings"
    
    # Processing
    batch_size: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    save_intermediate: bool = True
    output_dir: str = "./data/processed"


@dataclass
class ProcessingResult:
    """Result of processing a single paper."""
    paper_id: str
    source_file: str
    status: ProcessingStatus
    paper_structure: Optional[PaperStructure] = None
    chunks: Optional[List[TextChunk]] = None
    summary: Optional[SummaryResult] = None
    embedding_ids: Optional[List[str]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: str = ""


class ProgressTracker:
    """Track progress of pipeline operations."""
    
    def __init__(self, total_steps: int = 0):
        self.total_steps = total_steps
        self.completed_steps = 0
        self.current_step = ""
        self.start_time = time.time()
        self.step_times = []
        
    def update(self, step_name: str, completed: int = None):
        """Update progress."""
        if completed is not None:
            self.completed_steps = completed
        else:
            self.completed_steps += 1
            
        self.current_step = step_name
        self.step_times.append(time.time())
        
        if self.total_steps > 0:
            progress = (self.completed_steps / self.total_steps) * 100
            elapsed = time.time() - self.start_time
            
            if self.completed_steps > 0:
                avg_time_per_step = elapsed / self.completed_steps
                eta = avg_time_per_step * (self.total_steps - self.completed_steps)
            else:
                eta = 0
            
            logger.info(f"Progress: {progress:.1f}% ({self.completed_steps}/{self.total_steps}) - "
                       f"Current: {step_name} - ETA: {eta:.1f}s")
    
    def complete(self):
        """Mark as completed."""
        self.completed_steps = self.total_steps
        elapsed = time.time() - self.start_time
        logger.info(f"Processing completed in {elapsed:.2f}s")


class SummarizationPipeline:
    """
    Main pipeline for processing research papers.
    
    Coordinates PDF extraction, chunking, summarization, and vector indexing
    with comprehensive error handling and progress tracking.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Create output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.index_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.pdf_extractor = None
        self.chunker = None
        self.summarizer = None
        self.vector_index = None
        
        self._init_components()
        
        # Processing state
        self.processing_results = {}  # paper_id -> ProcessingResult
        
        logger.info("Summarization pipeline initialized")
    
    def _init_components(self):
        """Initialize pipeline components."""
        try:
            # PDF Extractor
            self.pdf_extractor = PDFExtractor(
                use_ocr=self.config.use_ocr,
                ocr_fallback_threshold=self.config.ocr_fallback_threshold
            )
            
            # Chunker
            self.chunker = ResearchPaperChunker(
                strategy=self.config.chunking_strategy,
                max_chunk_tokens=self.config.max_chunk_tokens,
                overlap_tokens=self.config.overlap_tokens,
                min_chunk_tokens=self.config.min_chunk_tokens
            )
            
            # Summarizer
            summary_config = SummaryConfig(
                summary_type=self.config.summary_type,
                max_summary_tokens=self.config.max_summary_tokens,
                temperature=self.config.temperature
            )
            
            self.summarizer = ResearchPaperSummarizer(
                provider=self.config.llm_provider,
                model_name=self.config.model_name,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                config=summary_config
            )
            
            # Vector Index
            self.vector_index = FaissVectorIndex(
                index_path=self.config.index_path,
                embedding_model=self.config.embedding_model,
                index_type=self.config.index_type
            )
            
            logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def process_paper(
        self,
        pdf_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ProcessingResult:
        """
        Process a single research paper through the entire pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing result with all extracted information
        """
        paper_id = str(uuid.uuid4())
        start_time = time.time()
        
        result = ProcessingResult(
            paper_id=paper_id,
            source_file=pdf_path,
            status=ProcessingStatus.PENDING,
            timestamp=datetime.now().isoformat()
        )
        
        self.processing_results[paper_id] = result
        
        logger.info(f"Starting pipeline processing for: {pdf_path}")
        
        try:
            # Step 1: PDF Extraction
            result.status = ProcessingStatus.EXTRACTING
            if progress_callback:
                progress_callback("Extracting PDF text", 0.2)
            
            result.paper_structure = self._extract_pdf_with_retry(pdf_path)
            
            if self.config.save_intermediate:
                self._save_paper_structure(result.paper_structure, paper_id)
            
            # Step 2: Chunking
            result.status = ProcessingStatus.CHUNKING
            if progress_callback:
                progress_callback("Chunking text", 0.4)
            
            result.chunks = self.chunker.chunk_paper(result.paper_structure)
            
            if self.config.save_intermediate:
                self._save_chunks(result.chunks, paper_id)
            
            # Step 3: Summarization
            result.status = ProcessingStatus.SUMMARIZING
            if progress_callback:
                progress_callback("Generating summary", 0.7)
            
            result.summary = self._summarize_with_retry(result.chunks)
            
            if self.config.save_intermediate:
                self._save_summary(result.summary, paper_id)
            
            # Step 4: Vector Indexing
            result.status = ProcessingStatus.INDEXING
            if progress_callback:
                progress_callback("Creating embeddings", 0.9)
            
            result.embedding_ids = self.vector_index.add_chunks(
                result.chunks,
                paper_title=result.paper_structure.title,
                authors=result.paper_structure.authors
            )
            
            # Add summary information to embeddings
            self.vector_index.add_summary(result.summary, pdf_path)
            
            # Save vector index
            self.vector_index.save_index()
            
            # Completion
            result.status = ProcessingStatus.COMPLETED
            result.processing_time = time.time() - start_time
            
            if progress_callback:
                progress_callback("Processing completed", 1.0)
            
            logger.info(f"Successfully processed paper in {result.processing_time:.2f}s")
            
        except Exception as e:
            result.status = ProcessingStatus.FAILED
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            
            logger.error(f"Failed to process paper {pdf_path}: {e}")
        
        return result
    
    def process_batch(
        self,
        pdf_paths: List[str],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, ProcessingResult]:
        """
        Process multiple papers in batch.
        
        Args:
            pdf_paths: List of PDF file paths
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary mapping paper IDs to processing results
        """
        logger.info(f"Starting batch processing of {len(pdf_paths)} papers")
        
        tracker = ProgressTracker(len(pdf_paths))
        results = {}
        
        for i, pdf_path in enumerate(pdf_paths):
            tracker.update(f"Processing {Path(pdf_path).name}")
            
            if progress_callback:
                overall_progress = i / len(pdf_paths)
                progress_callback(f"Paper {i+1}/{len(pdf_paths)}: {Path(pdf_path).name}", overall_progress)
            
            try:
                result = self.process_paper(pdf_path)
                results[result.paper_id] = result
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                # Continue with next paper
                continue
        
        tracker.complete()
        
        # Generate batch summary
        successful = sum(1 for r in results.values() if r.status == ProcessingStatus.COMPLETED)
        failed = len(results) - successful
        
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        
        return results
    
    def _extract_pdf_with_retry(self, pdf_path: str) -> PaperStructure:
        """Extract PDF with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                return self.pdf_extractor.extract_from_file(pdf_path)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                logger.warning(f"PDF extraction attempt {attempt + 1} failed: {e}")
                time.sleep(self.config.retry_delay * (attempt + 1))
    
    def _summarize_with_retry(self, chunks: List[TextChunk]) -> SummaryResult:
        """Summarize with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                return self.summarizer.summarize_chunks(chunks)
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                logger.warning(f"Summarization attempt {attempt + 1} failed: {e}")
                time.sleep(self.config.retry_delay * (attempt + 1))
    
    def _save_paper_structure(self, structure: PaperStructure, paper_id: str):
        """Save paper structure to file."""
        output_file = Path(self.config.output_dir) / f"{paper_id}_structure.json"
        
        try:
            structure_dict = asdict(structure)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(structure_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save paper structure: {e}")
    
    def _save_chunks(self, chunks: List[TextChunk], paper_id: str):
        """Save chunks to file."""
        output_file = Path(self.config.output_dir) / f"{paper_id}_chunks.json"
        
        try:
            chunks_dict = [chunk.to_dict() for chunk in chunks]
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save chunks: {e}")
    
    def _save_summary(self, summary: SummaryResult, paper_id: str):
        """Save summary to file."""
        output_file = Path(self.config.output_dir) / f"{paper_id}_summary.json"
        
        try:
            summary_dict = asdict(summary)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save summary: {e}")
    
    def search_papers(
        self,
        query: str,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Search for relevant papers using the vector index.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of search results
        """
        return self.vector_index.search(query, k=k, filter_metadata=filter_metadata)
    
    def get_similar_papers(self, source_file: str, k: int = 5) -> List[Any]:
        """
        Find papers similar to a given paper.
        
        Args:
            source_file: Source file path
            k: Number of similar papers to return
            
        Returns:
            List of similar papers
        """
        return self.vector_index.get_similar_papers(source_file, k=k)
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = {
            'config': asdict(self.config),
            'processing_results': len(self.processing_results),
            'successful_papers': len([r for r in self.processing_results.values() 
                                     if r.status == ProcessingStatus.COMPLETED]),
            'failed_papers': len([r for r in self.processing_results.values() 
                                if r.status == ProcessingStatus.FAILED]),
            'vector_index': self.vector_index.get_statistics() if self.vector_index else {}
        }
        
        # Processing time statistics
        completed_results = [r for r in self.processing_results.values() 
                           if r.status == ProcessingStatus.COMPLETED]
        
        if completed_results:
            processing_times = [r.processing_time for r in completed_results]
            stats['processing_times'] = {
                'avg': sum(processing_times) / len(processing_times),
                'min': min(processing_times),
                'max': max(processing_times),
                'total': sum(processing_times)
            }
        
        return stats
    
    def export_results(self, output_file: str):
        """Export all processing results to a file."""
        results_dict = {}
        
        for paper_id, result in self.processing_results.items():
            result_dict = asdict(result)
            # Convert complex objects to dicts for JSON serialization
            if result_dict.get('paper_structure'):
                result_dict['paper_structure'] = asdict(result.paper_structure)
            if result_dict.get('chunks'):
                result_dict['chunks'] = [chunk.to_dict() for chunk in result.chunks]
            if result_dict.get('summary'):
                result_dict['summary'] = asdict(result.summary)
            
            results_dict[paper_id] = result_dict
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(results_dict)} results to {output_file}")


def create_pipeline(
    config: Optional[PipelineConfig] = None,
    llm_provider: LLMProvider = LLMProvider.OLLAMA,
    model_name: str = "llama2",
    api_key: Optional[str] = None
) -> SummarizationPipeline:
    """
    Convenience function to create a summarization pipeline.
    
    Args:
        config: Pipeline configuration
        llm_provider: LLM provider to use
        model_name: Model name
        api_key: API key for cloud providers
        
    Returns:
        Configured pipeline
    """
    if config is None:
        config = PipelineConfig(
            llm_provider=llm_provider,
            model_name=model_name,
            api_key=api_key
        )
    
    return SummarizationPipeline(config)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <pdf_path_or_directory> [provider] [model]")
        print("Providers: ollama, openai, anthropic")
        sys.exit(1)
    
    input_path = sys.argv[1]
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
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        # Create pipeline
        pipeline = create_pipeline(
            llm_provider=provider,
            model_name=model_name
        )
        
        # Determine input files
        input_path = Path(input_path)
        
        if input_path.is_file() and input_path.suffix.lower() == '.pdf':
            # Single file
            pdf_files = [str(input_path)]
        elif input_path.is_dir():
            # Directory
            pdf_files = [str(f) for f in input_path.glob("*.pdf")]
        else:
            print(f"Invalid input path: {input_path}")
            sys.exit(1)
        
        if not pdf_files:
            print("No PDF files found")
            sys.exit(1)
        
        print(f"Processing {len(pdf_files)} PDF file(s)...")
        
        # Process files
        if len(pdf_files) == 1:
            result = pipeline.process_paper(pdf_files[0])
            
            print(f"\nProcessing Result:")
            print(f"Status: {result.status.value}")
            print(f"Processing time: {result.processing_time:.2f}s")
            
            if result.status == ProcessingStatus.COMPLETED:
                print(f"Title: {result.paper_structure.title}")
                print(f"Authors: {', '.join(result.paper_structure.authors)}")
                print(f"Sections: {len(result.paper_structure.sections)}")
                print(f"Chunks: {len(result.chunks)}")
                print(f"Summary length: {len(result.summary.summary)} chars")
                print(f"Embeddings: {len(result.embedding_ids)}")
            else:
                print(f"Error: {result.error_message}")
        
        else:
            results = pipeline.process_batch(pdf_files)
            
            successful = sum(1 for r in results.values() if r.status == ProcessingStatus.COMPLETED)
            failed = len(results) - successful
            
            print(f"\nBatch Processing Results:")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            
            # Export results
            output_file = f"processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            pipeline.export_results(output_file)
            print(f"Results exported to: {output_file}")
        
        # Show statistics
        stats = pipeline.get_pipeline_statistics()
        print(f"\nPipeline Statistics:")
        print(f"Vector index: {stats['vector_index'].get('total_embeddings', 0)} embeddings")
        print(f"Papers indexed: {stats['vector_index'].get('total_papers', 0)}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
