"""
Command Line Interface for Research Paper Summarizer

Provides CLI access to all summarizer functionality including:
- Single paper processing
- Batch processing
- Vector search
- Configuration management
- Statistics and reporting
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import List, Optional
import time

from .pipeline import (
    SummarizationPipeline, 
    PipelineConfig, 
    create_pipeline,
    ProcessingStatus
)
from .summarizer import LLMProvider, SummaryType
from .chunker import ChunkingStrategy
from .pdf_extractor import extract_pdf_structure
from .faiss_index import FaissVectorIndex

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def process_single_paper(args):
    """Process a single PDF paper."""
    print(f"Processing: {args.pdf_path}")
    
    # Create configuration
    config = PipelineConfig(
        llm_provider=LLMProvider(args.provider),
        model_name=args.model,
        api_key=args.api_key,
        summary_type=SummaryType(args.summary_type),
        chunking_strategy=ChunkingStrategy(args.chunking_strategy),
        max_chunk_tokens=args.max_chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        temperature=args.temperature,
        max_summary_tokens=args.max_summary_tokens,
        index_path=args.index_path,
        output_dir=args.output_dir,
        save_intermediate=args.save_intermediate
    )
    
    # Create pipeline
    pipeline = SummarizationPipeline(config)
    
    # Process paper
    def progress_callback(step: str, progress: float):
        print(f"[{progress*100:.0f}%] {step}")
    
    result = pipeline.process_paper(args.pdf_path, progress_callback)
    
    # Display results
    print("\n" + "="*60)
    print("PROCESSING RESULTS")
    print("="*60)
    
    print(f"Status: {result.status.value}")
    print(f"Processing time: {result.processing_time:.2f}s")
    
    if result.status == ProcessingStatus.COMPLETED:
        print(f"Paper ID: {result.paper_id}")
        
        if result.paper_structure:
            print(f"\nPaper Details:")
            print(f"  Title: {result.paper_structure.title}")
            print(f"  Authors: {', '.join(result.paper_structure.authors)}")
            print(f"  Sections: {len(result.paper_structure.sections)}")
            print(f"  Total pages: {result.paper_structure.total_pages}")
            print(f"  Extraction method: {result.paper_structure.extraction_method}")
        
        if result.chunks:
            print(f"\nChunking Results:")
            print(f"  Total chunks: {len(result.chunks)}")
            chunk_types = {}
            for chunk in result.chunks:
                chunk_types[chunk.metadata.chunk_type] = chunk_types.get(chunk.metadata.chunk_type, 0) + 1
            for chunk_type, count in chunk_types.items():
                print(f"  {chunk_type}: {count}")
        
        if result.summary:
            print(f"\nSummary Results:")
            print(f"  Model: {result.summary.model_used}")
            print(f"  Confidence: {result.summary.confidence_score:.2f}")
            print(f"  Summary length: {len(result.summary.summary)} chars")
            print(f"  Key points: {len(result.summary.key_points)}")
            
            if args.show_summary:
                print(f"\n{'='*40}")
                print("SUMMARY")
                print("="*40)
                print(result.summary.summary)
                
                if result.summary.key_points:
                    print(f"\n{'='*40}")
                    print("KEY POINTS")
                    print("="*40)
                    for i, point in enumerate(result.summary.key_points, 1):
                        print(f"{i}. {point}")
        
        if result.embedding_ids:
            print(f"\nVector Index:")
            print(f"  Embeddings created: {len(result.embedding_ids)}")
    
    else:
        print(f"Error: {result.error_message}")
        return 1
    
    return 0


def process_batch(args):
    """Process multiple PDF papers."""
    input_path = Path(args.input_path)
    
    # Find PDF files
    if input_path.is_file():
        pdf_files = [str(input_path)] if input_path.suffix.lower() == '.pdf' else []
    elif input_path.is_dir():
        pdf_files = [str(f) for f in input_path.glob("**/*.pdf")]
    else:
        print(f"Invalid input path: {input_path}")
        return 1
    
    if not pdf_files:
        print("No PDF files found")
        return 1
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Create configuration
    config = PipelineConfig(
        llm_provider=LLMProvider(args.provider),
        model_name=args.model,
        api_key=args.api_key,
        summary_type=SummaryType(args.summary_type),
        chunking_strategy=ChunkingStrategy(args.chunking_strategy),
        max_chunk_tokens=args.max_chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        temperature=args.temperature,
        max_summary_tokens=args.max_summary_tokens,
        index_path=args.index_path,
        output_dir=args.output_dir,
        save_intermediate=args.save_intermediate,
        batch_size=args.batch_size
    )
    
    # Create pipeline
    pipeline = SummarizationPipeline(config)
    
    # Process batch
    def progress_callback(step: str, progress: float):
        print(f"[{progress*100:.0f}%] {step}")
    
    start_time = time.time()
    results = pipeline.process_batch(pdf_files, progress_callback)
    total_time = time.time() - start_time
    
    # Display results
    successful = sum(1 for r in results.values() if r.status == ProcessingStatus.COMPLETED)
    failed = len(results) - successful
    
    print("\n" + "="*60)
    print("BATCH PROCESSING RESULTS")
    print("="*60)
    
    print(f"Total papers: {len(pdf_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per paper: {total_time/len(pdf_files):.2f}s")
    
    if failed > 0:
        print(f"\nFailed papers:")
        for result in results.values():
            if result.status == ProcessingStatus.FAILED:
                print(f"  {Path(result.source_file).name}: {result.error_message}")
    
    # Export results
    if args.export_results:
        output_file = args.export_results
        pipeline.export_results(output_file)
        print(f"\nResults exported to: {output_file}")
    
    # Show statistics
    stats = pipeline.get_pipeline_statistics()
    print(f"\nVector Index Statistics:")
    vector_stats = stats.get('vector_index', {})
    print(f"  Total embeddings: {vector_stats.get('total_embeddings', 0)}")
    print(f"  Total papers: {vector_stats.get('total_papers', 0)}")
    print(f"  Index type: {vector_stats.get('index_type', 'unknown')}")
    
    return 0 if successful > 0 else 1


def search_papers(args):
    """Search for papers using vector similarity."""
    if not Path(args.index_path).exists():
        print(f"Index not found at: {args.index_path}")
        return 1
    
    print(f"Loading vector index from: {args.index_path}")
    index = FaissVectorIndex(args.index_path)
    
    stats = index.get_statistics()
    print(f"Index contains {stats.get('total_embeddings', 0)} embeddings from {stats.get('total_papers', 0)} papers")
    
    if args.interactive:
        # Interactive search mode
        print("\nInteractive search mode. Type 'quit' to exit.")
        
        while True:
            query = input("\nEnter search query: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            results = index.search(query, k=args.k, min_similarity=args.min_similarity)
            
            if not results:
                print("No results found")
                continue
            
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Similarity: {result.similarity_score:.3f}")
                print(f"   Paper: {result.embedding_metadata.paper_title or 'Unknown'}")
                print(f"   File: {Path(result.embedding_metadata.source_file).name}")
                print(f"   Section: {result.embedding_metadata.section_title}")
                print(f"   Type: {result.embedding_metadata.chunk_type}")
    
    else:
        # Single query
        if not args.query:
            print("Query required for non-interactive mode")
            return 1
        
        results = index.search(args.query, k=args.k, min_similarity=args.min_similarity)
        
        if not results:
            print("No results found")
            return 0
        
        print(f"Search results for: '{args.query}'")
        print("-" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Similarity: {result.similarity_score:.3f}")
            print(f"   Paper: {result.embedding_metadata.paper_title or 'Unknown'}")
            print(f"   File: {Path(result.embedding_metadata.source_file).name}")
            print(f"   Section: {result.embedding_metadata.section_title}")
            print(f"   Type: {result.embedding_metadata.chunk_type}")
            
            if args.verbose:
                print(f"   Authors: {', '.join(result.embedding_metadata.authors)}")
                print(f"   Timestamp: {result.embedding_metadata.timestamp}")
                print(f"   Token count: {result.embedding_metadata.token_count}")
    
    return 0


def show_stats(args):
    """Show pipeline and index statistics."""
    stats = {}
    
    # Vector Index Stats
    if Path(args.index_path).exists():
        index = FaissVectorIndex(args.index_path)
        stats['vector_index'] = index.get_statistics()
    
    # Output directory stats
    if Path(args.output_dir).exists():
        output_path = Path(args.output_dir)
        structure_files = list(output_path.glob("*_structure.json"))
        chunk_files = list(output_path.glob("*_chunks.json"))
        summary_files = list(output_path.glob("*_summary.json"))
        
        stats['processed_papers'] = {
            'structures': len(structure_files),
            'chunks': len(chunk_files), 
            'summaries': len(summary_files)
        }
    
    print("="*50)
    print("RESEARCH COPILOT STATISTICS")
    print("="*50)
    
    if 'vector_index' in stats:
        vector_stats = stats['vector_index']
        print(f"\nVector Index:")
        print(f"  Path: {args.index_path}")
        print(f"  Total embeddings: {vector_stats.get('total_embeddings', 0)}")
        print(f"  Total papers: {vector_stats.get('total_papers', 0)}")
        print(f"  Index type: {vector_stats.get('index_type', 'unknown')}")
        print(f"  Embedding dimension: {vector_stats.get('embedding_dimension', 0)}")
        print(f"  Metric: {vector_stats.get('metric', 'unknown')}")
        
        chunk_types = vector_stats.get('chunk_types', {})
        if chunk_types:
            print(f"  Chunk types:")
            for chunk_type, count in chunk_types.items():
                print(f"    {chunk_type}: {count}")
    
    if 'processed_papers' in stats:
        processed = stats['processed_papers']
        print(f"\nProcessed Files:")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Paper structures: {processed['structures']}")
        print(f"  Chunk files: {processed['chunks']}")
        print(f"  Summary files: {processed['summaries']}")
    
    return 0


def extract_text(args):
    """Extract text from PDF (testing/debugging)."""
    print(f"Extracting text from: {args.pdf_path}")
    
    try:
        structure = extract_pdf_structure(args.pdf_path)
        
        print(f"\nExtraction Results:")
        print(f"  Title: {structure.title}")
        print(f"  Authors: {', '.join(structure.authors)}")
        print(f"  Abstract length: {len(structure.abstract)} chars")
        print(f"  Sections: {len(structure.sections)}")
        print(f"  References length: {len(structure.references)} chars")
        print(f"  Total pages: {structure.total_pages}")
        print(f"  Method: {structure.extraction_method}")
        
        if args.show_sections:
            print(f"\nSections:")
            for i, section in enumerate(structure.sections, 1):
                print(f"  {i}. {section.title} ({len(section.content)} chars)")
        
        if args.show_abstract and structure.abstract:
            print(f"\nAbstract:")
            print("-" * 40)
            print(structure.abstract)
        
        if args.save_structure:
            output_file = args.save_structure
            with open(output_file, 'w', encoding='utf-8') as f:
                import dataclasses
                json.dump(dataclasses.asdict(structure), f, indent=2, ensure_ascii=False)
            print(f"\nStructure saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Research Paper Summarizer CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single paper
  python -m summarizer.cli process paper.pdf --show-summary
  
  # Batch process directory
  python -m summarizer.cli batch papers/ --export-results results.json
  
  # Search papers
  python -m summarizer.cli search "machine learning" --k 5
  
  # Interactive search
  python -m summarizer.cli search --interactive
  
  # Show statistics
  python -m summarizer.cli stats
        """
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--index-path', default='./data/embeddings', help='Vector index path')
    parser.add_argument('--output-dir', default='./data/processed', help='Output directory')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process single paper
    process_parser = subparsers.add_parser('process', help='Process a single PDF paper')
    process_parser.add_argument('pdf_path', help='Path to PDF file')
    process_parser.add_argument('--provider', choices=['ollama', 'openai', 'anthropic'], 
                               default='ollama', help='LLM provider')
    process_parser.add_argument('--model', default='llama2', help='Model name')
    process_parser.add_argument('--api-key', help='API key for cloud providers')
    process_parser.add_argument('--summary-type', choices=['extractive', 'abstractive', 'structured', 'bullet_points'],
                               default='structured', help='Summary type')
    process_parser.add_argument('--chunking-strategy', choices=['section_aware', 'token_based', 'semantic', 'hybrid'],
                               default='hybrid', help='Chunking strategy')
    process_parser.add_argument('--max-chunk-tokens', type=int, default=1000, help='Max tokens per chunk')
    process_parser.add_argument('--overlap-tokens', type=int, default=100, help='Overlap between chunks')
    process_parser.add_argument('--temperature', type=float, default=0.3, help='LLM temperature')
    process_parser.add_argument('--max-summary-tokens', type=int, default=500, help='Max summary tokens')
    process_parser.add_argument('--show-summary', action='store_true', help='Show full summary')
    process_parser.add_argument('--save-intermediate', action='store_true', default=True, help='Save intermediate files')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Process multiple PDF papers')
    batch_parser.add_argument('input_path', help='Directory containing PDF files or single PDF')
    batch_parser.add_argument('--provider', choices=['ollama', 'openai', 'anthropic'], 
                             default='ollama', help='LLM provider')
    batch_parser.add_argument('--model', default='llama2', help='Model name')
    batch_parser.add_argument('--api-key', help='API key for cloud providers')
    batch_parser.add_argument('--summary-type', choices=['extractive', 'abstractive', 'structured', 'bullet_points'],
                             default='structured', help='Summary type')
    batch_parser.add_argument('--chunking-strategy', choices=['section_aware', 'token_based', 'semantic', 'hybrid'],
                             default='hybrid', help='Chunking strategy')
    batch_parser.add_argument('--max-chunk-tokens', type=int, default=1000, help='Max tokens per chunk')
    batch_parser.add_argument('--overlap-tokens', type=int, default=100, help='Overlap between chunks')
    batch_parser.add_argument('--temperature', type=float, default=0.3, help='LLM temperature')
    batch_parser.add_argument('--max-summary-tokens', type=int, default=500, help='Max summary tokens')
    batch_parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    batch_parser.add_argument('--export-results', help='Export results to JSON file')
    batch_parser.add_argument('--save-intermediate', action='store_true', default=True, help='Save intermediate files')
    
    # Search papers
    search_parser = subparsers.add_parser('search', help='Search for papers using vector similarity')
    search_parser.add_argument('query', nargs='?', help='Search query (omit for interactive mode)')
    search_parser.add_argument('--interactive', '-i', action='store_true', help='Interactive search mode')
    search_parser.add_argument('--k', type=int, default=10, help='Number of results to return')
    search_parser.add_argument('--min-similarity', type=float, default=0.0, help='Minimum similarity threshold')
    
    # Statistics
    stats_parser = subparsers.add_parser('stats', help='Show pipeline statistics')
    
    # Extract text (debugging)
    extract_parser = subparsers.add_parser('extract', help='Extract text from PDF (debugging)')
    extract_parser.add_argument('pdf_path', help='Path to PDF file')
    extract_parser.add_argument('--show-sections', action='store_true', help='Show section list')
    extract_parser.add_argument('--show-abstract', action='store_true', help='Show abstract')
    extract_parser.add_argument('--save-structure', help='Save structure to JSON file')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Execute command
    try:
        if args.command == 'process':
            return process_single_paper(args)
        elif args.command == 'batch':
            return process_batch(args)
        elif args.command == 'search':
            return search_papers(args)
        elif args.command == 'stats':
            return show_stats(args)
        elif args.command == 'extract':
            return extract_text(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
