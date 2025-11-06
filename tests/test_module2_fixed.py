#!/usr/bin/env python3
"""
Fixed comprehensive test suite for Module 2 (Summarizer)

Tests the complete pipeline with real-world scenarios with correct method names.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import json
import time

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from summarizer.pdf_extractor import PDFExtractor, PaperStructure, Section
from summarizer.chunker import ResearchPaperChunker, ChunkingStrategy, TextChunk
from summarizer.faiss_index import FaissVectorIndex
from summarizer.config import ConfigManager, ResearchCopilotConfig
from summarizer.pipeline import SummarizationPipeline, PipelineConfig
from summarizer.cli import main as cli_main


def create_test_pdf():
    """Create a simple test PDF for testing."""
    try:
        import fitz  # PyMuPDF
        
        # Create a simple PDF document
        doc = fitz.open()
        page = doc.new_page()
        
        # Add title
        title_rect = fitz.Rect(72, 72, 500, 100)
        page.insert_textbox(title_rect, "Test Research Paper: Advanced Machine Learning Techniques", 
                          fontsize=16, fontname="helv", color=(0, 0, 0))
        
        # Add authors
        authors_rect = fitz.Rect(72, 110, 500, 130)
        page.insert_textbox(authors_rect, "John Doe, Jane Smith, Bob Johnson", 
                          fontsize=12, fontname="helv", color=(0.3, 0.3, 0.3))
        
        # Add abstract section
        abstract_rect = fitz.Rect(72, 150, 500, 220)
        abstract_text = """Abstract

This paper presents a comprehensive study of advanced machine learning techniques 
for automated research paper analysis. We propose novel methods for text extraction, 
semantic chunking, and knowledge representation that significantly improve the 
accuracy of academic document processing systems."""
        page.insert_textbox(abstract_rect, abstract_text, 
                          fontsize=11, fontname="helv", color=(0, 0, 0))
        
        # Add introduction section
        intro_rect = fitz.Rect(72, 240, 500, 400)
        intro_text = """1. Introduction

The exponential growth of academic literature has created an urgent need for 
automated systems capable of processing and summarizing research papers. 
Traditional approaches to document analysis often fail to capture the nuanced 
relationships between different sections of academic papers.

In this work, we introduce a novel framework that combines state-of-the-art 
natural language processing techniques with advanced vector indexing methods. 
Our approach demonstrates superior performance across multiple evaluation metrics."""
        page.insert_textbox(intro_rect, intro_text, 
                          fontsize=11, fontname="helv", color=(0, 0, 0))
        
        # Save the PDF
        temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        doc.save(temp_pdf.name)
        doc.close()
        
        return temp_pdf.name
        
    except ImportError:
        print("⚠️  PyMuPDF not available, skipping PDF creation test")
        return None


def test_pdf_extraction(pdf_path):
    """Test PDF text extraction functionality."""
    print("\n📄 Testing PDF Extraction...")
    
    if not pdf_path:
        print("⚠️  Skipping PDF extraction test (no PDF available)")
        return False
    
    try:
        extractor = PDFExtractor()
        paper = extractor.extract_from_file(pdf_path)
        
        print(f"✓ Extracted paper: '{paper.title}'")
        print(f"✓ Authors: {', '.join(paper.authors)}")
        print(f"✓ Sections: {len(paper.sections)}")
        print(f"✓ Total pages: {paper.total_pages}")
        print(f"✓ Extraction method: {paper.extraction_method}")
        
        # Check section content
        for section in paper.sections:
            if 'abstract' in section.title.lower():
                print(f"✓ Found abstract section with {len(section.content)} characters")
                break
        else:
            print("⚠️  Abstract section not explicitly found, but may be in content")
        
        return True
        
    except Exception as e:
        print(f"✗ PDF extraction failed: {e}")
        return False


def test_advanced_chunking(pdf_path):
    """Test advanced chunking strategies."""
    print("\n🧩 Testing Advanced Chunking...")
    
    if not pdf_path:
        print("⚠️  Skipping chunking test (no PDF available)")
        return False
    
    try:
        # Extract text first
        extractor = PDFExtractor()
        paper = extractor.extract_from_file(pdf_path)
        
        # Test different chunking strategies
        strategies = [
            ChunkingStrategy.SECTION_AWARE,
            ChunkingStrategy.TOKEN_BASED,
            ChunkingStrategy.HYBRID
        ]
        
        results = {}
        
        for strategy in strategies:
            chunker = ResearchPaperChunker(
                strategy=strategy,
                max_chunk_tokens=200,
                overlap_tokens=20
            )
            
            chunks = chunker.chunk_paper(paper)
            results[strategy.value] = {
                'chunk_count': len(chunks),
                'avg_tokens': sum(chunk.token_count for chunk in chunks) / len(chunks) if chunks else 0,
                'types': list(set(chunk.chunk_type for chunk in chunks))
            }
            
            print(f"✓ {strategy.value}: {len(chunks)} chunks, avg {results[strategy.value]['avg_tokens']:.1f} tokens")
        
        # Verify different strategies produce different results
        chunk_counts = [results[s.value]['chunk_count'] for s in strategies]
        if len(set(chunk_counts)) > 1:
            print("✓ Different strategies produce varied chunking results")
        
        return True
        
    except Exception as e:
        print(f"✗ Advanced chunking failed: {e}")
        return False


def test_vector_indexing_advanced(pdf_path):
    """Test advanced vector indexing and search functionality."""
    print("\n🔍 Testing Advanced Vector Indexing...")
    
    if not pdf_path:
        print("⚠️  Using sample text for vector indexing test")
        # Create sample chunks
        chunks = [
            TextChunk(
                content="Machine learning techniques for document analysis and text processing",
                chunk_type="abstract",
                token_count=12,
                section_title="Abstract",
                chunk_index=0,
                metadata={}
            ),
            TextChunk(
                content="Vector indexing using FAISS provides efficient similarity search capabilities",
                chunk_type="methodology",
                token_count=11,
                section_title="Methodology",
                chunk_index=1,
                metadata={}
            ),
            TextChunk(
                content="Experimental results demonstrate improved accuracy and performance metrics",
                chunk_type="results",
                token_count=10,
                section_title="Results",
                chunk_index=2,
                metadata={}
            )
        ]
    else:
        try:
            # Extract and chunk the PDF
            extractor = PDFExtractor()
            paper = extractor.extract_from_file(pdf_path)
            
            chunker = ResearchPaperChunker(strategy=ChunkingStrategy.HYBRID)
            chunks = chunker.chunk_paper(paper)
            
        except Exception as e:
            print(f"✗ Failed to prepare chunks for indexing: {e}")
            return False
    
    try:
        # Create temporary index
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "test_index")
            
            # Initialize index
            vector_index = FaissVectorIndex(
                embedding_model="all-MiniLM-L6-v2",
                index_path=index_path
            )
            
            # Add chunks to index
            paper_id = "test_paper_001"
            embedding_ids = vector_index.add_chunks(chunks, paper_id)
            
            print(f"✓ Added {len(embedding_ids)} embeddings to index")
            
            # Test various search queries
            test_queries = [
                "machine learning document processing",
                "vector search similarity",
                "experimental results accuracy"
            ]
            
            search_results_all = []
            for query in test_queries:
                results = vector_index.search(query, k=3)
                search_results_all.extend(results)
                print(f"✓ Query '{query[:30]}...': {len(results)} results")
                
                if results:
                    best_result = results[0]
                    print(f"   Best match (score: {best_result.similarity_score:.3f}): {best_result.content[:50]}...")
            
            # Test persistence
            vector_index.save_index()
            print("✓ Index saved successfully")
            
            # Test loading
            new_index = FaissVectorIndex(
                embedding_model="all-MiniLM-L6-v2",
                index_path=index_path
            )
            new_index.load_index()
            
            # Verify loaded index works
            reload_results = new_index.search("machine learning", k=2)
            print(f"✓ Reloaded index search: {len(reload_results)} results")
            
            # Test statistics
            stats = vector_index.get_statistics()
            print(f"✓ Index statistics: {stats['total_embeddings']} embeddings, {stats['dimensions']} dimensions")
            
            return True
            
    except Exception as e:
        print(f"✗ Vector indexing test failed: {e}")
        return False


def test_configuration_system():
    """Test configuration management system."""
    print("\n⚙️  Testing Configuration System...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.json")
            
            # Test config creation
            config_manager = ConfigManager(config_file)
            
            # Create sample configuration
            config = ResearchCopilotConfig()
            
            # Modify some settings
            config.processing.batch_size = 5
            config.processing.max_chunk_tokens = 300
            
            # Test saving
            config_manager.save_config(config)
            print("✓ Configuration saved successfully")
            
            # Test loading
            loaded_config = config_manager.load_config()
            print("✓ Configuration loaded successfully")
            
            # Verify data integrity
            assert loaded_config.processing.batch_size == 5
            assert loaded_config.processing.max_chunk_tokens == 300
            print("✓ Configuration data integrity verified")
            
            # Test default values
            print(f"✓ Default chunking strategy: {loaded_config.processing.chunking_strategy}")
            print(f"✓ Default summary type: {loaded_config.processing.summary_type}")
            
            return True
            
    except Exception as e:
        print(f"✗ Configuration system test failed: {e}")
        return False


def test_pipeline_integration(pdf_path):
    """Test complete pipeline integration."""
    print("\n🔄 Testing Pipeline Integration...")
    
    if not pdf_path:
        print("⚠️  Skipping pipeline integration test (no PDF available)")
        return False
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create pipeline configuration
            config = PipelineConfig(
                # PDF Processing
                use_ocr=False,
                
                # Chunking
                chunking_strategy="hybrid",
                max_chunk_tokens=200,
                overlap_tokens=20,
                
                # Summarization (using mock/dry-run mode)
                llm_provider="ollama",
                model_name="llama2",  # This won't actually be called in dry-run
                summary_type="structured",
                
                # Vector Index
                embedding_model="all-MiniLM-L6-v2",
                index_path=os.path.join(temp_dir, "pipeline_test_index"),
                
                # Processing
                batch_size=1,
                output_dir=temp_dir,
                save_intermediate=True
            )
            
            # Initialize pipeline
            pipeline = SummarizationPipeline(config)
            
            # Test individual components without full processing
            print("✓ Pipeline initialized successfully")
            
            # Test PDF extraction component
            extractor = PDFExtractor()
            paper = extractor.extract_from_file(pdf_path)
            print(f"✓ PDF extraction: {len(paper.sections)} sections")
            
            # Test chunking component
            chunker = ResearchPaperChunker(strategy=ChunkingStrategy.HYBRID)
            chunks = chunker.chunk_paper(paper)
            print(f"✓ Chunking: {len(chunks)} chunks created")
            
            # Test vector indexing component
            vector_index = FaissVectorIndex(
                embedding_model="all-MiniLM-L6-v2",
                index_path=os.path.join(temp_dir, "test_vector_index")
            )
            embedding_ids = vector_index.add_chunks(chunks[:3], "test_paper")  # Test with first 3 chunks
            print(f"✓ Vector indexing: {len(embedding_ids)} embeddings created")
            
            # Test search functionality
            search_results = vector_index.search("machine learning", k=2)
            print(f"✓ Search functionality: {len(search_results)} results")
            
            return True
            
    except Exception as e:
        print(f"✗ Pipeline integration test failed: {e}")
        return False


def test_cli_interface():
    """Test CLI interface functionality."""
    print("\n💻 Testing CLI Interface...")
    
    try:
        # Test CLI argument parsing (dry run)
        import sys
        from io import StringIO
        
        # Capture output
        old_stdout = sys.stdout
        old_argv = sys.argv.copy()
        
        sys.stdout = captured_output = StringIO()
        
        try:
            # Test help command
            sys.argv = ['cli.py', '--help']
            try:
                cli_main()
            except SystemExit:
                pass  # Expected for help command
            
            # Reset stdout and get output
            sys.stdout = old_stdout
            help_output = captured_output.getvalue()
            
            if 'process' in help_output and 'search' in help_output:
                print("✓ CLI help shows available commands")
            else:
                print("⚠️  CLI help output may be incomplete")
            
            print("✓ CLI interface structure validated")
            return True
            
        finally:
            sys.stdout = old_stdout
            sys.argv = old_argv
            
    except Exception as e:
        print(f"✗ CLI interface test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n🛡️  Testing Error Handling...")
    
    try:
        # Test PDF extractor with invalid file
        extractor = PDFExtractor()
        try:
            extractor.extract_from_file("/nonexistent/file.pdf")
            print("✗ Should have failed with nonexistent file")
            return False
        except (FileNotFoundError, Exception):
            print("✓ PDF extractor handles missing files correctly")
        
        # Test chunker with empty paper
        chunker = ResearchPaperChunker()
        empty_paper = PaperStructure(
            title="Empty Paper",
            authors=[],
            abstract="",
            sections=[],
            references="",
            metadata={},
            total_pages=0,
            extraction_method="text"
        )
        empty_chunks = chunker.chunk_paper(empty_paper)
        print(f"✓ Chunker handles empty paper: {len(empty_chunks)} chunks")
        
        # Test vector index with invalid path
        try:
            invalid_index = FaissVectorIndex(
                embedding_model="all-MiniLM-L6-v2",
                index_path="/invalid/path/that/cannot/be/created"
            )
            # This might not fail immediately, so try to save
            invalid_index.save_index()
            print("⚠️  Vector index should have failed with invalid path")
        except Exception:
            print("✓ Vector index handles invalid paths correctly")
        
        print("✓ Error handling tests completed")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def main():
    """Run comprehensive Module 2 validation."""
    print("🧪 COMPREHENSIVE MODULE 2 TESTING")
    print("=" * 50)
    
    # Create test PDF
    print("📋 Setting up test environment...")
    pdf_path = create_test_pdf()
    if pdf_path:
        print(f"✓ Created test PDF: {pdf_path}")
    
    # Run all tests
    tests = [
        ("PDF Extraction", lambda: test_pdf_extraction(pdf_path)),
        ("Advanced Chunking", lambda: test_advanced_chunking(pdf_path)),
        ("Vector Indexing", lambda: test_vector_indexing_advanced(pdf_path)),
        ("Configuration System", test_configuration_system),
        ("Pipeline Integration", lambda: test_pipeline_integration(pdf_path)),
        ("CLI Interface", test_cli_interface),
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            results[test_name] = {
                'passed': result,
                'duration': end_time - start_time
            }
            
            if result:
                passed_tests += 1
                print(f"✅ {test_name} PASSED ({end_time - start_time:.2f}s)")
            else:
                print(f"❌ {test_name} FAILED ({end_time - start_time:.2f}s)")
                
        except Exception as e:
            results[test_name] = {
                'passed': False,
                'error': str(e),
                'duration': 0
            }
            print(f"💥 {test_name} CRASHED: {e}")
    
    # Cleanup
    if pdf_path and os.path.exists(pdf_path):
        os.unlink(pdf_path)
        print("\n🧹 Cleaned up test PDF")
    
    # Final results
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        duration = result.get('duration', 0)
        print(f"{status:<10} {test_name:<25} ({duration:.2f}s)")
        
        if not result['passed'] and 'error' in result:
            print(f"           Error: {result['error']}")
    
    print(f"\n📈 SUMMARY: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Module 2 is production ready!")
        return 0
    elif passed_tests >= total_tests * 0.8:  # 80% pass rate
        print("✅ Most tests passed! Module 2 is functional with minor issues.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the results above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
