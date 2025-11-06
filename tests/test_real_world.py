#!/usr/bin/env python3
"""
Real-world test of the Paper Collector module.
Tests actual paper collection from ArXiv.
"""

import sys
import os
import tempfile
import shutil
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collector import PaperCollector

def test_real_arxiv_search():
    """Test real ArXiv search with a small query."""
    print("Testing real ArXiv search...")
    print("=" * 40)
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    
    try:
        collector = PaperCollector(
            data_dir=temp_dir,
            db_path=os.path.join(temp_dir, "test_real.db")
        )
        
        # Search for a specific, recent paper (should be reliable)
        print("Searching for 'attention is all you need' papers...")
        results = collector.search(
            query="attention is all you need",
            max_results=3,
            download_pdfs=False,  # Skip PDFs for speed
            use_scholar_fallback=False  # ArXiv only for speed
        )
        
        print(f"Search completed!")
        print(f"Total found: {results.get('total_found', 0)}")
        print(f"Papers added: {results.get('papers_added', 0)}")
        print(f"Papers skipped: {results.get('papers_skipped', 0)}")
        
        if results.get('errors'):
            print(f"Errors: {results['errors']}")
        
        # Test local search
        if results.get('papers_added', 0) > 0:
            print("\nTesting local search...")
            local_papers = collector.search_local("attention")
            print(f"Found {len(local_papers)} papers locally")
            
            if local_papers:
                paper = local_papers[0]
                print(f"Sample paper: {paper['title'][:60]}...")
                print(f"Authors: {', '.join(paper['authors'][:2])}...")
                print(f"Source: {paper['source']}")
        
        # Get stats
        stats = collector.get_stats()
        print(f"\nDatabase stats:")
        print(f"Total papers: {stats.get('total_papers', 0)}")
        print(f"By source: {stats.get('by_source', {})}")
        
        success = results.get('total_found', 0) > 0
        
    except Exception as e:
        print(f"❌ Real search test failed: {e}")
        success = False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
    
    return success

def test_cli_interface():
    """Test the CLI interface."""
    print("\n" + "=" * 40)
    print("Testing CLI interface...")
    
    try:
        # Test categories command
        result = os.system(f"/Users/damian/Documents/projects/Research-Copilot/.venv/bin/python -m collector.cli categories")
        if result == 0:
            print("✅ CLI categories command works")
            return True
        else:
            print("❌ CLI categories command failed")
            return False
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def main():
    """Run real-world tests."""
    print("Paper Collector Real-World Testing")
    print("=" * 50)
    print(f"Starting tests at {datetime.now()}")
    print()
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Real ArXiv search
    if test_real_arxiv_search():
        tests_passed += 1
        print("✅ Real ArXiv search test passed")
    else:
        print("❌ Real ArXiv search test failed")
    
    # Test 2: CLI interface
    if test_cli_interface():
        tests_passed += 1
        print("✅ CLI interface test passed")
    else:
        print("❌ CLI interface test failed")
    
    print("\n" + "=" * 50)
    print(f"Real-world test results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All real-world tests passed! Module 1 is fully functional.")
        return True
    else:
        print("⚠️  Some real-world tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
