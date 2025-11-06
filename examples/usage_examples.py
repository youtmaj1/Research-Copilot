#!/usr/bin/env python3
"""
Example usage script for the Paper Collector module.

Demonstrates various ways to use the collector for research paper ingestion.
"""

import logging
import os
from datetime import datetime
from collector import PaperCollector, get_popular_categories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_basic_usage():
    """Example 1: Basic paper collection."""
    print("=" * 60)
    print("Example 1: Basic Paper Collection")
    print("=" * 60)
    
    # Initialize collector
    collector = PaperCollector(
        data_dir="example_data",
        db_path="example_papers.db"
    )
    
    # Search for papers on a specific topic
    query = "large language models"
    results = collector.search(
        query=query,
        max_results=10,
        download_pdfs=True
    )
    
    print(f"Search Query: {query}")
    print(f"Papers Found: {results['total_found']}")
    print(f"Papers Added: {results['papers_added']}")
    print(f"Papers Skipped: {results['papers_skipped']}")
    print(f"PDFs Downloaded: {results['pdfs_downloaded']}")
    
    if results['errors']:
        print(f"Errors encountered: {len(results['errors'])}")
    
    print()


def example_category_updates():
    """Example 2: Update recent papers from multiple categories."""
    print("=" * 60)
    print("Example 2: Category Updates")
    print("=" * 60)
    
    collector = PaperCollector(
        data_dir="example_data",
        db_path="example_papers.db"
    )
    
    # Update recent papers from multiple AI-related categories
    categories = ["cs.AI", "cs.LG", "cs.CL"]
    
    for category in categories:
        print(f"Updating category: {category}")
        results = collector.update_recent(
            category=category,
            days_back=3,  # Last 3 days
            download_pdfs=False  # Skip PDFs for faster processing
        )
        
        print(f"  Papers Found: {results['total_found']}")
        print(f"  Papers Added: {results['papers_added']}")
        print()


def example_advanced_search():
    """Example 3: Advanced search with multiple queries."""
    print("=" * 60)
    print("Example 3: Advanced Search")
    print("=" * 60)
    
    collector = PaperCollector(
        data_dir="example_data",
        db_path="example_papers.db"
    )
    
    # Search for papers on multiple advanced topics
    advanced_topics = [
        "transformer architecture attention mechanisms",
        "diffusion models image generation",
        "reinforcement learning from human feedback",
        "multimodal large language models"
    ]
    
    all_results = {}
    for topic in advanced_topics:
        print(f"Searching: {topic}")
        results = collector.search(
            query=topic,
            max_results=20,
            download_pdfs=True,
            use_scholar_fallback=True
        )
        all_results[topic] = results
        
        print(f"  ArXiv Papers: {results['arxiv_papers']}")
        print(f"  Scholar Papers: {results['scholar_papers']}")
        print(f"  Total Added: {results['papers_added']}")
        print()
    
    # Summary
    total_added = sum(r['papers_added'] for r in all_results.values())
    total_pdfs = sum(r['pdfs_downloaded'] for r in all_results.values())
    print(f"Summary - Total Papers Added: {total_added}")
    print(f"Summary - Total PDFs Downloaded: {total_pdfs}")
    print()


def example_local_search():
    """Example 4: Search locally stored papers."""
    print("=" * 60)
    print("Example 4: Local Paper Search")
    print("=" * 60)
    
    collector = PaperCollector(
        data_dir="example_data",
        db_path="example_papers.db"
    )
    
    # Search locally stored papers
    search_terms = ["attention", "transformer", "BERT"]
    
    for term in search_terms:
        papers = collector.search_local(
            query=term,
            limit=5
        )
        
        print(f"Local search for '{term}': {len(papers)} papers found")
        
        for i, paper in enumerate(papers[:3], 1):  # Show first 3
            print(f"  {i}. {paper['title'][:60]}...")
            print(f"     Authors: {', '.join(paper['authors'][:2])}...")
            print(f"     Source: {paper['source']}")
        
        if len(papers) > 3:
            print(f"     ... and {len(papers) - 3} more")
        print()


def example_statistics_monitoring():
    """Example 5: Monitor collection statistics."""
    print("=" * 60)
    print("Example 5: Statistics and Monitoring")
    print("=" * 60)
    
    collector = PaperCollector(
        data_dir="example_data",
        db_path="example_papers.db"
    )
    
    # Get comprehensive statistics
    stats = collector.get_stats()
    
    print("Collection Statistics:")
    print(f"  Total Papers: {stats.get('total_papers', 0)}")
    print(f"  Papers with PDFs: {stats.get('papers_with_pdfs', 0)}")
    print(f"  Recent Papers (30 days): {stats.get('recent_papers', 0)}")
    
    # Source breakdown
    by_source = stats.get('by_source', {})
    if by_source:
        print("\n  By Source:")
        for source, count in by_source.items():
            print(f"    {source.capitalize()}: {count}")
    
    # Calculate success rates
    total = stats.get('total_papers', 0)
    with_pdfs = stats.get('papers_with_pdfs', 0)
    if total > 0:
        pdf_rate = (with_pdfs / total) * 100
        print(f"\n  PDF Success Rate: {pdf_rate:.1f}%")
    
    print(f"\n  ArXiv Available: {stats.get('arxiv_available', False)}")
    print(f"  Scholar Available: {stats.get('scholar_available', False)}")
    print()


def example_error_handling():
    """Example 6: Error handling and recovery."""
    print("=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)
    
    collector = PaperCollector(
        data_dir="example_data",
        db_path="example_papers.db"
    )
    
    # Search with potential for errors (network issues, rate limiting, etc.)
    try:
        results = collector.search(
            query="quantum machine learning algorithms",
            max_results=50,
            download_pdfs=True
        )
        
        print("Search completed successfully!")
        print(f"Papers added: {results['papers_added']}")
        
        if results['errors']:
            print(f"Errors encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        print("Collection failed, but error was handled gracefully.")
    
    print()


def example_batch_processing():
    """Example 7: Batch processing for large-scale collection."""
    print("=" * 60)
    print("Example 7: Batch Processing")
    print("=" * 60)
    
    collector = PaperCollector(
        data_dir="example_data",
        db_path="example_papers.db"
    )
    
    # Define research areas for comprehensive collection
    research_areas = {
        "Natural Language Processing": [
            "natural language processing",
            "language models", 
            "machine translation",
            "text summarization"
        ],
        "Computer Vision": [
            "computer vision",
            "image recognition",
            "object detection",
            "image segmentation"
        ],
        "Machine Learning": [
            "deep learning",
            "neural networks",
            "reinforcement learning",
            "supervised learning"
        ]
    }
    
    total_collected = 0
    
    for area, queries in research_areas.items():
        print(f"Processing {area}...")
        area_total = 0
        
        for query in queries:
            results = collector.search(
                query=query,
                max_results=25,
                download_pdfs=False  # Skip PDFs for faster batch processing
            )
            area_total += results['papers_added']
            print(f"  '{query}': {results['papers_added']} papers")
        
        print(f"  {area} Total: {area_total} papers")
        total_collected += area_total
        print()
    
    print(f"Batch Processing Complete - Total Collected: {total_collected} papers")
    print()


def cleanup_example_data():
    """Clean up example data files."""
    import shutil
    
    try:
        if os.path.exists("example_data"):
            shutil.rmtree("example_data")
            print("Cleaned up example data directory")
        
        if os.path.exists("example_papers.db"):
            os.remove("example_papers.db")
            print("Cleaned up example database")
    except Exception as e:
        print(f"Cleanup warning: {e}")


def main():
    """Run all examples."""
    print("Research Paper Collector - Usage Examples")
    print("=" * 60)
    print(f"Starting examples at {datetime.now()}")
    print()
    
    try:
        # Run examples in sequence
        example_basic_usage()
        example_category_updates() 
        example_advanced_search()
        example_local_search()
        example_statistics_monitoring()
        example_error_handling()
        example_batch_processing()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
        # Show popular categories for reference
        print("\nPopular ArXiv Categories for AI/ML Research:")
        categories = get_popular_categories()
        for code, description in list(categories.items())[:5]:
            print(f"  {code}: {description}")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        logger.error(f"Examples failed: {e}")
        print("Some examples may have failed - check logs for details")
    finally:
        # Clean up example data
        print("\nCleaning up example data...")
        cleanup_example_data()


if __name__ == "__main__":
    main()
