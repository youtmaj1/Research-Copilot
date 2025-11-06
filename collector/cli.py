#!/usr/bin/env python3
"""
Command Line Interface for the Paper Collector module.

Provides command-line access to paper collection functionality.
"""

import argparse
import logging
import sys
import json
from pathlib import Path

from collector import PaperCollector, get_popular_categories


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def search_command(args):
    """Execute search command."""
    collector = PaperCollector(
        data_dir=args.data_dir,
        db_path=args.db_path
    )
    
    print(f"Searching for: {args.query}")
    print(f"Max results: {args.max_results}")
    print(f"Download PDFs: {args.download_pdfs}")
    print("=" * 50)
    
    results = collector.search(
        query=args.query,
        max_results=args.max_results,
        download_pdfs=args.download_pdfs,
        use_scholar_fallback=args.use_scholar
    )
    
    # Print results
    print(f"Search completed!")
    print(f"Total found: {results['total_found']}")
    print(f"Papers added: {results['papers_added']}")
    print(f"Papers skipped: {results['papers_skipped']}")
    if args.download_pdfs:
        print(f"PDFs downloaded: {results['pdfs_downloaded']}")
    
    if results['errors']:
        print(f"Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")


def update_command(args):
    """Execute update recent command."""
    collector = PaperCollector(
        data_dir=args.data_dir,
        db_path=args.db_path
    )
    
    print(f"Updating recent papers for category: {args.category}")
    print(f"Days back: {args.days_back}")
    print(f"Download PDFs: {args.download_pdfs}")
    print("=" * 50)
    
    results = collector.update_recent(
        category=args.category,
        days_back=args.days_back,
        download_pdfs=args.download_pdfs
    )
    
    # Print results
    print(f"Update completed!")
    print(f"Total found: {results['total_found']}")
    print(f"Papers added: {results['papers_added']}")
    print(f"Papers skipped: {results['papers_skipped']}")
    if args.download_pdfs:
        print(f"PDFs downloaded: {results['pdfs_downloaded']}")


def stats_command(args):
    """Execute stats command."""
    collector = PaperCollector(
        data_dir=args.data_dir,
        db_path=args.db_path
    )
    
    stats = collector.get_stats()
    
    print("Paper Collector Statistics")
    print("=" * 50)
    print(f"Total papers: {stats.get('total_papers', 0)}")
    print(f"Papers with PDFs: {stats.get('papers_with_pdfs', 0)}")
    print(f"Recent papers (30 days): {stats.get('recent_papers', 0)}")
    
    by_source = stats.get('by_source', {})
    if by_source:
        print("\nBy source:")
        for source, count in by_source.items():
            print(f"  {source}: {count}")
    
    print(f"\nData directory: {stats.get('data_directory', 'N/A')}")
    print(f"ArXiv available: {stats.get('arxiv_available', False)}")
    print(f"Scholar available: {stats.get('scholar_available', False)}")


def local_search_command(args):
    """Execute local search command."""
    collector = PaperCollector(
        data_dir=args.data_dir,
        db_path=args.db_path
    )
    
    papers = collector.search_local(
        query=args.query,
        source=args.source,
        limit=args.limit
    )
    
    print(f"Local search results for: {args.query}")
    print(f"Found {len(papers)} papers")
    print("=" * 50)
    
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'])}")
        print(f"   Source: {paper['source']}")
        if paper.get('published_date'):
            print(f"   Published: {paper['published_date'][:10]}")
        if paper.get('pdf_path'):
            print(f"   PDF: Available")
        print()


def categories_command(args):
    """Execute categories command."""
    categories = get_popular_categories()
    print("Popular ArXiv categories:")
    print("=" * 30)
    for category, description in categories.items():
        print(f"{category:10} - {description}")


def fetch_and_chat_command(args):
    """Fetch papers and start interactive chat."""
    from datetime import datetime, timedelta
    from config.ollama_config import OllamaConfigManager
    
    collector = PaperCollector(
        data_dir=args.data_dir,
        db_path=args.db_path
    )
    
    # Calculate date filter if specified
    date_filter = None
    if args.days_back:
        date_filter = (datetime.now() - timedelta(days=args.days_back)).strftime('%Y-%m-%d')
        print(f"📅 Looking for papers from the last {args.days_back} days (since {date_filter})")
    
    print(f"🔍 Searching for: '{args.topic}'")
    print(f"📄 Max results: {args.max_results}")
    print("=" * 50)
    
    # Search for papers
    results = collector.search(
        query=args.topic,
        max_results=args.max_results,
        download_pdfs=False,  # Skip PDFs for faster interaction
        use_scholar_fallback=True
    )
    
    papers_found = results.get('papers_added', 0)
    print(f"✅ Collected {papers_found} new papers on '{args.topic}'")
    
    # Also check existing papers in database
    import sqlite3
    conn = sqlite3.connect(args.db_path or 'papers.db')
    cursor = conn.cursor()
    
    # Search existing papers with flexible matching
    search_terms = args.topic.lower().split()
    where_conditions = []
    params = []
    
    for term in search_terms:
        where_conditions.append("(LOWER(title) LIKE ? OR LOWER(abstract) LIKE ?)")
        params.extend([f"%{term}%", f"%{term}%"])
    
    where_clause = " OR ".join(where_conditions)
    
    cursor.execute(f'SELECT COUNT(*) FROM papers WHERE {where_clause}', params)
    existing_papers = cursor.fetchone()[0]
    
    total_papers = papers_found + existing_papers
    print(f"📚 Total relevant papers available: {total_papers} ({existing_papers} existing + {papers_found} new)")
    
    if total_papers == 0:
        print("❌ No papers found. Try different keywords or increase max-results.")
        conn.close()
        return
    
    # Show some existing papers
    if existing_papers > 0:
        cursor.execute(f'SELECT title, authors FROM papers WHERE {where_clause} ORDER BY created_at DESC LIMIT 5', params)
        existing_paper_list = cursor.fetchall()
        
        print("\n📖 Relevant papers found:")
        for i, (title, authors) in enumerate(existing_paper_list, 1):
            print(f"  {i}. {title}")
            print(f"     Authors: {authors}")
    
    conn.close()
    
    # Initialize LLM
    print("\n🤖 Initializing phi4-mini:3.8b for chat...")
    try:
        ollama = OllamaConfigManager()
        print(f"✅ LLM ready: {ollama.model} (temperature: 0.3)")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return
    

    
    # Start interactive chat session
    print("\n" + "🤖 " + "=" * 58)
    print(f"Research Assistant ready! Ask me anything about '{args.topic}'")
    print("Type your questions naturally. Use /quit to exit.")
    print("=" * 60)
    
    while True:
        try:
            # Simple input prompt like ollama
            question = input("\n>>> ").strip()
            
            if question.lower() in ['/quit', '/exit', '/q', 'quit', 'exit']:
                break
            
            if not question:
                continue
            
            # Build the research-focused prompt - keep it simple
            research_prompt = f"""Based on research papers about "{args.topic}", answer this question: {question}

Provide a comprehensive, factual answer."""
            
            print("🤖 Thinking...")
            
            result = ollama.generate_completion(
                research_prompt, 
                max_tokens=500, 
                temperature=0.3
            )
            
            if result and result.get('success'):
                print("\n" + result['response'] + "\n")
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No response'
                print(f"\n❌ Error: {error_msg}\n")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except EOFError:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
    
    print("Session ended.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Research Paper Collector - Collect papers from ArXiv and Scholar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s search "large language models" --max-results 50
  %(prog)s update cs.AI --days-back 7
  %(prog)s stats
  %(prog)s local-search "transformer" --source arxiv
  %(prog)s categories
        """
    )
    
    # Global arguments
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory path (default: data)"
    )
    parser.add_argument(
        "--db-path", 
        default="papers.db",
        help="Database file path (default: papers.db)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for papers")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=100,
        help="Maximum number of results (default: 100)"
    )
    search_parser.add_argument(
        "--no-pdfs",
        dest="download_pdfs",
        action="store_false",
        help="Don't download PDF files"
    )
    search_parser.add_argument(
        "--no-scholar",
        dest="use_scholar",
        action="store_false", 
        help="Don't use Scholar fallback"
    )
    search_parser.set_defaults(func=search_command)
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update recent papers")
    update_parser.add_argument("category", help="ArXiv category (e.g., cs.AI)")
    update_parser.add_argument(
        "--days-back", "-d",
        type=int,
        default=7,
        help="Number of days to look back (default: 7)"
    )
    update_parser.add_argument(
        "--no-pdfs",
        dest="download_pdfs",
        action="store_false",
        help="Don't download PDF files"
    )
    update_parser.set_defaults(func=update_command)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.set_defaults(func=stats_command)
    
    # Local search command
    local_parser = subparsers.add_parser("local-search", help="Search local papers")
    local_parser.add_argument("query", help="Search query")
    local_parser.add_argument(
        "--source", "-s",
        choices=["arxiv", "scholar"],
        help="Filter by source"
    )
    local_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=50,
        help="Maximum number of results (default: 50)"
    )
    local_parser.set_defaults(func=local_search_command)
    
    # Categories command
    categories_parser = subparsers.add_parser("categories", help="List ArXiv categories")
    categories_parser.set_defaults(func=categories_command)
    
    # Fetch and chat command
    fetch_chat_parser = subparsers.add_parser("fetch-and-chat", help="Fetch papers and start interactive chat")
    fetch_chat_parser.add_argument("topic", help="Paper topic to search for")
    fetch_chat_parser.add_argument(
        "--max-results", "-n",
        type=int,
        default=10,
        help="Maximum number of papers to fetch (default: 10)"
    )
    fetch_chat_parser.add_argument(
        "--days-back", "-d",
        type=int,
        help="Only papers from the last N days (optional)"
    )
    fetch_chat_parser.set_defaults(func=fetch_and_chat_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
