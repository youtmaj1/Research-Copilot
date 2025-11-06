#!/usr/bin/env python3
"""
Cross-Reference CLI

Command-line interface for research paper cross-referencing operations.
Provides tools for processing papers, building knowledge graphs, and managing
the cross-reference database.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import sqlite3

from .pipeline import CrossRefPipeline, CrossRefConfig, create_crossref_pipeline
from .graph import GraphExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrossRefCLI:
    """Command-line interface for cross-reference operations."""
    
    def __init__(self):
        self.pipeline = None
    
    def setup_pipeline(self, args) -> CrossRefPipeline:
        """Setup pipeline with configuration from arguments."""
        if self.pipeline is None:
            config_kwargs = {}
            
            # Database configuration
            if hasattr(args, 'database') and args.database:
                config_kwargs['database_path'] = args.database
            
            # Output configuration
            if hasattr(args, 'output_dir') and args.output_dir:
                config_kwargs['output_dir'] = args.output_dir
            
            # Processing configuration
            if hasattr(args, 'similarity_threshold') and args.similarity_threshold:
                config_kwargs['similarity_threshold'] = args.similarity_threshold
            
            if hasattr(args, 'citation_threshold') and args.citation_threshold:
                config_kwargs['citation_confidence_threshold'] = args.citation_threshold
            
            if hasattr(args, 'embedding_model') and args.embedding_model:
                config_kwargs['embedding_model'] = args.embedding_model
            
            if hasattr(args, 'batch_size') and args.batch_size:
                config_kwargs['batch_size'] = args.batch_size
            
            # Export formats
            if hasattr(args, 'export_formats') and args.export_formats:
                config_kwargs['graph_export_formats'] = args.export_formats.split(',')
            
            self.pipeline = create_crossref_pipeline(**config_kwargs)
        
        return self.pipeline
    
    def cmd_process(self, args):
        """Process papers for cross-referencing."""
        print(f"🔄 Processing papers from {args.papers_file}")
        
        # Load papers data
        if not Path(args.papers_file).exists():
            print(f"❌ Papers file not found: {args.papers_file}")
            return 1
        
        with open(args.papers_file, 'r') as f:
            papers = json.load(f)
        
        print(f"📄 Loaded {len(papers)} papers")
        
        # Setup PDF paths if directory provided
        pdf_paths = {}
        if args.pdf_dir:
            pdf_dir = Path(args.pdf_dir)
            if not pdf_dir.exists():
                print(f"⚠️  PDF directory not found: {args.pdf_dir}")
            else:
                for paper_id in papers:
                    for ext in ['.pdf', '.PDF']:
                        pdf_file = pdf_dir / f"{paper_id}{ext}"
                        if pdf_file.exists():
                            pdf_paths[paper_id] = str(pdf_file)
                            break
                
                print(f"📎 Found PDFs for {len(pdf_paths)} papers")
        
        # Setup pipeline
        pipeline = self.setup_pipeline(args)
        
        # Process papers
        start_time = time.time()
        graph = pipeline.process_papers(papers, pdf_paths)
        processing_time = time.time() - start_time
        
        # Get statistics
        stats = pipeline.get_pipeline_statistics()
        
        print(f"\n✅ Processing completed in {processing_time:.2f} seconds")
        print(f"📊 Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        print(f"🔗 Relationships: {stats['database']['total_relationships']}")
        print(f"📝 Citations: {stats['database']['total_citations']}")
        print(f"🎯 Similarities: {stats['database']['total_similarities']}")
        
        if args.verbose:
            print("\n📈 Relationship breakdown:")
            for rel_type, count in stats['database']['relationships_by_type'].items():
                print(f"  {rel_type}: {count}")
        
        return 0
    
    def cmd_add_paper(self, args):
        """Add a single paper to the cross-reference database."""
        print(f"➕ Adding paper: {args.paper_id}")
        
        # Load paper data
        if args.paper_file:
            with open(args.paper_file, 'r') as f:
                paper_data = json.load(f)
        else:
            # Create paper data from command line arguments
            paper_data = {
                'title': args.title or '',
                'authors': args.authors.split(',') if args.authors else [],
                'abstract': args.abstract or '',
                'doi': args.doi,
                'arxiv_id': args.arxiv_id,
                'year': args.year,
                'keywords': args.keywords.split(',') if args.keywords else []
            }
        
        # Setup pipeline
        pipeline = self.setup_pipeline(args)
        
        # Process single paper
        results = pipeline.process_single_paper(
            args.paper_id,
            paper_data,
            args.pdf_path
        )
        
        print(f"✅ Added paper {args.paper_id}")
        print(f"🔗 Found {len(results['citations'])} citations")
        print(f"🎯 Found {len(results['similarities'])} similar papers")
        
        if args.verbose and results['similarities']:
            print("\n🎯 Most similar papers:")
            for sim in results['similarities'][:5]:
                print(f"  {sim['target_paper_id']}: {sim['similarity_score']:.3f}")
        
        return 0
    
    def cmd_export(self, args):
        """Export cross-reference graph."""
        print(f"📤 Exporting graph from database: {args.database}")
        
        # Setup pipeline
        pipeline = self.setup_pipeline(args)
        
        # Get papers and relationships from database
        papers = pipeline.database.get_paper_metadata()
        relationships = pipeline.database.get_relationships()
        
        if not papers:
            print("❌ No papers found in database")
            return 1
        
        # Rebuild graph from database
        from .graph import CrossRefGraph
        graph = CrossRefGraph()
        graph.add_papers(papers)
        
        # Add relationships as edges
        for rel in relationships:
            graph.add_edge(
                rel['source_paper'],
                rel['target_paper'],
                relationship=rel['relation'],
                weight=rel['score'],
                metadata=rel.get('metadata', {})
            )
        
        # Export in requested formats
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        formats = args.format.split(',')
        
        for fmt in formats:
            fmt = fmt.strip().lower()
            
            try:
                if fmt == 'json':
                    output_file = output_dir / f"crossref_graph_{timestamp}.json"
                    GraphExporter.export_to_json(graph, str(output_file))
                    print(f"📄 Exported JSON: {output_file}")
                
                elif fmt == 'neo4j':
                    output_path = output_dir / f"neo4j_export_{timestamp}"
                    GraphExporter.export_to_neo4j_csv(graph, str(output_path))
                    print(f"📊 Exported Neo4j CSV: {output_path}")
                
                elif fmt == 'networkx':
                    output_file = output_dir / f"networkx_graph_{timestamp}.pkl"
                    GraphExporter.export_to_networkx_pickle(graph, str(output_file))
                    print(f"🐍 Exported NetworkX: {output_file}")
                
                elif fmt == 'edgelist':
                    output_file = output_dir / f"edgelist_{timestamp}.csv"
                    GraphExporter.export_edge_list(graph, str(output_file))
                    print(f"📋 Exported edge list: {output_file}")
                
                else:
                    print(f"⚠️  Unknown format: {fmt}")
            
            except Exception as e:
                print(f"❌ Failed to export {fmt}: {e}")
        
        return 0
    
    def cmd_stats(self, args):
        """Show database and pipeline statistics."""
        pipeline = self.setup_pipeline(args)
        stats = pipeline.get_pipeline_statistics()
        
        print(f"📊 Cross-Reference Statistics")
        print(f"{'='*50}")
        
        # Database stats
        db_stats = stats['database']
        print(f"📄 Total papers: {db_stats['total_papers']}")
        print(f"🔗 Total relationships: {db_stats['total_relationships']}")
        print(f"📝 Citations: {db_stats['total_citations']}")
        print(f"🎯 Similarities: {db_stats['total_similarities']}")
        
        if 'relationships_by_type' in db_stats:
            print(f"\n📈 Relationships by type:")
            for rel_type, count in db_stats['relationships_by_type'].items():
                print(f"  {rel_type}: {count}")
        
        # Similarity engine stats
        if 'similarity_engine' in stats:
            sim_stats = stats['similarity_engine']
            print(f"\n🎯 Similarity Engine:")
            print(f"  Papers indexed: {sim_stats.get('papers_indexed', 0)}")
            print(f"  Index size: {sim_stats.get('index_size', 0)}")
            print(f"  Embedding model: {sim_stats.get('embedding_model', 'unknown')}")
        
        # Configuration
        if args.verbose:
            print(f"\n⚙️  Configuration:")
            config = stats['config']
            for key, value in config.items():
                if key != 'graph_export_formats' or value:
                    print(f"  {key}: {value}")
        
        return 0
    
    def cmd_relationships(self, args):
        """Show relationships for a specific paper."""
        pipeline = self.setup_pipeline(args)
        relationships = pipeline.get_paper_relationships(args.paper_id)
        
        if not relationships:
            print(f"❌ No relationships found for paper: {args.paper_id}")
            return 1
        
        print(f"🔗 Relationships for paper: {args.paper_id}")
        print(f"{'='*60}")
        
        for rel_type, rels in relationships.items():
            print(f"\n📌 {rel_type.upper()} ({len(rels)} relationships)")
            
            # Sort by score (descending)
            sorted_rels = sorted(rels, key=lambda x: x['score'], reverse=True)
            
            # Show top relationships
            limit = args.limit if hasattr(args, 'limit') else 10
            for rel in sorted_rels[:limit]:
                score = rel['score']
                target = rel['target_paper']
                confidence = rel['confidence']
                
                print(f"  → {target}")
                print(f"    Score: {score:.3f}, Confidence: {confidence:.3f}")
                
                if args.verbose and rel.get('metadata'):
                    metadata = rel['metadata']
                    if isinstance(metadata, dict):
                        for key, value in metadata.items():
                            print(f"    {key}: {value}")
                print()
        
        return 0
    
    def cmd_search(self, args):
        """Search for papers by title, author, or content."""
        pipeline = self.setup_pipeline(args)
        
        # Get all papers
        papers = pipeline.database.get_paper_metadata()
        
        if not papers:
            print("❌ No papers in database")
            return 1
        
        # Simple text search
        query = args.query.lower()
        matches = []
        
        for paper_id, paper_data in papers.items():
            # Search in title
            if query in paper_data.get('title', '').lower():
                matches.append((paper_id, paper_data, 'title'))
                continue
            
            # Search in authors
            authors = ' '.join(paper_data.get('authors', [])).lower()
            if query in authors:
                matches.append((paper_id, paper_data, 'author'))
                continue
            
            # Search in abstract
            if query in paper_data.get('abstract', '').lower():
                matches.append((paper_id, paper_data, 'abstract'))
                continue
            
            # Search in keywords
            keywords = ' '.join(paper_data.get('keywords', [])).lower()
            if query in keywords:
                matches.append((paper_id, paper_data, 'keywords'))
        
        if not matches:
            print(f"❌ No papers found matching: {query}")
            return 1
        
        print(f"🔍 Found {len(matches)} papers matching: {query}")
        print(f"{'='*60}")
        
        for paper_id, paper_data, match_field in matches[:args.limit]:
            print(f"\n📄 {paper_id}")
            print(f"   Title: {paper_data.get('title', 'No title')}")
            print(f"   Authors: {', '.join(paper_data.get('authors', []))}")
            print(f"   Year: {paper_data.get('year', 'Unknown')}")
            print(f"   Match: {match_field}")
            
            if args.verbose:
                if paper_data.get('doi'):
                    print(f"   DOI: {paper_data['doi']}")
                if paper_data.get('arxiv_id'):
                    print(f"   arXiv: {paper_data['arxiv_id']}")
        
        return 0
    
    def cmd_init(self, args):
        """Initialize a new cross-reference database."""
        db_path = Path(args.database)
        
        if db_path.exists() and not args.force:
            print(f"❌ Database already exists: {db_path}")
            print("Use --force to overwrite")
            return 1
        
        print(f"🆕 Initializing database: {db_path}")
        
        # Create pipeline (this will initialize the database)
        pipeline = self.setup_pipeline(args)
        
        print(f"✅ Database initialized successfully")
        print(f"📍 Location: {db_path.absolute()}")
        
        return 0
    
    def cmd_validate(self, args):
        """Validate database integrity and relationships."""
        pipeline = self.setup_pipeline(args)
        
        print("🔍 Validating database...")
        
        issues = []
        
        # Check database connection
        try:
            stats = pipeline.get_pipeline_statistics()
        except Exception as e:
            issues.append(f"Database connection error: {e}")
            
        # Validate relationships
        try:
            relationships = pipeline.database.get_relationships()
            paper_ids = set(pipeline.database.get_paper_metadata().keys())
            
            for rel in relationships:
                if rel['source_paper'] not in paper_ids:
                    issues.append(f"Invalid source paper: {rel['source_paper']}")
                
                if rel['target_paper'] not in paper_ids:
                    issues.append(f"Invalid target paper: {rel['target_paper']}")
                
                if not (0 <= rel['score'] <= 1):
                    issues.append(f"Invalid score: {rel['score']} for relationship {rel['id']}")
        
        except Exception as e:
            issues.append(f"Relationship validation error: {e}")
        
        if issues:
            print(f"❌ Found {len(issues)} issues:")
            for issue in issues:
                print(f"  • {issue}")
            return 1
        else:
            print("✅ Database validation passed")
            return 0


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Cross-Reference CLI for research paper analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process papers for cross-referencing
  crossref process papers.json --pdf-dir pdfs/ --output-dir results/
  
  # Add a single paper
  crossref add-paper paper123 --title "Deep Learning" --authors "Smith,Jones"
  
  # Export graph to Neo4j format
  crossref export --format neo4j --output-dir exports/
  
  # Show database statistics
  crossref stats --verbose
  
  # Find relationships for a paper
  crossref relationships paper123 --limit 5
  
  # Search for papers
  crossref search "neural networks" --limit 10
        """
    )
    
    # Global options
    parser.add_argument(
        '--database', '-d',
        default='crossref.db',
        help='Path to SQLite database (default: crossref.db)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process papers for cross-referencing')
    process_parser.add_argument('papers_file', help='JSON file containing paper metadata')
    process_parser.add_argument('--pdf-dir', help='Directory containing PDF files')
    process_parser.add_argument('--output-dir', default='data/crossref', help='Output directory')
    process_parser.add_argument('--similarity-threshold', type=float, default=0.7)
    process_parser.add_argument('--citation-threshold', type=float, default=0.5)
    process_parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2')
    process_parser.add_argument('--batch-size', type=int, default=10)
    process_parser.add_argument('--export-formats', default='json', help='Comma-separated export formats')
    
    # Add paper command
    add_parser = subparsers.add_parser('add-paper', help='Add a single paper')
    add_parser.add_argument('paper_id', help='Unique paper identifier')
    add_parser.add_argument('--paper-file', help='JSON file with paper metadata')
    add_parser.add_argument('--pdf-path', help='Path to PDF file')
    add_parser.add_argument('--title', help='Paper title')
    add_parser.add_argument('--authors', help='Comma-separated authors')
    add_parser.add_argument('--abstract', help='Paper abstract')
    add_parser.add_argument('--doi', help='DOI identifier')
    add_parser.add_argument('--arxiv-id', help='arXiv identifier')
    add_parser.add_argument('--year', type=int, help='Publication year')
    add_parser.add_argument('--keywords', help='Comma-separated keywords')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export cross-reference graph')
    export_parser.add_argument('--format', default='json', help='Export format (json,neo4j,networkx,edgelist)')
    export_parser.add_argument('--output-dir', default='exports', help='Output directory')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    # Relationships command
    rel_parser = subparsers.add_parser('relationships', help='Show paper relationships')
    rel_parser.add_argument('paper_id', help='Paper ID to show relationships for')
    rel_parser.add_argument('--limit', type=int, default=10, help='Maximum relationships to show')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for papers')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=20, help='Maximum results to show')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize new database')
    init_parser.add_argument('--force', action='store_true', help='Overwrite existing database')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate database integrity')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create CLI instance
    cli = CrossRefCLI()
    
    # Run command
    try:
        command_method = getattr(cli, f'cmd_{args.command.replace("-", "_")}')
        return command_method(args)
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
