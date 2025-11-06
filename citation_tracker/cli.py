#!/usr/bin/env python3
"""
Citation Tracker Command Line Interface

This CLI provides access to all citation tracking functionality including:
- Citation extraction from papers
- Citation resolution and matching
- Graph building and analysis
- Temporal analysis and trend detection
- Export functionality

Usage:
    python cli.py extract --paper-path paper.pdf --output citations.json
    python cli.py resolve --citations citations.json --database papers.db
    python cli.py graph --build --database papers.db --export graph.json
    python cli.py analyze --temporal --paper-id paper123 --days 30
    python cli.py export --format json --output citation_graph.json
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from extractor import CitationExtractor, ExtractedCitation
    from resolver import CitationResolver, CitationMatch
    from graph import CitationGraph, PaperNode, CitationEdge
    from temporal import TimeSeriesAnalyzer, TrendingPaper
    from exporter import GraphExporter
    from database_schema import create_citation_tables, verify_schema
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the citation_tracker directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class CitationTrackerCLI:
    """Command-line interface for citation tracking functionality."""
    
    def __init__(self):
        self.extractor = None
        self.resolver = None
        self.graph = None
        self.temporal_analyzer = None
        self.exporter = None
    
    def initialize_components(self, database_path="papers.db"):
        """Initialize all components with database path."""
        logger.info(f"Initializing components with database: {database_path}")
        
        # Check if database exists
        if not os.path.exists(database_path):
            logger.error(f"Database not found: {database_path}")
            return False
        
        # Initialize components
        self.extractor = CitationExtractor()
        self.resolver = CitationResolver(database_path)
        self.graph = CitationGraph(database_path)
        self.temporal_analyzer = TimeSeriesAnalyzer(database_path, self.graph)
        self.exporter = GraphExporter()
        
        return True
    
    def extract_citations(self, args):
        """Extract citations from a paper."""
        if not self.extractor:
            self.extractor = CitationExtractor()
        
        logger.info(f"Extracting citations from: {args.paper_path}")
        
        if not os.path.exists(args.paper_path):
            logger.error(f"Paper file not found: {args.paper_path}")
            return False
        
        # Determine source paper ID
        source_paper_id = args.source_id or Path(args.paper_path).stem
        
        # Extract citations based on file type
        if args.paper_path.lower().endswith('.pdf'):
            citations = self.extractor.extract_citations_from_pdf(args.paper_path, source_paper_id)
        else:
            # Assume text file
            with open(args.paper_path, 'r', encoding='utf-8') as f:
                text = f.read()
            citations = self.extractor.extract_citations_from_text(text, source_paper_id)
        
        logger.info(f"Extracted {len(citations)} citations")
        
        # Output results
        if args.output:
            output_data = {
                'source_paper_id': source_paper_id,
                'source_path': args.paper_path,
                'extraction_timestamp': datetime.now().isoformat(),
                'citation_count': len(citations),
                'citations': [
                    {
                        'raw_text': c.raw_text,
                        'title': c.title,
                        'authors': c.authors,
                        'year': c.year,
                        'doi': c.doi,
                        'arxiv_id': c.arxiv_id,
                        'venue': c.venue,
                        'confidence': c.confidence
                    }
                    for c in citations
                ]
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Citations saved to: {args.output}")
        else:
            # Print to stdout
            for i, citation in enumerate(citations, 1):
                print(f"\n=== Citation {i} ===")
                print(f"Raw text: {citation.raw_text}")
                if citation.title:
                    print(f"Title: {citation.title}")
                if citation.authors:
                    print(f"Authors: {citation.authors}")
                if citation.year:
                    print(f"Year: {citation.year}")
                if citation.doi:
                    print(f"DOI: {citation.doi}")
                if citation.arxiv_id:
                    print(f"arXiv: {citation.arxiv_id}")
                print(f"Confidence: {citation.confidence:.3f}")
        
        return True
    
    def resolve_citations(self, args):
        """Resolve citations against database."""
        if not self.initialize_components(args.database):
            return False
        
        # Load citations from file or extract from paper
        if args.citations_file:
            logger.info(f"Loading citations from: {args.citations_file}")
            with open(args.citations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            citations = []
            for c_data in data.get('citations', []):
                citation = ExtractedCitation(
                    raw_text=c_data['raw_text'],
                    source_paper_id=data.get('source_paper_id', 'unknown'),
                    title=c_data.get('title'),
                    authors=c_data.get('authors'),
                    year=c_data.get('year'),
                    doi=c_data.get('doi'),
                    arxiv_id=c_data.get('arxiv_id'),
                    venue=c_data.get('venue'),
                    confidence=c_data.get('confidence', 0.0)
                )
                citations.append(citation)
        
        elif args.paper_path:
            # Extract and resolve in one step
            source_paper_id = args.source_id or Path(args.paper_path).stem
            
            if args.paper_path.lower().endswith('.pdf'):
                citations = self.extractor.extract_citations_from_pdf(args.paper_path, source_paper_id)
            else:
                with open(args.paper_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                citations = self.extractor.extract_citations_from_text(text, source_paper_id)
        
        else:
            logger.error("Must provide either --citations-file or --paper-path")
            return False
        
        logger.info(f"Resolving {len(citations)} citations against database")
        
        # Resolve citations
        matches = self.resolver.resolve_citations(citations)
        
        logger.info(f"Found {len(matches)} matches")
        
        # Output results
        output_data = {
            'resolution_timestamp': datetime.now().isoformat(),
            'input_citations': len(citations),
            'resolved_matches': len(matches),
            'matches': []
        }
        
        for match in matches:
            match_data = {
                'citation_text': match.citation.raw_text,
                'matched_paper_id': match.paper_id,
                'match_type': match.match_type,
                'confidence': match.confidence,
                'paper_title': getattr(match, 'paper_title', None),
                'paper_authors': getattr(match, 'paper_authors', None),
                'paper_year': getattr(match, 'paper_year', None)
            }
            output_data['matches'].append(match_data)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Resolution results saved to: {args.output}")
        else:
            # Print summary
            print(f"\n=== Resolution Summary ===")
            print(f"Input citations: {len(citations)}")
            print(f"Resolved matches: {len(matches)}")
            print(f"Success rate: {len(matches)/len(citations)*100:.1f}%")
            
            if matches:
                print(f"\n=== Top Matches ===")
                for i, match in enumerate(matches[:10], 1):
                    print(f"{i}. {match.citation.raw_text[:80]}...")
                    print(f"   -> {match.paper_id} (confidence: {match.confidence:.3f})")
        
        return True
    
    def build_graph(self, args):
        """Build citation graph."""
        if not self.initialize_components(args.database):
            return False
        
        logger.info("Building citation graph")
        
        if args.from_matches:
            # Load matches from file
            logger.info(f"Loading matches from: {args.from_matches}")
            with open(args.from_matches, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to CitationMatch objects (simplified)
            matches = []
            for match_data in data.get('matches', []):
                citation = ExtractedCitation(
                    raw_text=match_data['citation_text'],
                    source_paper_id='unknown'  # Would need to track this properly
                )
                match = CitationMatch(
                    citation=citation,
                    paper_id=match_data['matched_paper_id'],
                    match_type=match_data['match_type'],
                    confidence=match_data['confidence']
                )
                matches.append(match)
            
            # Load matches into graph
            added_count = self.graph.load_from_citation_matches(matches)
            logger.info(f"Added {added_count} citations to graph")
        
        # Calculate graph metrics
        metrics = self.graph.calculate_metrics()
        
        print(f"\n=== Graph Statistics ===")
        print(f"Nodes (papers): {metrics['nodes']}")
        print(f"Edges (citations): {metrics['edges']}")
        print(f"Graph density: {metrics['density']:.4f}")
        if 'connected_components' in metrics:
            print(f"Connected components: {metrics['connected_components']}")
        if 'average_clustering' in metrics:
            print(f"Average clustering: {metrics['average_clustering']:.4f}")
        
        # Show most cited papers
        most_cited = self.graph.get_most_cited_papers(10)
        if most_cited:
            print(f"\n=== Most Cited Papers ===")
            for i, (paper_id, citations) in enumerate(most_cited, 1):
                paper_node = self.graph.paper_nodes.get(paper_id)
                title = paper_node.title if paper_node else "Unknown"
                print(f"{i}. {title[:60]}... ({citations} citations)")
        
        # Export if requested
        if args.export:
            success = self.exporter.export_to_json(self.graph, args.export)
            if success:
                logger.info(f"Graph exported to: {args.export}")
            else:
                logger.error("Failed to export graph")
        
        return True
    
    def analyze_temporal(self, args):
        """Perform temporal analysis."""
        if not self.initialize_components(args.database):
            return False
        
        logger.info("Performing temporal analysis")
        
        if args.paper_id:
            # Analyze specific paper
            print(f"\n=== Temporal Analysis for {args.paper_id} ===")
            
            # Get citation forecast
            try:
                forecast = self.temporal_analyzer.get_citation_forecast(args.paper_id, args.days or 30)
                print(f"Current citations: {forecast.get('current_citations', 'N/A')}")
                print(f"Forecasted citations ({args.days or 30} days): {forecast.get('forecasted_total_citations', 'N/A')}")
                print(f"Growth rate: {forecast.get('growth_rate', 'N/A')}")
            except Exception as e:
                logger.warning(f"Could not generate forecast: {e}")
        
        else:
            # Analyze trending papers
            days = args.days or 30
            trending_papers = self.temporal_analyzer.analyze_trends(days)
            
            print(f"\n=== Trending Papers (last {days} days) ===")
            if trending_papers:
                for i, paper in enumerate(trending_papers[:10], 1):
                    print(f"{i}. {paper.paper_id}")
                    print(f"   Growth rate: {paper.growth_rate:.2f}%")
                    print(f"   Trend type: {paper.trend_type}")
                    print(f"   Citations: {paper.current_citations}")
            else:
                print("No trending papers found (may need historical data)")
        
        return True
    
    def export_graph(self, args):
        """Export citation graph in various formats."""
        if not self.initialize_components(args.database):
            return False
        
        if not self.exporter:
            self.exporter = GraphExporter()
        
        logger.info(f"Exporting graph in {args.format.upper()} format")
        
        # Export based on format
        success = False
        if args.format == 'json':
            success = self.exporter.export_to_json(self.graph, args.output, include_metrics=True)
        elif args.format == 'graphml':
            success = self.exporter.export_to_graphml(self.graph, args.output)
        elif args.format == 'csv':
            # Create directory if it doesn't exist
            os.makedirs(args.output, exist_ok=True)
            success = self.exporter.export_to_csv(self.graph, args.output)
        elif args.format == 'neo4j':
            if args.neo4j_uri and args.neo4j_user and args.neo4j_password:
                success = self.exporter.export_to_neo4j(
                    self.graph, 
                    args.neo4j_uri, 
                    args.neo4j_user, 
                    args.neo4j_password
                )
            else:
                logger.error("Neo4j export requires --neo4j-uri, --neo4j-user, and --neo4j-password")
                return False
        
        if success:
            logger.info(f"Export completed successfully to: {args.output}")
        else:
            logger.error("Export failed")
        
        return success
    
    def setup_database(self, args):
        """Set up or verify citation tracking database schema."""
        logger.info(f"Setting up database schema: {args.database}")
        
        # Create citation tables
        success = create_citation_tables(args.database)
        if success:
            logger.info("Citation tracking schema created successfully")
        else:
            logger.error("Failed to create citation tracking schema")
            return False
        
        # Verify schema
        schema_info = verify_schema(args.database)
        
        if schema_info.get('schema_exists'):
            logger.info("Schema verification passed")
            print(f"\n=== Database Setup Complete ===")
            print(f"Database: {args.database}")
            print(f"Tables created: {len(schema_info.get('existing_tables', []))}")
            print(f"Schema version: {schema_info.get('version', 'Unknown')}")
        else:
            logger.error("Schema verification failed")
            missing = schema_info.get('missing_tables', [])
            if missing:
                print(f"Missing tables: {missing}")
        
        return schema_info.get('schema_exists', False)

def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Citation Tracker - Extract, resolve, and analyze academic citations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract citations from a PDF
  python cli.py extract --paper-path paper.pdf --output citations.json
  
  # Resolve citations against database
  python cli.py resolve --citations-file citations.json --database papers.db --output matches.json
  
  # Build citation graph
  python cli.py graph --database papers.db --export graph.json
  
  # Analyze trending papers
  python cli.py analyze --database papers.db --days 30
  
  # Export graph to GraphML
  python cli.py export --database papers.db --format graphml --output graph.graphml
  
  # Set up database schema
  python cli.py setup --database citations.db
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract citations from papers')
    extract_parser.add_argument('--paper-path', required=True, help='Path to paper file (PDF or text)')
    extract_parser.add_argument('--source-id', help='Source paper ID (default: filename)')
    extract_parser.add_argument('--output', help='Output JSON file (default: stdout)')
    
    # Resolve command
    resolve_parser = subparsers.add_parser('resolve', help='Resolve citations against database')
    resolve_parser.add_argument('--database', required=True, help='Path to papers database')
    resolve_group = resolve_parser.add_mutually_exclusive_group(required=True)
    resolve_group.add_argument('--citations-file', help='JSON file with extracted citations')
    resolve_group.add_argument('--paper-path', help='Paper file to extract and resolve')
    resolve_parser.add_argument('--source-id', help='Source paper ID (for --paper-path)')
    resolve_parser.add_argument('--output', help='Output JSON file (default: stdout)')
    
    # Graph command
    graph_parser = subparsers.add_parser('graph', help='Build and analyze citation graph')
    graph_parser.add_argument('--database', required=True, help='Path to papers database')
    graph_parser.add_argument('--from-matches', help='Load graph from matches JSON file')
    graph_parser.add_argument('--export', help='Export graph to JSON file')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Perform temporal analysis')
    analyze_parser.add_argument('--database', required=True, help='Path to papers database')
    analyze_parser.add_argument('--paper-id', help='Specific paper ID to analyze')
    analyze_parser.add_argument('--days', type=int, default=30, help='Time window in days')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export citation graph')
    export_parser.add_argument('--database', required=True, help='Path to papers database')
    export_parser.add_argument('--format', required=True, 
                              choices=['json', 'graphml', 'csv', 'neo4j'],
                              help='Export format')
    export_parser.add_argument('--output', required=True, help='Output file or directory')
    export_parser.add_argument('--neo4j-uri', help='Neo4j database URI')
    export_parser.add_argument('--neo4j-user', help='Neo4j username')
    export_parser.add_argument('--neo4j-password', help='Neo4j password')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up citation tracking database')
    setup_parser.add_argument('--database', required=True, help='Path to database file')
    
    return parser

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = CitationTrackerCLI()
    
    try:
        if args.command == 'extract':
            success = cli.extract_citations(args)
        elif args.command == 'resolve':
            success = cli.resolve_citations(args)
        elif args.command == 'graph':
            success = cli.build_graph(args)
        elif args.command == 'analyze':
            success = cli.analyze_temporal(args)
        elif args.command == 'export':
            success = cli.export_graph(args)
        elif args.command == 'setup':
            success = cli.setup_database(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        return 0 if success else 1
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
