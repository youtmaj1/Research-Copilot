#!/usr/bin/env python3
"""
Citation Tracker Demonstration

This script demonstrates the complete citation tracking pipeline:
1. Database setup and schema creation
2. Citation extraction from sample text
3. Citation resolution against a mock database
4. Graph building and analysis
5. Temporal analysis simulation
6. Export functionality

Run this to see the Citation Tracker in action!
"""

import sys
import tempfile
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from citation_tracker.extractor import CitationExtractor, ExtractedCitation
    from citation_tracker.resolver import CitationResolver, CitationMatch
    from citation_tracker.graph import CitationGraph, PaperNode, CitationEdge
    from citation_tracker.temporal import TimeSeriesAnalyzer, TrendingPaper
    from citation_tracker.exporter import GraphExporter
    from citation_tracker.database_schema import create_citation_tables, verify_schema
except ImportError as e:
    # Try direct imports if package import fails
    try:
        import extractor, resolver, graph as citation_graph, temporal, exporter, database_schema
        CitationExtractor = extractor.CitationExtractor
        ExtractedCitation = extractor.ExtractedCitation
        CitationResolver = resolver.CitationResolver
        CitationMatch = resolver.CitationMatch
        CitationGraph = citation_graph.CitationGraph
        PaperNode = citation_graph.PaperNode
        CitationEdge = citation_graph.CitationEdge
        TimeSeriesAnalyzer = temporal.TimeSeriesAnalyzer
        TrendingPaper = temporal.TrendingPaper
        GraphExporter = exporter.GraphExporter
        create_citation_tables = database_schema.create_citation_tables
        verify_schema = database_schema.verify_schema
    except ImportError as e2:
        print(f"Import error: {e2}")
        print("Available modules:")
        for f in Path(__file__).parent.glob("*.py"):
            if f.name != "__init__.py":
                print(f"  {f.name}")
        sys.exit(1)

def setup_demo_database(db_path):
    """Set up a demonstration database with sample papers."""
    print("🗄️  Setting up demonstration database...")
    
    # Create main papers table
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create papers table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT,
            year INTEGER,
            doi TEXT,
            arxiv_id TEXT,
            venue TEXT,
            abstract TEXT,
            url TEXT
        )
    """)
    
    # Insert sample papers representing a realistic research domain
    sample_papers = [
        ('attention2017', 'Attention Is All You Need', 'Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A.N.; Kaiser, L.; Polosukhin, I.', 2017, None, '1706.03762', 'NeurIPS', 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.'),
        
        ('bert2018', 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding', 'Devlin, J.; Chang, M.W.; Lee, K.; Toutanova, K.', 2018, None, '1810.04805', 'NAACL', 'We introduce a new language representation model called BERT.'),
        
        ('gpt2019', 'Language Models are Unsupervised Multitask Learners', 'Radford, A.; Wu, J.; Child, R.; Luan, D.; Amodei, D.; Sutskever, I.', 2019, None, None, 'OpenAI', 'Natural language processing tasks have benefited from the use of transfer learning.'),
        
        ('t52019', 'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer', 'Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; Liu, P.J.', 2019, None, '1910.10683', 'JMLR', 'Transfer learning has become a dominant approach in NLP.'),
        
        ('gpt32020', 'Language Models are Few-Shot Learners', 'Brown, T.B.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.', 2020, None, '2005.14165', 'NeurIPS', 'Recent work has demonstrated substantial gains on many NLP tasks.'),
        
        ('roberta2019', 'RoBERTa: A Robustly Optimized BERT Pretraining Approach', 'Liu, Y.; Ott, M.; Goyal, N.; Du, J.; Joshi, M.; Chen, D.; Levy, O.; Lewis, M.; Zettlemoyer, L.; Stoyanov, V.', 2019, None, '1907.11692', 'arXiv', 'Language model pretraining has led to significant performance gains.'),
        
        ('electra2020', 'ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators', 'Clark, K.; Luong, M.T.; Le, Q.V.; Manning, C.D.', 2020, None, '2003.10555', 'ICLR', 'Masked language modeling (MLM) pretraining methods such as BERT.'),
        
        ('albert2019', 'ALBERT: A Lite BERT for Self-supervised Learning of Language Representations', 'Lan, Z.; Chen, M.; Goodman, S.; Gimpel, K.; Sharma, P.; Soricut, R.', 2019, None, '1909.11942', 'ICLR', 'Increasing model size when pretraining natural language representations.'),
        
        ('xlnet2019', 'XLNet: Generalized Autoregressive Pretraining for Language Understanding', 'Yang, Z.; Dai, Z.; Yang, Y.; Carbonell, J.; Salakhutdinov, R.; Le, Q.V.', 2019, None, '1906.08237', 'NeurIPS', 'With the capability of modeling bidirectional contexts.'),
        
        ('longformer2020', 'Longformer: The Long-Document Transformer', 'Beltagy, I.; Peters, M.E.; Cohan, A.', 2020, None, '2004.05150', 'arXiv', 'Transformer-based models are unable to process long sequences.'),
    ]
    
    cursor.executemany("""
        INSERT OR REPLACE INTO papers (id, title, authors, year, doi, arxiv_id, venue, abstract)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, sample_papers)
    
    conn.commit()
    conn.close()
    
    # Create citation tracking schema
    success = create_citation_tables(db_path)
    if not success:
        print("❌ Failed to create citation tracking schema")
        return False
    
    print(f"✅ Database setup complete with {len(sample_papers)} sample papers")
    return True

def demonstrate_extraction():
    """Demonstrate citation extraction."""
    print("\n📄 Demonstrating Citation Extraction...")
    
    # Sample paper reference section (realistic ML paper)
    sample_paper_text = """
    References

    [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). 
    Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008). arXiv:1706.03762

    [2] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional 
    Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

    [3] Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Askell, A. (2020). 
    Language models are few-shot learners. Advances in neural information processing systems, 33, 1877-1901. 
    arXiv:2005.14165

    [4] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). 
    Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.

    [5] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). 
    RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692.

    [6] Clark, K., Luong, M. T., Le, Q. V., & Manning, C. D. (2020). ELECTRA: Pre-training text encoders 
    as discriminators rather than generators. arXiv preprint arXiv:2003.10555.
    """
    
    extractor = CitationExtractor()
    citations = extractor.extract_citations_from_text(sample_paper_text, "demo_paper_2024")
    
    print(f"✅ Extracted {len(citations)} citations:")
    for i, citation in enumerate(citations, 1):
        print(f"  {i}. {citation.raw_text[:80]}...")
        if citation.title:
            print(f"     Title: {citation.title}")
        if citation.authors:
            print(f"     Authors: {citation.authors[:50]}...")
        if citation.year:
            print(f"     Year: {citation.year}")
        if citation.doi:
            print(f"     DOI: {citation.doi}")
        if citation.arxiv_id:
            print(f"     arXiv: {citation.arxiv_id}")
        print(f"     Confidence: {citation.confidence:.3f}")
        print()
    
    return citations

def demonstrate_resolution(citations, db_path):
    """Demonstrate citation resolution."""
    print("🔍 Demonstrating Citation Resolution...")
    
    resolver = CitationResolver(db_path)
    matches = resolver.resolve_citations(citations)
    
    print(f"✅ Resolved {len(matches)} citations:")
    for i, match in enumerate(matches, 1):
        print(f"  {i}. Citation: {match.citation.raw_text[:60]}...")
        print(f"     Matched to: {match.paper_id}")
        print(f"     Match type: {match.match_type}")
        print(f"     Confidence: {match.confidence:.3f}")
        print()
    
    return matches

def demonstrate_graph_building(matches, db_path):
    """Demonstrate graph building and analysis."""
    print("📊 Demonstrating Graph Building...")
    
    graph = CitationGraph(db_path)
    
    # Load citation matches into graph
    added_count = graph.load_from_citation_matches(matches)
    print(f"✅ Added {added_count} citation relationships to graph")
    
    # Calculate graph metrics
    metrics = graph.calculate_metrics()
    print(f"\n📈 Graph Statistics:")
    print(f"  Nodes (papers): {metrics['nodes']}")
    print(f"  Edges (citations): {metrics['edges']}")
    print(f"  Graph density: {metrics['density']:.4f}")
    if 'connected_components' in metrics:
        print(f"  Connected components: {metrics['connected_components']}")
    if 'average_clustering' in metrics:
        print(f"  Average clustering: {metrics['average_clustering']:.4f}")
    
    # Show most cited papers
    most_cited = graph.get_most_cited_papers(5)
    if most_cited:
        print(f"\n🌟 Most Cited Papers:")
        for i, (paper_id, citations) in enumerate(most_cited, 1):
            paper_node = graph.paper_nodes.get(paper_id)
            title = paper_node.title if paper_node else "Unknown"
            print(f"  {i}. {title[:60]}... ({citations} citations)")
    
    return graph

def demonstrate_temporal_analysis(graph, db_path):
    """Demonstrate temporal analysis with simulated data."""
    print("\n⏰ Demonstrating Temporal Analysis...")
    
    temporal_analyzer = TimeSeriesAnalyzer(db_path, graph)
    
    # Simulate citation growth data for demonstration
    base_date = datetime.now() - timedelta(days=365)
    
    # Create realistic citation growth patterns for key papers
    citation_patterns = {
        'attention2017': [500, 750, 1200, 1800, 2500, 3200, 4000, 4800, 5500, 6200, 6800, 7400],  # Foundational paper
        'bert2018': [0, 50, 200, 500, 1000, 1800, 2800, 4200, 5800, 7500, 9200, 11000],        # Explosive growth
        'gpt32020': [0, 0, 0, 0, 100, 300, 600, 1200, 2000, 3200, 4800, 6500],                 # Recent breakthrough
        'roberta2019': [0, 20, 80, 200, 400, 700, 1100, 1600, 2200, 2800, 3400, 4000],         # Steady growth
    }
    
    print("📈 Recording simulated citation snapshots...")
    for paper_id, monthly_counts in citation_patterns.items():
        for month, count in enumerate(monthly_counts):
            timestamp = base_date + timedelta(days=month * 30)
            temporal_analyzer.record_citation_snapshot(paper_id, count, timestamp)
    
    # Analyze trends
    trending_papers = temporal_analyzer.analyze_trends(180)  # 6 months
    
    print(f"✅ Identified {len(trending_papers)} trending papers:")
    for i, paper in enumerate(trending_papers[:5], 1):
        print(f"  {i}. Paper ID: {paper.paper_id}")
        print(f"     Current citations: {paper.current_citations}")
        print(f"     Growth rate: {paper.growth_rate:.1f}%")
        print(f"     Trend type: {paper.trend_type}")
        print(f"     Velocity: {paper.velocity:.1f} citations/day")
        print()
    
    # Demonstrate forecasting
    if trending_papers:
        paper_id = trending_papers[0].paper_id
        forecast = temporal_analyzer.get_citation_forecast(paper_id, 90)
        print(f"🔮 Citation Forecast for {paper_id} (next 90 days):")
        print(f"  Current citations: {forecast.get('current_citations', 'N/A')}")
        print(f"  Forecasted total: {forecast.get('forecasted_total_citations', 'N/A')}")
        print(f"  Expected growth: {forecast.get('expected_new_citations', 'N/A')}")
        print(f"  Growth rate: {forecast.get('growth_rate', 'N/A'):.2f}%")
    
    return temporal_analyzer

def demonstrate_export(graph):
    """Demonstrate graph export functionality."""
    print("\n💾 Demonstrating Export Functionality...")
    
    exporter = GraphExporter()
    
    # Create temporary files for demonstration
    with tempfile.TemporaryDirectory() as temp_dir:
        # Export to JSON
        json_path = os.path.join(temp_dir, 'citation_graph.json')
        success = exporter.export_to_json(graph, json_path, include_metrics=True)
        if success:
            print(f"✅ JSON export successful: {json_path}")
            # Show file size
            size = os.path.getsize(json_path)
            print(f"   File size: {size:,} bytes")
            
            # Show sample of exported data
            with open(json_path, 'r') as f:
                data = json.load(f)
            print(f"   Contains: {len(data['graph']['nodes'])} nodes, {len(data['graph']['edges'])} edges")
        
        # Export to GraphML
        graphml_path = os.path.join(temp_dir, 'citation_graph.graphml')
        success = exporter.export_to_graphml(graph, graphml_path)
        if success:
            print(f"✅ GraphML export successful: {graphml_path}")
            size = os.path.getsize(graphml_path)
            print(f"   File size: {size:,} bytes")
        
        # Export to CSV
        csv_dir = os.path.join(temp_dir, 'csv_export')
        success = exporter.export_to_csv(graph, csv_dir)
        if success:
            print(f"✅ CSV export successful: {csv_dir}")
            nodes_file = os.path.join(csv_dir, 'papers.csv')
            edges_file = os.path.join(csv_dir, 'citations.csv')
            if os.path.exists(nodes_file):
                print(f"   Papers CSV: {os.path.getsize(nodes_file):,} bytes")
            if os.path.exists(edges_file):
                print(f"   Citations CSV: {os.path.getsize(edges_file):,} bytes")

def main():
    """Run the complete Citation Tracker demonstration."""
    print("🚀 Citation Tracker Demonstration")
    print("=" * 50)
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        # Step 1: Database setup
        if not setup_demo_database(db_path):
            return 1
        
        # Step 2: Citation extraction
        citations = demonstrate_extraction()
        if not citations:
            print("❌ No citations extracted")
            return 1
        
        # Step 3: Citation resolution
        matches = demonstrate_resolution(citations, db_path)
        if not matches:
            print("❌ No citations resolved")
            return 1
        
        # Step 4: Graph building
        graph = demonstrate_graph_building(matches, db_path)
        if not graph:
            print("❌ Graph building failed")
            return 1
        
        # Step 5: Temporal analysis
        temporal_analyzer = demonstrate_temporal_analysis(graph, db_path)
        
        # Step 6: Export functionality
        demonstrate_export(graph)
        
        print("\n🎉 Citation Tracker Demonstration Complete!")
        print("\nKey Features Demonstrated:")
        print("  ✅ Citation extraction from reference sections")
        print("  ✅ Fuzzy matching to resolve citations to papers")
        print("  ✅ Graph construction and network analysis")
        print("  ✅ Temporal analysis and trend detection")
        print("  ✅ Citation forecasting and growth modeling")
        print("  ✅ Multi-format export (JSON, GraphML, CSV)")
        print("  ✅ Comprehensive database schema")
        
        print(f"\n💾 Demo database available at: {db_path}")
        print("   (Will be cleaned up automatically)")
        
        return 0
    
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up temporary database
        try:
            os.unlink(db_path)
        except:
            pass

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
