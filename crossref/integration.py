"""
Integration Module for Cross-Reference System

Connects Module 3 (Cross-Referencer) with Module 1 (Collector) and Module 2 (Summarizer)
to provide seamless paper cross-referencing workflow.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import from other modules (assuming they exist)
try:
    from collector import PaperCollector, CollectorConfig  # Module 1
except ImportError:
    print("⚠️  Module 1 (Collector) not found - integration will be limited")
    PaperCollector = None
    CollectorConfig = None

try:
    from summarizer import PaperSummarizer, SummarizerConfig  # Module 2  
except ImportError:
    print("⚠️  Module 2 (Summarizer) not found - integration will be limited")
    PaperSummarizer = None
    SummarizerConfig = None

# Import Module 3 components
from .pipeline import CrossRefPipeline, CrossRefConfig
from .graph import CrossRefGraph
from .cli import CrossRefCLI

logger = logging.getLogger(__name__)


class IntegratedResearchPipeline:
    """
    Integrated pipeline combining collection, summarization, and cross-referencing.
    
    This class orchestrates the complete research workflow:
    1. Collect papers from various sources (Module 1)
    2. Generate summaries and extract insights (Module 2)
    3. Build cross-reference relationships and knowledge graphs (Module 3)
    """
    
    def __init__(
        self,
        collector_config: Optional[Dict] = None,
        summarizer_config: Optional[Dict] = None,
        crossref_config: Optional[Dict] = None,
        output_dir: str = "data/integrated"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.collector = None
        self.summarizer = None
        self.crossref_pipeline = None
        
        # Setup collector (Module 1)
        if PaperCollector and CollectorConfig:
            config = CollectorConfig(**(collector_config or {}))
            self.collector = PaperCollector(config)
            logger.info("Collector (Module 1) initialized")
        else:
            logger.warning("Collector (Module 1) not available")
        
        # Setup summarizer (Module 2)
        if PaperSummarizer and SummarizerConfig:
            config = SummarizerConfig(**(summarizer_config or {}))
            self.summarizer = PaperSummarizer(config)
            logger.info("Summarizer (Module 2) initialized")
        else:
            logger.warning("Summarizer (Module 2) not available")
        
        # Setup cross-referencer (Module 3)
        crossref_config = crossref_config or {}
        crossref_config.setdefault('output_dir', str(self.output_dir / 'crossref'))
        crossref_config.setdefault('database_path', str(self.output_dir / 'crossref.db'))
        
        self.crossref_config = CrossRefConfig(**crossref_config)
        self.crossref_pipeline = CrossRefPipeline(self.crossref_config)
        logger.info("Cross-Referencer (Module 3) initialized")
    
    def run_complete_pipeline(
        self,
        search_queries: List[str],
        max_papers_per_query: int = 50,
        include_pdfs: bool = True,
        generate_summaries: bool = True,
        build_knowledge_graph: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete integrated research pipeline.
        
        Args:
            search_queries: List of search queries for paper collection
            max_papers_per_query: Maximum papers to collect per query
            include_pdfs: Whether to download PDF files
            generate_summaries: Whether to generate paper summaries
            build_knowledge_graph: Whether to build cross-reference graph
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        logger.info("Starting complete research pipeline")
        
        results = {
            'start_time': datetime.now().isoformat(),
            'search_queries': search_queries,
            'papers_collected': 0,
            'papers_summarized': 0,
            'relationships_found': 0,
            'knowledge_graph': None,
            'output_files': []
        }
        
        # Step 1: Collect papers (Module 1)
        all_papers = {}
        pdf_paths = {}
        
        if self.collector:
            logger.info("Step 1: Collecting papers")
            
            for query in search_queries:
                try:
                    # Collect papers for this query
                    collected = self.collector.search_papers(
                        query, 
                        max_results=max_papers_per_query
                    )
                    
                    # Download PDFs if requested
                    if include_pdfs:
                        for paper_id, paper_data in collected.items():
                            if paper_data.get('pdf_url'):
                                pdf_path = self.collector.download_pdf(
                                    paper_data['pdf_url'],
                                    str(self.output_dir / 'pdfs' / f"{paper_id}.pdf")
                                )
                                if pdf_path:
                                    pdf_paths[paper_id] = pdf_path
                    
                    all_papers.update(collected)
                    logger.info(f"Query '{query}': collected {len(collected)} papers")
                
                except Exception as e:
                    logger.error(f"Failed to collect papers for query '{query}': {e}")
            
            results['papers_collected'] = len(all_papers)
            
            # Save collected papers
            papers_file = self.output_dir / 'collected_papers.json'
            with open(papers_file, 'w') as f:
                json.dump(all_papers, f, indent=2, default=str)
            results['output_files'].append(str(papers_file))
            
        else:
            logger.warning("Skipping paper collection - Module 1 not available")
            # Load papers from file if available
            papers_file = self.output_dir / 'collected_papers.json'
            if papers_file.exists():
                with open(papers_file, 'r') as f:
                    all_papers = json.load(f)
                logger.info(f"Loaded {len(all_papers)} papers from file")
        
        if not all_papers:
            logger.error("No papers available for processing")
            return results
        
        # Step 2: Generate summaries (Module 2)
        if generate_summaries and self.summarizer:
            logger.info("Step 2: Generating summaries")
            
            try:
                summaries = self.summarizer.summarize_papers(all_papers)
                
                # Add summaries to paper data
                for paper_id, summary_data in summaries.items():
                    if paper_id in all_papers:
                        all_papers[paper_id].update(summary_data)
                
                results['papers_summarized'] = len(summaries)
                
                # Save enriched papers
                enriched_file = self.output_dir / 'enriched_papers.json'
                with open(enriched_file, 'w') as f:
                    json.dump(all_papers, f, indent=2, default=str)
                results['output_files'].append(str(enriched_file))
                
                logger.info(f"Generated summaries for {len(summaries)} papers")
            
            except Exception as e:
                logger.error(f"Failed to generate summaries: {e}")
        
        else:
            logger.info("Skipping summarization")
        
        # Step 3: Build cross-reference relationships (Module 3)
        if build_knowledge_graph:
            logger.info("Step 3: Building knowledge graph")
            
            try:
                # Process papers through cross-reference pipeline
                graph = self.crossref_pipeline.process_papers(all_papers, pdf_paths)
                
                # Get statistics
                stats = self.crossref_pipeline.get_pipeline_statistics()
                results['relationships_found'] = stats['database']['total_relationships']
                results['knowledge_graph'] = {
                    'nodes': len(graph.nodes),
                    'edges': len(graph.edges),
                    'statistics': stats
                }
                
                logger.info(f"Built knowledge graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            
            except Exception as e:
                logger.error(f"Failed to build knowledge graph: {e}")
                results['knowledge_graph'] = {'error': str(e)}
        
        # Step 4: Generate integrated report
        report = self._generate_integrated_report(results, all_papers)
        report_file = self.output_dir / 'integrated_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        results['output_files'].append(str(report_file))
        
        results['end_time'] = datetime.now().isoformat()
        logger.info("Integrated research pipeline completed")
        
        return results
    
    def _generate_integrated_report(self, results: Dict, papers: Dict) -> Dict:
        """Generate integrated analysis report."""
        report = {
            'pipeline_summary': results,
            'collection_analysis': self._analyze_collection(papers),
            'cross_reference_analysis': self._analyze_cross_references(),
            'recommendations': self._generate_recommendations(papers)
        }
        
        return report
    
    def _analyze_collection(self, papers: Dict) -> Dict:
        """Analyze the collected papers."""
        if not papers:
            return {'error': 'No papers to analyze'}
        
        analysis = {
            'total_papers': len(papers),
            'years': {},
            'authors': {},
            'sources': {},
            'has_pdf': 0,
            'has_abstract': 0
        }
        
        for paper_id, paper_data in papers.items():
            # Year distribution
            year = paper_data.get('year') or paper_data.get('published_date', '')[:4]
            if year:
                try:
                    year = int(year)
                    analysis['years'][year] = analysis['years'].get(year, 0) + 1
                except ValueError:
                    pass
            
            # Author statistics
            authors = paper_data.get('authors', [])
            for author in authors:
                analysis['authors'][author] = analysis['authors'].get(author, 0) + 1
            
            # Source statistics
            source = paper_data.get('source', 'unknown')
            analysis['sources'][source] = analysis['sources'].get(source, 0) + 1
            
            # Content availability
            if paper_data.get('pdf_url') or paper_data.get('pdf_path'):
                analysis['has_pdf'] += 1
            
            if paper_data.get('abstract'):
                analysis['has_abstract'] += 1
        
        # Top items
        analysis['top_years'] = sorted(analysis['years'].items(), key=lambda x: x[1], reverse=True)[:5]
        analysis['top_authors'] = sorted(analysis['authors'].items(), key=lambda x: x[1], reverse=True)[:10]
        analysis['top_sources'] = sorted(analysis['sources'].items(), key=lambda x: x[1], reverse=True)[:5]
        
        return analysis
    
    def _analyze_cross_references(self) -> Dict:
        """Analyze cross-reference relationships."""
        try:
            stats = self.crossref_pipeline.get_pipeline_statistics()
            db_stats = stats['database']
            
            analysis = {
                'total_relationships': db_stats.get('total_relationships', 0),
                'citations': db_stats.get('total_citations', 0),
                'similarities': db_stats.get('total_similarities', 0),
                'relationship_types': db_stats.get('relationships_by_type', {})
            }
            
            # Add graph metrics if available
            if db_stats.get('total_relationships', 0) > 0:
                # Get all relationships
                relationships = self.crossref_pipeline.database.get_relationships()
                
                # Calculate network metrics
                papers_with_relations = set()
                for rel in relationships:
                    papers_with_relations.add(rel['source_paper'])
                    papers_with_relations.add(rel['target_paper'])
                
                analysis['connected_papers'] = len(papers_with_relations)
                analysis['average_relationships_per_paper'] = (
                    len(relationships) / len(papers_with_relations) if papers_with_relations else 0
                )
            
            return analysis
        
        except Exception as e:
            return {'error': f'Failed to analyze cross-references: {e}'}
    
    def _generate_recommendations(self, papers: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if not papers:
            return ["No papers collected - check search queries and data sources"]
        
        # Collection recommendations
        years = {}
        for paper_data in papers.values():
            year = paper_data.get('year')
            if year:
                years[year] = years.get(year, 0) + 1
        
        if years:
            latest_year = max(years.keys())
            if latest_year < datetime.now().year - 2:
                recommendations.append("Consider including more recent papers to get current research trends")
        
        # PDF availability
        pdf_count = sum(1 for p in papers.values() if p.get('pdf_url') or p.get('pdf_path'))
        if pdf_count < len(papers) * 0.5:
            recommendations.append("Low PDF availability may limit citation extraction - consider additional sources")
        
        # Cross-reference recommendations
        try:
            stats = self.crossref_pipeline.get_pipeline_statistics()
            rel_count = stats['database'].get('total_relationships', 0)
            
            if rel_count == 0:
                recommendations.append("No cross-references found - papers may be too diverse or lack citations")
            elif rel_count < len(papers) * 0.3:
                recommendations.append("Few cross-references found - consider papers from related research areas")
        except:
            pass
        
        # General recommendations
        if len(papers) < 20:
            recommendations.append("Small paper collection - consider expanding search queries for better analysis")
        
        if not recommendations:
            recommendations.append("Good paper collection with diverse sources and strong cross-references")
        
        return recommendations
    
    def export_for_visualization(
        self, 
        output_path: str,
        include_node_details: bool = True,
        include_summaries: bool = True
    ):
        """Export data for external visualization tools."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get papers and relationships
        papers = self.crossref_pipeline.database.get_paper_metadata()
        relationships = self.crossref_pipeline.database.get_relationships()
        
        # Create visualization data
        viz_data = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'created': datetime.now().isoformat(),
                'total_nodes': len(papers),
                'total_edges': len(relationships)
            }
        }
        
        # Add nodes (papers)
        for paper_id, paper_data in papers.items():
            node = {
                'id': paper_id,
                'label': paper_data.get('title', paper_id)[:50] + '...',
                'title': paper_data.get('title', ''),
                'year': paper_data.get('year'),
                'authors': paper_data.get('authors', [])
            }
            
            if include_node_details:
                node.update({
                    'doi': paper_data.get('doi'),
                    'keywords': paper_data.get('keywords', [])
                })
            
            if include_summaries:
                node.update({
                    'abstract': paper_data.get('abstract', '')[:200] + '...',
                    'summary': paper_data.get('summary', '')
                })
            
            viz_data['nodes'].append(node)
        
        # Add edges (relationships)
        for rel in relationships:
            edge = {
                'source': rel['source_paper'],
                'target': rel['target_paper'],
                'relationship': rel['relation'],
                'weight': rel['score'],
                'confidence': rel['confidence']
            }
            
            viz_data['edges'].append(edge)
        
        # Save visualization data
        viz_file = output_path / 'visualization_data.json'
        with open(viz_file, 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        # Create Cytoscape.js format
        cytoscape_data = {
            'elements': {
                'nodes': [{'data': node} for node in viz_data['nodes']],
                'edges': [{'data': edge} for edge in viz_data['edges']]
            }
        }
        
        cytoscape_file = output_path / 'cytoscape_data.json'
        with open(cytoscape_file, 'w') as f:
            json.dump(cytoscape_data, f, indent=2)
        
        logger.info(f"Exported visualization data to {output_path}")
        return [str(viz_file), str(cytoscape_file)]
    
    def get_paper_insights(self, paper_id: str) -> Dict[str, Any]:
        """Get comprehensive insights for a specific paper."""
        # Get paper metadata
        papers = self.crossref_pipeline.database.get_paper_metadata([paper_id])
        if paper_id not in papers:
            return {'error': f'Paper {paper_id} not found'}
        
        paper_data = papers[paper_id]
        
        # Get relationships
        relationships = self.crossref_pipeline.get_paper_relationships(paper_id)
        
        # Build insights
        insights = {
            'paper_info': paper_data,
            'relationships': relationships,
            'network_position': self._analyze_paper_position(paper_id),
            'research_impact': self._analyze_research_impact(paper_id, relationships),
            'related_topics': self._find_related_topics(paper_id)
        }
        
        return insights
    
    def _analyze_paper_position(self, paper_id: str) -> Dict:
        """Analyze a paper's position in the research network."""
        try:
            # Get all papers and build a temporary graph for analysis
            papers = self.crossref_pipeline.database.get_paper_metadata()
            relationships = self.crossref_pipeline.database.get_relationships()
            
            # Build graph
            from .graph import CrossRefGraph
            graph = CrossRefGraph()
            graph.add_papers(papers)
            
            for rel in relationships:
                graph.add_edge(
                    rel['source_paper'],
                    rel['target_paper'],
                    rel['relation'],
                    rel['score']
                )
            
            # Compute centrality metrics
            centrality = graph.compute_centrality_metrics()
            
            position = {}
            for metric, values in centrality.items():
                if paper_id in values:
                    position[metric] = values[paper_id]
            
            # Add ranking
            for metric, values in centrality.items():
                if paper_id in values:
                    sorted_papers = sorted(values.items(), key=lambda x: x[1], reverse=True)
                    rank = next(i for i, (pid, _) in enumerate(sorted_papers) if pid == paper_id) + 1
                    position[f'{metric}_rank'] = rank
            
            return position
        
        except Exception as e:
            return {'error': f'Failed to analyze position: {e}'}
    
    def _analyze_research_impact(self, paper_id: str, relationships: Dict) -> Dict:
        """Analyze research impact of a paper."""
        impact = {
            'cited_by_count': 0,
            'cites_count': 0,
            'similar_papers_count': 0,
            'collaboration_count': 0
        }
        
        for rel_type, rels in relationships.items():
            if rel_type == 'cites':
                # Count papers that cite this paper
                citing_this = sum(1 for r in rels if r['target_paper'] == paper_id)
                cited_by_this = sum(1 for r in rels if r['source_paper'] == paper_id)
                
                impact['cited_by_count'] = citing_this
                impact['cites_count'] = cited_by_this
            
            elif rel_type == 'similar_to':
                impact['similar_papers_count'] = len(rels)
            
            elif rel_type == 'same_author':
                impact['collaboration_count'] = len(rels)
        
        # Calculate impact score
        impact['impact_score'] = (
            impact['cited_by_count'] * 2 +
            impact['cites_count'] * 1 +
            impact['similar_papers_count'] * 0.5 +
            impact['collaboration_count'] * 0.3
        )
        
        return impact
    
    def _find_related_topics(self, paper_id: str) -> List[str]:
        """Find related research topics for a paper."""
        try:
            # Get similar papers
            relationships = self.crossref_pipeline.get_paper_relationships(paper_id)
            similar_papers = relationships.get('similar_to', [])
            
            if not similar_papers:
                return []
            
            # Get similar paper IDs
            similar_ids = [r['target_paper'] for r in similar_papers[:10]]
            
            # Get metadata for similar papers
            similar_metadata = self.crossref_pipeline.database.get_paper_metadata(similar_ids)
            
            # Extract topics from titles and keywords
            topics = set()
            for metadata in similar_metadata.values():
                # Add keywords
                topics.update(metadata.get('keywords', []))
                
                # Extract key terms from titles
                title = metadata.get('title', '').lower()
                key_terms = [
                    'machine learning', 'deep learning', 'neural networks',
                    'computer vision', 'natural language processing',
                    'artificial intelligence', 'data mining', 'robotics',
                    'quantum computing', 'bioinformatics'
                ]
                
                for term in key_terms:
                    if term in title:
                        topics.add(term)
            
            return list(topics)[:10]
        
        except Exception as e:
            logger.error(f"Failed to find related topics: {e}")
            return []


def create_integrated_pipeline(
    output_dir: str = "data/integrated",
    **config_overrides
) -> IntegratedResearchPipeline:
    """
    Create an integrated research pipeline with default configurations.
    
    Args:
        output_dir: Output directory for all pipeline results
        **config_overrides: Override default configurations
        
    Returns:
        Configured integrated pipeline
    """
    collector_config = config_overrides.get('collector_config', {})
    summarizer_config = config_overrides.get('summarizer_config', {})
    crossref_config = config_overrides.get('crossref_config', {})
    
    return IntegratedResearchPipeline(
        collector_config=collector_config,
        summarizer_config=summarizer_config,
        crossref_config=crossref_config,
        output_dir=output_dir
    )


class IntegratedCLI:
    """Extended CLI for integrated research operations."""
    
    def __init__(self):
        self.crossref_cli = CrossRefCLI()
    
    def cmd_research(self, args):
        """Run complete research pipeline."""
        print(f"🔬 Starting integrated research pipeline")
        
        # Parse search queries
        queries = args.queries.split(',')
        queries = [q.strip() for q in queries]
        
        print(f"📝 Search queries: {', '.join(queries)}")
        
        # Create integrated pipeline
        pipeline = create_integrated_pipeline(
            output_dir=args.output_dir,
            crossref_config={'similarity_threshold': args.similarity_threshold}
        )
        
        # Run pipeline
        results = pipeline.run_complete_pipeline(
            search_queries=queries,
            max_papers_per_query=args.max_papers,
            include_pdfs=args.include_pdfs,
            generate_summaries=args.generate_summaries,
            build_knowledge_graph=args.build_graph
        )
        
        # Print results
        print(f"\n✅ Pipeline completed!")
        print(f"📄 Papers collected: {results['papers_collected']}")
        print(f"📊 Papers summarized: {results['papers_summarized']}")
        print(f"🔗 Relationships found: {results['relationships_found']}")
        
        if results['knowledge_graph']:
            kg = results['knowledge_graph']
            print(f"🕸️  Knowledge graph: {kg['nodes']} nodes, {kg['edges']} edges")
        
        print(f"📁 Output files:")
        for file_path in results['output_files']:
            print(f"  • {file_path}")
        
        return 0
    
    def cmd_insights(self, args):
        """Get insights for a specific paper."""
        pipeline = create_integrated_pipeline(output_dir=args.output_dir)
        insights = pipeline.get_paper_insights(args.paper_id)
        
        if 'error' in insights:
            print(f"❌ {insights['error']}")
            return 1
        
        print(f"📄 Paper Insights: {args.paper_id}")
        print("=" * 50)
        
        # Paper info
        info = insights['paper_info']
        print(f"Title: {info.get('title', 'N/A')}")
        print(f"Authors: {', '.join(info.get('authors', []))}")
        print(f"Year: {info.get('year', 'N/A')}")
        
        # Network position
        if 'network_position' in insights:
            pos = insights['network_position']
            print(f"\n🌐 Network Position:")
            for metric, value in pos.items():
                if not metric.endswith('_rank'):
                    print(f"  {metric}: {value:.3f}")
        
        # Research impact
        if 'research_impact' in insights:
            impact = insights['research_impact']
            print(f"\n📈 Research Impact:")
            print(f"  Cited by: {impact['cited_by_count']} papers")
            print(f"  Cites: {impact['cites_count']} papers")
            print(f"  Similar papers: {impact['similar_papers_count']}")
            print(f"  Impact score: {impact['impact_score']:.2f}")
        
        # Related topics
        if insights.get('related_topics'):
            print(f"\n🏷️  Related Topics:")
            for topic in insights['related_topics'][:5]:
                print(f"  • {topic}")
        
        return 0


if __name__ == "__main__":
    # Example usage
    print("🔬 Integrated Research Pipeline")
    print("=" * 40)
    
    # Create sample configuration
    config = {
        'crossref_config': {
            'similarity_threshold': 0.6,
            'citation_confidence_threshold': 0.5
        }
    }
    
    # Create pipeline
    pipeline = create_integrated_pipeline(**config)
    
    print("✅ Integrated pipeline created successfully")
    print("🎯 Ready for research workflow integration")
