"""
Cross-Reference Pipeline

Orchestrates the complete cross-referencing workflow:
1. Citation extraction from PDFs
2. Semantic similarity computation
3. Knowledge graph construction
4. Database storage
"""

import logging
import sqlite3
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import pickle

from .citation_extractor import CitationExtractor, CitationMatch
from .similarity import SimilarityEngine, SimilarityResult
from .graph import CrossRefGraph, GraphExporter, GraphNode, GraphEdge

logger = logging.getLogger(__name__)


@dataclass
class CrossRefConfig:
    """Configuration for cross-reference pipeline."""
    # Database
    database_path: str = "crossref.db"
    
    # Citation extraction
    citation_confidence_threshold: float = 0.5
    
    # Similarity computation
    similarity_threshold: float = 0.7
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_top_k: int = 10
    
    # Graph construction
    include_author_relationships: bool = True
    include_topic_clusters: bool = True
    topic_similarity_threshold: float = 0.8
    
    # Output paths
    output_dir: str = "data/crossref"
    graph_export_formats: List[str] = None  # ['json', 'neo4j', 'networkx']
    
    # Processing
    batch_size: int = 10
    save_intermediate: bool = True
    
    def __post_init__(self):
        if self.graph_export_formats is None:
            self.graph_export_formats = ['json']


class CrossRefDatabase:
    """Database operations for cross-reference data."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with schema."""
        # Read schema from file
        schema_path = Path(__file__).parent / "schema.sql"
        
        if not schema_path.exists():
            logger.error(f"Schema file not found: {schema_path}")
            return
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema_sql)
        
        logger.info(f"Initialized database: {self.db_path}")
    
    def store_paper_metadata(self, papers: Dict[str, Dict]):
        """Store paper metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for paper_id, paper_data in papers.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO papers_metadata 
                    (paper_id, title, authors, year, doi, arxiv_id, abstract, keywords, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    paper_id,
                    paper_data.get('title', ''),
                    json.dumps(paper_data.get('authors', [])),
                    self._extract_year(paper_data.get('published_date')),
                    paper_data.get('doi'),
                    paper_data.get('arxiv_id'),
                    paper_data.get('abstract', ''),
                    json.dumps(paper_data.get('keywords', [])),
                    json.dumps(paper_data)
                ))
            
            conn.commit()
        
        logger.info(f"Stored metadata for {len(papers)} papers")
    
    def store_citations(self, citations: List[CitationMatch]):
        """Store citation relationships."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Store citation details
            for match in citations:
                cursor.execute("""
                    INSERT OR REPLACE INTO citations
                    (citing_paper, cited_paper, raw_citation, extracted_title, 
                     extracted_authors, extracted_year, extracted_doi, extracted_arxiv,
                     match_type, match_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match.source_paper_id,
                    match.cited_paper_id,
                    match.citation.raw_text,
                    match.citation.title,
                    json.dumps(match.citation.authors),
                    match.citation.year,
                    match.citation.doi,
                    match.citation.arxiv_id,
                    match.match_type,
                    match.confidence
                ))
                
                # Store in crossref_relationships
                cursor.execute("""
                    INSERT OR REPLACE INTO crossref_relationships
                    (source_paper, relation, target_paper, score, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    match.source_paper_id,
                    'cites',
                    match.cited_paper_id,
                    1.0,
                    match.confidence,
                    json.dumps({
                        'match_type': match.match_type,
                        'citation_text': match.citation.raw_text[:200]
                    })
                ))
            
            conn.commit()
        
        logger.info(f"Stored {len(citations)} citation relationships")
    
    def store_similarities(self, similarities: List[SimilarityResult]):
        """Store similarity relationships."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for result in similarities:
                # Ensure consistent ordering for paper pairs
                paper1, paper2 = sorted([result.source_paper_id, result.target_paper_id])
                
                cursor.execute("""
                    INSERT OR REPLACE INTO similarities
                    (paper1, paper2, similarity_score, similarity_type, embedding_model, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    paper1,
                    paper2,
                    result.similarity_score,
                    result.match_type,
                    "unknown",  # Will be filled by pipeline
                    json.dumps(result.metadata)
                ))
                
                # Store in crossref_relationships (both directions)
                for source, target in [(paper1, paper2), (paper2, paper1)]:
                    cursor.execute("""
                        INSERT OR REPLACE INTO crossref_relationships
                        (source_paper, relation, target_paper, score, confidence, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        source,
                        'similar_to',
                        target,
                        result.similarity_score,
                        result.similarity_score,
                        json.dumps({
                            'similarity_type': result.match_type,
                            'embedding_model': "unknown"
                        })
                    ))
            
            conn.commit()
        
        logger.info(f"Stored {len(similarities)} similarity relationships")
    
    def store_author_relationships(self, author_rels: List[Tuple[str, str, float, List[str]]]):
        """Store author collaboration relationships."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for paper1, paper2, overlap, shared_authors in author_rels:
                # Ensure consistent ordering
                p1, p2 = sorted([paper1, paper2])
                
                cursor.execute("""
                    INSERT OR REPLACE INTO author_relationships
                    (paper1, paper2, shared_authors, author_overlap)
                    VALUES (?, ?, ?, ?)
                """, (
                    p1, p2,
                    json.dumps(shared_authors),
                    overlap
                ))
                
                # Store in crossref_relationships (both directions)
                for source, target in [(p1, p2), (p2, p1)]:
                    cursor.execute("""
                        INSERT OR REPLACE INTO crossref_relationships
                        (source_paper, relation, target_paper, score, confidence, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        source,
                        'same_author',
                        target,
                        overlap,
                        1.0,
                        json.dumps({'shared_authors': shared_authors})
                    ))
            
            conn.commit()
        
        logger.info(f"Stored {len(author_rels)} author relationships")
    
    def get_paper_metadata(self, paper_ids: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Retrieve paper metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if paper_ids:
                placeholders = ','.join('?' * len(paper_ids))
                cursor.execute(f"""
                    SELECT paper_id, title, authors, year, doi, arxiv_id, abstract, keywords, metadata
                    FROM papers_metadata
                    WHERE paper_id IN ({placeholders})
                """, paper_ids)
            else:
                cursor.execute("""
                    SELECT paper_id, title, authors, year, doi, arxiv_id, abstract, keywords, metadata
                    FROM papers_metadata
                """)
            
            papers = {}
            for row in cursor.fetchall():
                paper_id, title, authors, year, doi, arxiv_id, abstract, keywords, metadata = row
                
                papers[paper_id] = {
                    'title': title,
                    'authors': json.loads(authors) if authors else [],
                    'year': year,
                    'doi': doi,
                    'arxiv_id': arxiv_id,
                    'abstract': abstract,
                    'keywords': json.loads(keywords) if keywords else [],
                    'metadata': json.loads(metadata) if metadata else {}
                }
            
            return papers
    
    def get_relationships(
        self, 
        paper_id: Optional[str] = None,
        relation_type: Optional[str] = None
    ) -> List[Dict]:
        """Get cross-reference relationships."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM crossref_relationships WHERE 1=1"
            params = []
            
            if paper_id:
                query += " AND (source_paper = ? OR target_paper = ?)"
                params.extend([paper_id, paper_id])
            
            if relation_type:
                query += " AND relation = ?"
                params.append(relation_type)
            
            query += " ORDER BY score DESC"
            
            cursor.execute(query, params)
            
            relationships = []
            for row in cursor.fetchall():
                relationships.append({
                    'id': row[0],
                    'source_paper': row[1],
                    'relation': row[2],
                    'target_paper': row[3],
                    'score': row[4],
                    'confidence': row[5],
                    'created_date': row[6],
                    'metadata': json.loads(row[7]) if row[7] else {}
                })
            
            return relationships
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Paper count
            cursor.execute("SELECT COUNT(*) FROM papers_metadata")
            stats['total_papers'] = cursor.fetchone()[0]
            
            # Relationship counts by type
            cursor.execute("""
                SELECT relation, COUNT(*) 
                FROM crossref_relationships 
                GROUP BY relation
            """)
            stats['relationships_by_type'] = dict(cursor.fetchall())
            
            # Total relationships
            cursor.execute("SELECT COUNT(*) FROM crossref_relationships")
            stats['total_relationships'] = cursor.fetchone()[0]
            
            # Citation statistics
            cursor.execute("SELECT COUNT(*) FROM citations")
            stats['total_citations'] = cursor.fetchone()[0]
            
            # Similarity statistics
            cursor.execute("SELECT COUNT(*) FROM similarities")
            stats['total_similarities'] = cursor.fetchone()[0]
            
            return stats
    
    def _extract_year(self, date_str: Optional[str]) -> Optional[int]:
        """Extract year from date string."""
        if not date_str:
            return None
        try:
            return int(date_str[:4])
        except (ValueError, TypeError):
            return None


class CrossRefPipeline:
    """Main pipeline for cross-referencing research papers."""
    
    def __init__(self, config: CrossRefConfig):
        self.config = config
        
        # Initialize components
        self.citation_extractor = CitationExtractor()
        self.similarity_engine = SimilarityEngine(
            embedding_model_name=config.embedding_model,
            similarity_threshold=config.similarity_threshold
        )
        self.database = CrossRefDatabase(config.database_path)
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Cross-reference pipeline initialized")
    
    def process_papers(
        self, 
        papers: Dict[str, Dict],
        pdf_paths: Optional[Dict[str, str]] = None
    ) -> CrossRefGraph:
        """
        Process papers through the complete cross-reference pipeline.
        
        Args:
            papers: Dict of paper_id -> paper metadata
            pdf_paths: Optional dict of paper_id -> PDF file path
            
        Returns:
            Constructed knowledge graph
        """
        logger.info(f"Processing {len(papers)} papers for cross-referencing")
        
        # Store paper metadata
        self.database.store_paper_metadata(papers)
        
        # Step 1: Extract citations
        citation_matches = []
        if pdf_paths:
            citation_matches = self._extract_citations(papers, pdf_paths)
        
        # Step 2: Compute semantic similarities
        similarity_results = self._compute_similarities(papers)
        
        # Step 3: Build knowledge graph
        graph = self._build_knowledge_graph(papers, citation_matches, similarity_results)
        
        # Step 4: Store relationships in database
        if citation_matches:
            self.database.store_citations(citation_matches)
        
        if similarity_results:
            self.database.store_similarities(similarity_results)
        
        # Step 5: Add author relationships if enabled
        if self.config.include_author_relationships:
            author_rels = self._compute_author_relationships(graph)
            if author_rels:
                self.database.store_author_relationships(author_rels)
                graph.add_author_relationships()
        
        # Step 6: Export graph
        self._export_graph(graph)
        
        logger.info("Cross-reference processing completed")
        return graph
    
    def _extract_citations(
        self, 
        papers: Dict[str, Dict],
        pdf_paths: Dict[str, str]
    ) -> List[CitationMatch]:
        """Extract citations from PDFs."""
        logger.info("Extracting citations from PDFs")
        
        all_matches = []
        
        for paper_id, pdf_path in pdf_paths.items():
            if paper_id not in papers:
                continue
            
            try:
                # Extract citations from PDF
                citations = self.citation_extractor.extract_from_pdf(pdf_path)
                
                # Match citations to known papers
                matches = self.citation_extractor.match_citations_to_papers(
                    citations, paper_id, papers
                )
                
                # Filter by confidence threshold
                filtered_matches = [
                    match for match in matches 
                    if match.confidence >= self.config.citation_confidence_threshold
                ]
                
                all_matches.extend(filtered_matches)
                
                if self.config.save_intermediate:
                    # Save raw citations
                    citations_file = Path(self.config.output_dir) / f"{paper_id}_citations.json"
                    with open(citations_file, 'w') as f:
                        json.dump([asdict(c) for c in citations], f, indent=2, default=str)
                
            except Exception as e:
                logger.error(f"Failed to extract citations for {paper_id}: {e}")
                continue
        
        logger.info(f"Extracted {len(all_matches)} citation matches")
        return all_matches
    
    def _compute_similarities(self, papers: Dict[str, Dict]) -> List[SimilarityResult]:
        """Compute semantic similarities between papers."""
        logger.info("Computing semantic similarities")
        
        # Add papers to similarity engine
        self.similarity_engine.add_papers(papers, text_field="full_text")
        
        # Compute pairwise similarities
        similarities = self.similarity_engine.compute_pairwise_similarities(
            min_similarity=self.config.similarity_threshold
        )
        
        # Save similarity index if enabled
        if self.config.save_intermediate:
            index_path = Path(self.config.output_dir) / "similarity_index"
            self.similarity_engine.save_index(str(index_path))
        
        logger.info(f"Computed {len(similarities)} similarity relationships")
        return similarities
    
    def _build_knowledge_graph(
        self,
        papers: Dict[str, Dict],
        citation_matches: List[CitationMatch],
        similarity_results: List[SimilarityResult]
    ) -> CrossRefGraph:
        """Build knowledge graph from papers and relationships."""
        logger.info("Building knowledge graph")
        
        # Initialize graph
        graph = CrossRefGraph()
        
        # Add papers as nodes
        graph.add_papers(papers)
        
        # Add citation relationships
        if citation_matches:
            graph.add_citation_relationships(citation_matches)
        
        # Add similarity relationships
        if similarity_results:
            graph.add_similarity_relationships(similarity_results)
        
        logger.info(f"Built graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        return graph
    
    def _compute_author_relationships(self, graph: CrossRefGraph) -> List[Tuple[str, str, float, List[str]]]:
        """Compute author collaboration relationships."""
        logger.info("Computing author relationships")
        
        author_rels = []
        paper_ids = list(graph.nodes.keys())
        
        for i, paper1 in enumerate(paper_ids):
            for paper2 in paper_ids[i+1:]:
                node1 = graph.nodes[paper1]
                node2 = graph.nodes[paper2]
                
                if not node1.authors or not node2.authors:
                    continue
                
                # Find shared authors
                authors1 = set(a.lower().strip() for a in node1.authors)
                authors2 = set(a.lower().strip() for a in node2.authors)
                
                shared = authors1 & authors2
                if shared:
                    total = authors1 | authors2
                    overlap = len(shared) / len(total)
                    
                    if overlap > 0.1:  # Minimum threshold
                        author_rels.append((paper1, paper2, overlap, list(shared)))
        
        logger.info(f"Found {len(author_rels)} author relationships")
        return author_rels
    
    def _export_graph(self, graph: CrossRefGraph):
        """Export graph in requested formats."""
        logger.info("Exporting knowledge graph")
        
        output_dir = Path(self.config.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for export_format in self.config.graph_export_formats:
            try:
                if export_format == 'json':
                    output_file = output_dir / f"crossref_graph_{timestamp}.json"
                    GraphExporter.export_to_json(graph, str(output_file))
                
                elif export_format == 'neo4j':
                    output_path = output_dir / f"neo4j_export_{timestamp}"
                    GraphExporter.export_to_neo4j_csv(graph, str(output_path))
                
                elif export_format == 'networkx':
                    output_file = output_dir / f"networkx_graph_{timestamp}.pkl"
                    GraphExporter.export_to_networkx_pickle(graph, str(output_file))
                
                elif export_format == 'edgelist':
                    output_file = output_dir / f"edgelist_{timestamp}.csv"
                    GraphExporter.export_edge_list(graph, str(output_file))
                
            except Exception as e:
                logger.error(f"Failed to export graph in {export_format} format: {e}")
    
    def process_single_paper(
        self, 
        paper_id: str,
        paper_data: Dict,
        pdf_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single paper for cross-referencing against existing papers.
        
        Args:
            paper_id: ID of the paper
            paper_data: Paper metadata
            pdf_path: Optional path to PDF file
            
        Returns:
            Dictionary with relationships found
        """
        # Get existing papers from database
        existing_papers = self.database.get_paper_metadata()
        
        if not existing_papers:
            logger.warning("No existing papers in database for cross-referencing")
            return {}
        
        # Store new paper
        self.database.store_paper_metadata({paper_id: paper_data})
        
        results = {
            'paper_id': paper_id,
            'citations': [],
            'similarities': [],
            'author_relationships': []
        }
        
        # Extract citations if PDF available
        if pdf_path:
            try:
                citations = self.citation_extractor.extract_from_pdf(pdf_path)
                matches = self.citation_extractor.match_citations_to_papers(
                    citations, paper_id, existing_papers
                )
                
                # Filter and store
                filtered_matches = [
                    match for match in matches
                    if match.confidence >= self.config.citation_confidence_threshold
                ]
                
                if filtered_matches:
                    self.database.store_citations(filtered_matches)
                    results['citations'] = [asdict(m) for m in filtered_matches]
                
            except Exception as e:
                logger.error(f"Failed to extract citations for {paper_id}: {e}")
        
        # Compute similarities
        try:
            # Add new paper to similarity engine
            self.similarity_engine.add_papers({paper_id: paper_data})
            
            # Find similar papers
            similar_papers = self.similarity_engine.find_similar_papers(
                paper_id, k=self.config.similarity_top_k
            )
            
            if similar_papers:
                self.database.store_similarities(similar_papers)
                results['similarities'] = [asdict(s) for s in similar_papers]
        
        except Exception as e:
            logger.error(f"Failed to compute similarities for {paper_id}: {e}")
        
        return results
    
    def get_paper_relationships(self, paper_id: str) -> Dict[str, Any]:
        """Get all relationships for a specific paper."""
        relationships = self.database.get_relationships(paper_id=paper_id)
        
        # Group by relationship type
        grouped = {}
        for rel in relationships:
            rel_type = rel['relation']
            if rel_type not in grouped:
                grouped[rel_type] = []
            grouped[rel_type].append(rel)
        
        return grouped
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get statistics about the cross-reference pipeline."""
        db_stats = self.database.get_statistics()
        similarity_stats = self.similarity_engine.get_statistics()
        
        return {
            'database': db_stats,
            'similarity_engine': similarity_stats,
            'config': asdict(self.config)
        }


def create_crossref_pipeline(
    database_path: str = "crossref.db",
    output_dir: str = "data/crossref",
    similarity_threshold: float = 0.7,
    **kwargs
) -> CrossRefPipeline:
    """
    Create a cross-reference pipeline with default configuration.
    
    Args:
        database_path: Path to SQLite database
        output_dir: Output directory for exports
        similarity_threshold: Minimum similarity threshold
        **kwargs: Additional configuration options
        
    Returns:
        Configured pipeline
    """
    config = CrossRefConfig(
        database_path=database_path,
        output_dir=output_dir,
        similarity_threshold=similarity_threshold,
        **kwargs
    )
    
    return CrossRefPipeline(config)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <command> [args...]")
        print("Commands:")
        print("  process <papers.json> [pdf_dir] - Process papers for cross-referencing")
        print("  stats <database> - Show pipeline statistics")
        print("  relationships <database> <paper_id> - Show relationships for paper")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "process" and len(sys.argv) >= 3:
        papers_file = sys.argv[2]
        pdf_dir = sys.argv[3] if len(sys.argv) > 3 else None
        
        # Load papers
        with open(papers_file, 'r') as f:
            papers = json.load(f)
        
        # Create PDF paths if directory provided
        pdf_paths = {}
        if pdf_dir:
            pdf_dir_path = Path(pdf_dir)
            for paper_id in papers:
                pdf_file = pdf_dir_path / f"{paper_id}.pdf"
                if pdf_file.exists():
                    pdf_paths[paper_id] = str(pdf_file)
        
        # Create pipeline and process
        pipeline = create_crossref_pipeline()
        graph = pipeline.process_papers(papers, pdf_paths)
        
        print(f"Processed {len(papers)} papers")
        print(f"Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    elif command == "stats" and len(sys.argv) >= 3:
        database_path = sys.argv[2]
        
        pipeline = create_crossref_pipeline(database_path=database_path)
        stats = pipeline.get_pipeline_statistics()
        
        print("Pipeline Statistics:")
        print(json.dumps(stats, indent=2, default=str))
    
    elif command == "relationships" and len(sys.argv) >= 4:
        database_path = sys.argv[2]
        paper_id = sys.argv[3]
        
        pipeline = create_crossref_pipeline(database_path=database_path)
        relationships = pipeline.get_paper_relationships(paper_id)
        
        print(f"Relationships for paper {paper_id}:")
        for rel_type, rels in relationships.items():
            print(f"\n{rel_type}: {len(rels)} relationships")
            for rel in rels[:5]:  # Show first 5
                print(f"  -> {rel['target_paper']} (score: {rel['score']:.3f})")
    
    else:
        print("Invalid command or arguments")
        sys.exit(1)
