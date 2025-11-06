"""
Test Suite for Cross-Reference Module

Comprehensive tests for citation extraction, similarity analysis,
and knowledge graph generation.
"""

import unittest
import tempfile
import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import crossref components
from crossref.citation_extractor import CitationExtractor, Citation, CitationMatch
from crossref.similarity import SimilarityEngine, SimilarityResult
from crossref.graph import CrossRefGraph, GraphNode, GraphEdge, GraphExporter
from crossref.pipeline import CrossRefPipeline, CrossRefConfig, CrossRefDatabase


class TestCitationExtractor(unittest.TestCase):
    """Test citation extraction functionality."""
    
    def setUp(self):
        self.extractor = CitationExtractor()
        
        # Sample citation texts
        self.sample_citations = [
            "Smith, J. (2020). Deep Learning Advances. Nature, 123, 45-67.",
            "Johnson et al. Neural Networks in Practice. arXiv:2001.12345",
            "Brown, A., & Davis, B. (2019). Machine Learning Methods. DOI: 10.1038/nature12345",
            "Wilson, C. Computer Vision Applications. Conference on AI, 2021.",
            "Taylor, D. (2022). Natural Language Processing. Journal of AI, 15(3), 234-250."
        ]
        
        # Sample known papers
        self.known_papers = {
            "paper1": {
                "title": "Deep Learning Advances",
                "authors": ["Smith, J."],
                "year": 2020,
                "doi": None,
                "arxiv_id": None
            },
            "paper2": {
                "title": "Neural Networks in Practice", 
                "authors": ["Johnson, A.", "Lee, B."],
                "year": 2021,
                "doi": None,
                "arxiv_id": "2001.12345"
            },
            "paper3": {
                "title": "Machine Learning Methods",
                "authors": ["Brown, A.", "Davis, B."],
                "year": 2019,
                "doi": "10.1038/nature12345",
                "arxiv_id": None
            }
        }
    
    def test_citation_parsing(self):
        """Test basic citation parsing."""
        citation_text = "Smith, J. (2020). Deep Learning Advances. Nature, 123, 45-67."
        citation = self.extractor._parse_citation(citation_text)
        
        self.assertIsInstance(citation, Citation)
        self.assertEqual(citation.raw_text, citation_text)
        self.assertIn("Smith", citation.authors[0])
        self.assertEqual(citation.year, 2020)
        self.assertIn("Deep Learning", citation.title)
    
    def test_doi_extraction(self):
        """Test DOI pattern extraction."""
        citation_text = "Brown, A. (2019). ML Methods. DOI: 10.1038/nature12345"
        citation = self.extractor._parse_citation(citation_text)
        
        self.assertEqual(citation.doi, "10.1038/nature12345")
    
    def test_arxiv_extraction(self):
        """Test arXiv ID extraction."""
        citation_text = "Johnson et al. Neural Networks. arXiv:2001.12345"
        citation = self.extractor._parse_citation(citation_text)
        
        self.assertEqual(citation.arxiv_id, "2001.12345")
    
    def test_citation_matching_by_doi(self):
        """Test citation matching using DOI."""
        citations = [self.extractor._parse_citation(c) for c in self.sample_citations]
        
        matches = self.extractor.match_citations_to_papers(
            citations, "source_paper", self.known_papers
        )
        
        # Should find DOI match
        doi_matches = [m for m in matches if m.match_type == "doi"]
        self.assertGreater(len(doi_matches), 0)
        
        doi_match = doi_matches[0]
        self.assertEqual(doi_match.cited_paper_id, "paper3")
        self.assertGreater(doi_match.confidence, 0.9)
    
    def test_citation_matching_by_arxiv(self):
        """Test citation matching using arXiv ID."""
        citations = [self.extractor._parse_citation(c) for c in self.sample_citations]
        
        matches = self.extractor.match_citations_to_papers(
            citations, "source_paper", self.known_papers
        )
        
        # Should find arXiv match
        arxiv_matches = [m for m in matches if m.match_type == "arxiv"]
        self.assertGreater(len(arxiv_matches), 0)
        
        arxiv_match = arxiv_matches[0]
        self.assertEqual(arxiv_match.cited_paper_id, "paper2")
        self.assertGreater(arxiv_match.confidence, 0.9)
    
    def test_citation_matching_by_title(self):
        """Test citation matching using title similarity."""
        citations = [self.extractor._parse_citation(c) for c in self.sample_citations]
        
        matches = self.extractor.match_citations_to_papers(
            citations, "source_paper", self.known_papers
        )
        
        # Should find title matches
        title_matches = [m for m in matches if m.match_type == "title"]
        self.assertGreater(len(title_matches), 0)
        
        # Check confidence scores
        for match in title_matches:
            self.assertGreater(match.confidence, 0.3)
            self.assertIn(match.cited_paper_id, self.known_papers)
    
    @patch('fitz.open')
    def test_pdf_extraction(self, mock_fitz):
        """Test PDF citation extraction (mocked)."""
        # Mock PDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "\n".join(self.sample_citations)
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__len__.return_value = 1
        mock_fitz.return_value = mock_doc
        
        citations = self.extractor.extract_from_pdf("dummy.pdf")
        
        self.assertGreater(len(citations), 0)
        self.assertIsInstance(citations[0], Citation)
    
    def test_confidence_calculation(self):
        """Test citation match confidence calculation."""
        # Perfect DOI match should have high confidence
        citation = Citation(
            raw_text="Test citation",
            authors=["Smith, J."],
            title="Deep Learning Advances",
            year=2020,
            doi="10.1038/nature12345"
        )
        
        paper = self.known_papers["paper3"]
        confidence = self.extractor._calculate_confidence(citation, paper, "doi")
        
        self.assertGreater(confidence, 0.9)
    
    def test_empty_citations(self):
        """Test handling of empty citation lists."""
        matches = self.extractor.match_citations_to_papers(
            [], "source_paper", self.known_papers
        )
        
        self.assertEqual(len(matches), 0)
    
    def test_no_known_papers(self):
        """Test handling when no known papers provided."""
        citations = [self.extractor._parse_citation(c) for c in self.sample_citations]
        
        matches = self.extractor.match_citations_to_papers(
            citations, "source_paper", {}
        )
        
        self.assertEqual(len(matches), 0)


class TestSimilarityEngine(unittest.TestCase):
    """Test semantic similarity functionality."""
    
    def setUp(self):
        # Use a smaller model for testing
        self.engine = SimilarityEngine(
            embedding_model_name="all-MiniLM-L6-v2",
            similarity_threshold=0.5
        )
        
        # Sample papers
        self.sample_papers = {
            "paper1": {
                "title": "Deep Learning for Computer Vision",
                "abstract": "This paper presents advances in deep learning methods for computer vision tasks including image classification and object detection.",
                "full_text": "Deep learning computer vision image classification object detection neural networks"
            },
            "paper2": {
                "title": "Neural Networks in Natural Language Processing",
                "abstract": "We explore the application of neural networks to natural language processing tasks such as sentiment analysis and machine translation.",
                "full_text": "Neural networks natural language processing sentiment analysis machine translation"  
            },
            "paper3": {
                "title": "Computer Vision Applications in Robotics",
                "abstract": "This work demonstrates how computer vision techniques can be applied to robotic navigation and manipulation tasks.",
                "full_text": "Computer vision robotics navigation manipulation visual perception"
            },
            "paper4": {
                "title": "Quantum Computing Algorithms",
                "abstract": "Novel quantum algorithms for solving optimization problems and their implementation on quantum hardware.",
                "full_text": "Quantum computing algorithms optimization quantum hardware"
            }
        }
    
    def test_paper_addition(self):
        """Test adding papers to similarity engine."""
        initial_count = len(self.engine.paper_ids)
        
        self.engine.add_papers(self.sample_papers)
        
        self.assertEqual(len(self.engine.paper_ids), initial_count + len(self.sample_papers))
        
        for paper_id in self.sample_papers:
            self.assertIn(paper_id, self.engine.paper_ids)
    
    def test_embedding_generation(self):
        """Test text embedding generation."""
        text = "Deep learning for computer vision"
        embedding = self.engine._generate_embedding(text)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape[0], self.engine.embedding_dim)
    
    def test_pairwise_similarities(self):
        """Test pairwise similarity computation."""
        self.engine.add_papers(self.sample_papers)
        
        similarities = self.engine.compute_pairwise_similarities(min_similarity=0.1)
        
        self.assertGreater(len(similarities), 0)
        
        # Check similarity structure
        for sim in similarities:
            self.assertIsInstance(sim, SimilarityResult)
            self.assertIn(sim.source_paper_id, self.sample_papers)
            self.assertIn(sim.target_paper_id, self.sample_papers)
            self.assertGreaterEqual(sim.similarity_score, 0.1)
            self.assertLessEqual(sim.similarity_score, 1.0)
    
    def test_similar_paper_finding(self):
        """Test finding similar papers for a specific paper."""
        self.engine.add_papers(self.sample_papers)
        
        # Find papers similar to paper1 (computer vision)
        similar = self.engine.find_similar_papers("paper1", k=2)
        
        self.assertLessEqual(len(similar), 2)
        
        # Should not include the query paper itself
        for sim in similar:
            self.assertNotEqual(sim.source_paper_id, sim.target_paper_id)
        
        # paper3 should be similar (both about computer vision)
        similar_ids = [s.target_paper_id for s in similar]
        if "paper3" not in similar_ids and len(similar) > 0:
            # Check if paper3 has reasonable similarity
            paper3_sim = self.engine.compute_similarity("paper1", "paper3")
            # Computer vision papers should have some similarity
            self.assertGreater(paper3_sim, 0.2)
    
    def test_clustering(self):
        """Test paper clustering functionality."""
        self.engine.add_papers(self.sample_papers)
        
        clusters = self.engine.cluster_papers(n_clusters=2)
        
        self.assertIsInstance(clusters, dict)
        self.assertEqual(len(clusters), len(self.sample_papers))
        
        # Check cluster assignments
        for paper_id, cluster_id in clusters.items():
            self.assertIn(paper_id, self.sample_papers)
            self.assertIsInstance(cluster_id, (int, np.integer))
    
    def test_topic_communities(self):
        """Test topic community detection."""
        self.engine.add_papers(self.sample_papers)
        
        communities = self.engine.detect_topic_communities(similarity_threshold=0.3)
        
        self.assertIsInstance(communities, list)
        
        # Check community structure
        all_papers = set()
        for community in communities:
            self.assertIsInstance(community, list)
            for paper_id in community:
                self.assertIn(paper_id, self.sample_papers)
                all_papers.add(paper_id)
        
        # Most papers should be in some community
        self.assertGreaterEqual(len(all_papers), len(self.sample_papers) // 2)
    
    def test_empty_papers(self):
        """Test handling of empty paper collections."""
        similarities = self.engine.compute_pairwise_similarities()
        self.assertEqual(len(similarities), 0)
        
        similar = self.engine.find_similar_papers("nonexistent", k=5)
        self.assertEqual(len(similar), 0)
    
    def test_single_paper(self):
        """Test handling of single paper."""
        single_paper = {"paper1": self.sample_papers["paper1"]}
        self.engine.add_papers(single_paper)
        
        similarities = self.engine.compute_pairwise_similarities()
        self.assertEqual(len(similarities), 0)  # No pairs to compare
    
    def test_index_persistence(self):
        """Test saving and loading similarity index."""
        self.engine.add_papers(self.sample_papers)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "test_index"
            
            # Save index
            self.engine.save_index(str(index_path))
            
            # Create new engine and load index
            new_engine = SimilarityEngine(
                embedding_model_name="all-MiniLM-L6-v2",
                similarity_threshold=0.5
            )
            new_engine.load_index(str(index_path))
            
            # Should have same papers
            self.assertEqual(len(new_engine.paper_ids), len(self.sample_papers))
            
            # Should produce similar results
            original_sim = self.engine.compute_similarity("paper1", "paper2")
            loaded_sim = new_engine.compute_similarity("paper1", "paper2")
            
            self.assertAlmostEqual(original_sim, loaded_sim, places=3)


class TestCrossRefGraph(unittest.TestCase):
    """Test knowledge graph functionality."""
    
    def setUp(self):
        self.graph = CrossRefGraph()
        
        # Sample papers for graph
        self.sample_papers = {
            "paper1": {
                "title": "Deep Learning Fundamentals",
                "authors": ["Smith, J.", "Johnson, A."],
                "year": 2020,
                "doi": "10.1000/paper1"
            },
            "paper2": {
                "title": "Advanced Neural Networks",
                "authors": ["Brown, B.", "Davis, C."],
                "year": 2021,
                "doi": "10.1000/paper2"
            },
            "paper3": {
                "title": "Machine Learning Applications",
                "authors": ["Wilson, D.", "Taylor, E."],
                "year": 2019,
                "doi": "10.1000/paper3"
            }
        }
        
        # Sample citation matches
        self.citation_matches = [
            CitationMatch(
                source_paper_id="paper2",
                cited_paper_id="paper1",
                citation=Citation(
                    raw_text="Smith et al. Deep Learning Fundamentals",
                    authors=["Smith, J."],
                    title="Deep Learning Fundamentals",
                    year=2020,
                    doi="10.1000/paper1"
                ),
                match_type="doi",
                confidence=0.95
            )
        ]
        
        # Sample similarity results
        self.similarity_results = [
            SimilarityResult(
                source_paper_id="paper1",
                target_paper_id="paper3",
                similarity_score=0.75,
                match_type="semantic",
                metadata={"embedding_model": "test"}
            )
        ]
    
    def test_paper_addition(self):
        """Test adding papers as graph nodes."""
        self.graph.add_papers(self.sample_papers)
        
        self.assertEqual(len(self.graph.nodes), len(self.sample_papers))
        
        for paper_id, paper_data in self.sample_papers.items():
            self.assertIn(paper_id, self.graph.nodes)
            
            node = self.graph.nodes[paper_id]
            self.assertEqual(node.title, paper_data["title"])
            self.assertEqual(node.authors, paper_data["authors"])
            self.assertEqual(node.year, paper_data["year"])
    
    def test_citation_relationships(self):
        """Test adding citation relationships to graph."""
        self.graph.add_papers(self.sample_papers)
        self.graph.add_citation_relationships(self.citation_matches)
        
        # Should have citation edge
        edge_key = ("paper2", "paper1", "cites")
        self.assertIn(edge_key, self.graph.edges)
        
        edge = self.graph.edges[edge_key]
        self.assertEqual(edge.relationship, "cites")
        self.assertEqual(edge.weight, 0.95)
    
    def test_similarity_relationships(self):
        """Test adding similarity relationships to graph."""
        self.graph.add_papers(self.sample_papers)
        self.graph.add_similarity_relationships(self.similarity_results)
        
        # Should have similarity edges (bidirectional)
        edge1 = ("paper1", "paper3", "similar_to")
        edge2 = ("paper3", "paper1", "similar_to")
        
        self.assertIn(edge1, self.graph.edges)
        self.assertIn(edge2, self.graph.edges)
        
        # Check edge properties
        for edge_key in [edge1, edge2]:
            edge = self.graph.edges[edge_key]
            self.assertEqual(edge.relationship, "similar_to")
            self.assertEqual(edge.weight, 0.75)
    
    def test_centrality_metrics(self):
        """Test graph centrality metric calculation."""
        self.graph.add_papers(self.sample_papers)
        self.graph.add_citation_relationships(self.citation_matches)
        
        centrality = self.graph.compute_centrality_metrics()
        
        self.assertIn("degree", centrality)
        self.assertIn("betweenness", centrality)
        self.assertIn("closeness", centrality)
        self.assertIn("pagerank", centrality)
        
        # Check that all papers have centrality scores
        for paper_id in self.sample_papers:
            for metric in centrality:
                self.assertIn(paper_id, centrality[metric])
    
    def test_community_detection(self):
        """Test community detection in graph."""
        self.graph.add_papers(self.sample_papers)
        self.graph.add_citation_relationships(self.citation_matches)
        self.graph.add_similarity_relationships(self.similarity_results)
        
        communities = self.graph.detect_communities()
        
        self.assertIsInstance(communities, dict)
        
        # All papers should be assigned to communities
        for paper_id in self.sample_papers:
            self.assertIn(paper_id, communities)
    
    def test_subgraph_extraction(self):
        """Test extracting subgraphs."""
        self.graph.add_papers(self.sample_papers)
        self.graph.add_citation_relationships(self.citation_matches)
        
        # Extract subgraph around paper1
        subgraph = self.graph.get_subgraph(["paper1", "paper2"])
        
        self.assertIsInstance(subgraph, CrossRefGraph)
        self.assertEqual(len(subgraph.nodes), 2)
        self.assertIn("paper1", subgraph.nodes)
        self.assertIn("paper2", subgraph.nodes)
    
    def test_neighbor_finding(self):
        """Test finding neighboring papers."""
        self.graph.add_papers(self.sample_papers)
        self.graph.add_citation_relationships(self.citation_matches)
        self.graph.add_similarity_relationships(self.similarity_results)
        
        # Get neighbors of paper1
        neighbors = self.graph.get_neighbors("paper1")
        
        self.assertIsInstance(neighbors, list)
        # paper1 should have paper2 (cited by) and paper3 (similar to) as neighbors
        neighbor_ids = [n[0] for n in neighbors]
        self.assertIn("paper2", neighbor_ids)
        self.assertIn("paper3", neighbor_ids)
    
    def test_empty_graph(self):
        """Test operations on empty graph."""
        centrality = self.graph.compute_centrality_metrics()
        for metric in centrality:
            self.assertEqual(len(centrality[metric]), 0)
        
        communities = self.graph.detect_communities()
        self.assertEqual(len(communities), 0)
        
        neighbors = self.graph.get_neighbors("nonexistent")
        self.assertEqual(len(neighbors), 0)


class TestGraphExporter(unittest.TestCase):
    """Test graph export functionality."""
    
    def setUp(self):
        self.graph = CrossRefGraph()
        
        # Create sample graph
        sample_papers = {
            "paper1": {"title": "Paper 1", "authors": ["Author 1"], "year": 2020},
            "paper2": {"title": "Paper 2", "authors": ["Author 2"], "year": 2021}
        }
        
        self.graph.add_papers(sample_papers)
        self.graph.add_edge("paper1", "paper2", "cites", 1.0)
    
    def test_json_export(self):
        """Test JSON export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            GraphExporter.export_to_json(self.graph, f.name)
            
            # Read back and verify
            with open(f.name, 'r') as read_f:
                data = json.load(read_f)
            
            self.assertIn("nodes", data)
            self.assertIn("edges", data)
            self.assertEqual(len(data["nodes"]), 2)
            self.assertEqual(len(data["edges"]), 1)
    
    def test_networkx_pickle_export(self):
        """Test NetworkX pickle export."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            GraphExporter.export_to_networkx_pickle(self.graph, f.name)
            
            # File should exist and have content
            self.assertTrue(Path(f.name).exists())
            self.assertGreater(Path(f.name).stat().st_size, 0)
    
    def test_edge_list_export(self):
        """Test edge list CSV export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            GraphExporter.export_edge_list(self.graph, f.name)
            
            # Read back and verify
            with open(f.name, 'r') as read_f:
                content = read_f.read()
                
            self.assertIn("source", content)
            self.assertIn("target", content)
            self.assertIn("paper1", content)
            self.assertIn("paper2", content)
    
    def test_neo4j_export(self):
        """Test Neo4j CSV export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "neo4j_export"
            GraphExporter.export_to_neo4j_csv(self.graph, str(output_path))
            
            # Check files created
            self.assertTrue((output_path / "nodes.csv").exists())
            self.assertTrue((output_path / "relationships.csv").exists())
            
            # Verify content
            with open(output_path / "nodes.csv", 'r') as f:
                nodes_content = f.read()
                self.assertIn("paper1", nodes_content)
                self.assertIn("paper2", nodes_content)


class TestCrossRefDatabase(unittest.TestCase):
    """Test database operations."""
    
    def setUp(self):
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        self.database = CrossRefDatabase(self.temp_db.name)
        
        # Sample data
        self.sample_papers = {
            "paper1": {
                "title": "Deep Learning Paper",
                "authors": ["Smith, J.", "Jones, A."],
                "year": 2020,
                "doi": "10.1000/test1",
                "arxiv_id": None,
                "abstract": "A paper about deep learning",
                "keywords": ["deep learning", "AI"]
            }
        }
        
        self.sample_citations = [
            CitationMatch(
                source_paper_id="paper2",
                cited_paper_id="paper1",
                citation=Citation(
                    raw_text="Smith et al. 2020",
                    authors=["Smith, J."],
                    title="Deep Learning Paper",
                    year=2020,
                    doi="10.1000/test1"
                ),
                match_type="doi",
                confidence=0.9
            )
        ]
        
        self.sample_similarities = [
            SimilarityResult(
                source_paper_id="paper1",
                target_paper_id="paper2",
                similarity_score=0.8,
                match_type="semantic",
                metadata={"model": "test"}
            )
        ]
    
    def tearDown(self):
        """Clean up temporary database."""
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_paper_metadata_storage(self):
        """Test storing and retrieving paper metadata."""
        self.database.store_paper_metadata(self.sample_papers)
        
        retrieved = self.database.get_paper_metadata(["paper1"])
        
        self.assertEqual(len(retrieved), 1)
        self.assertIn("paper1", retrieved)
        
        paper = retrieved["paper1"]
        self.assertEqual(paper["title"], "Deep Learning Paper")
        self.assertEqual(paper["year"], 2020)
        self.assertEqual(paper["doi"], "10.1000/test1")
    
    def test_citation_storage(self):
        """Test storing citation relationships."""
        # Store paper metadata first
        self.database.store_paper_metadata(self.sample_papers)
        
        # Store citations
        self.database.store_citations(self.sample_citations)
        
        # Verify in database
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM citations")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
            
            cursor.execute("SELECT COUNT(*) FROM crossref_relationships WHERE relation = 'cites'")
            rel_count = cursor.fetchone()[0]
            self.assertEqual(rel_count, 1)
    
    def test_similarity_storage(self):
        """Test storing similarity relationships."""
        self.database.store_similarities(self.sample_similarities)
        
        # Verify in database
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM similarities")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
            
            cursor.execute("SELECT COUNT(*) FROM crossref_relationships WHERE relation = 'similar_to'")
            rel_count = cursor.fetchone()[0]
            self.assertEqual(rel_count, 2)  # Bidirectional
    
    def test_relationship_retrieval(self):
        """Test retrieving relationships."""
        self.database.store_paper_metadata(self.sample_papers)
        self.database.store_citations(self.sample_citations)
        
        relationships = self.database.get_relationships(paper_id="paper1")
        
        self.assertGreater(len(relationships), 0)
        
        # Should find the citation relationship
        cite_rels = [r for r in relationships if r['relation'] == 'cites']
        self.assertEqual(len(cite_rels), 1)
    
    def test_statistics(self):
        """Test database statistics."""
        self.database.store_paper_metadata(self.sample_papers)
        self.database.store_citations(self.sample_citations)
        
        stats = self.database.get_statistics()
        
        self.assertIn("total_papers", stats)
        self.assertIn("total_relationships", stats)
        self.assertIn("total_citations", stats)
        
        self.assertEqual(stats["total_papers"], 1)
        self.assertGreater(stats["total_relationships"], 0)


class TestCrossRefPipeline(unittest.TestCase):
    """Test complete pipeline functionality."""
    
    def setUp(self):
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        # Create temporary output directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Configure pipeline
        config = CrossRefConfig(
            database_path=self.temp_db.name,
            output_dir=self.temp_dir,
            similarity_threshold=0.5,
            citation_confidence_threshold=0.3
        )
        
        self.pipeline = CrossRefPipeline(config)
        
        # Sample papers
        self.sample_papers = {
            "paper1": {
                "title": "Deep Learning for Computer Vision",
                "authors": ["Smith, J.", "Brown, A."],
                "year": 2020,
                "abstract": "This paper explores deep learning methods for computer vision tasks.",
                "full_text": "Deep learning computer vision neural networks image classification",
                "doi": "10.1000/test1"
            },
            "paper2": {
                "title": "Neural Networks in NLP",
                "authors": ["Johnson, B.", "Davis, C."],
                "year": 2021,
                "abstract": "Application of neural networks to natural language processing.",
                "full_text": "Neural networks natural language processing text analysis",
                "arxiv_id": "2021.12345"
            }
        }
    
    def tearDown(self):
        """Clean up temporary files."""
        Path(self.temp_db.name).unlink(missing_ok=True)
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_paper_processing(self):
        """Test complete paper processing pipeline."""
        # Process papers (without PDFs for testing)
        graph = self.pipeline.process_papers(self.sample_papers)
        
        # Verify graph creation
        self.assertIsInstance(graph, CrossRefGraph)
        self.assertEqual(len(graph.nodes), len(self.sample_papers))
        
        # Verify database storage
        retrieved_papers = self.pipeline.database.get_paper_metadata()
        self.assertEqual(len(retrieved_papers), len(self.sample_papers))
        
        # Check statistics
        stats = self.pipeline.get_pipeline_statistics()
        self.assertEqual(stats['database']['total_papers'], len(self.sample_papers))
    
    def test_single_paper_processing(self):
        """Test processing a single paper."""
        # Add initial papers
        self.pipeline.database.store_paper_metadata(self.sample_papers)
        
        # Process new paper
        new_paper = {
            "title": "Machine Learning Applications",
            "authors": ["Wilson, D."],
            "year": 2022,
            "abstract": "Applications of machine learning in various domains.",
            "full_text": "Machine learning applications data science"
        }
        
        results = self.pipeline.process_single_paper("paper3", new_paper)
        
        self.assertIn("paper_id", results)
        self.assertEqual(results["paper_id"], "paper3")
        self.assertIn("similarities", results)
    
    def test_relationship_retrieval(self):
        """Test retrieving relationships for specific papers."""
        # Process papers first
        self.pipeline.process_papers(self.sample_papers)
        
        # Get relationships for paper1
        relationships = self.pipeline.get_paper_relationships("paper1")
        
        self.assertIsInstance(relationships, dict)
        # May have similarity relationships
        if relationships:
            for rel_type, rels in relationships.items():
                self.assertIsInstance(rels, list)


def run_integration_tests():
    """Run integration tests with real components."""
    print("Running integration tests...")
    
    # Test citation extraction with real text
    extractor = CitationExtractor()
    
    sample_text = """
    References:
    
    1. Smith, J., & Johnson, A. (2020). Deep Learning Fundamentals. 
       Nature Machine Intelligence, 1(2), 123-145. DOI: 10.1038/s42256-020-0001-x
    
    2. Brown, B. et al. (2021). Advanced Neural Networks. arXiv:2101.12345
    
    3. Davis, C. (2019). Machine Learning Methods. Conference on AI, pp. 234-250.
    """
    
    citations = extractor._extract_citations_from_text(sample_text)
    print(f"✓ Extracted {len(citations)} citations from sample text")
    
    # Test similarity engine with sample papers
    similarity_engine = SimilarityEngine(similarity_threshold=0.3)
    
    test_papers = {
        "paper1": {
            "title": "Deep Learning for Image Classification",
            "abstract": "This paper presents deep learning methods for image classification tasks using convolutional neural networks.",
            "full_text": "deep learning image classification convolutional neural networks computer vision"
        },
        "paper2": {
            "title": "Computer Vision Applications",
            "abstract": "Applications of computer vision techniques in robotics and autonomous systems.",
            "full_text": "computer vision robotics autonomous systems visual perception"
        },
        "paper3": {
            "title": "Natural Language Processing with Transformers",
            "abstract": "Using transformer models for natural language understanding and generation tasks.",
            "full_text": "natural language processing transformers BERT GPT text analysis"
        }
    }
    
    similarity_engine.add_papers(test_papers)
    similarities = similarity_engine.compute_pairwise_similarities(min_similarity=0.1)
    print(f"✓ Computed {len(similarities)} similarity relationships")
    
    # Test graph construction
    graph = CrossRefGraph()
    graph.add_papers(test_papers)
    graph.add_similarity_relationships(similarities)
    
    centrality = graph.compute_centrality_metrics()
    print(f"✓ Computed centrality metrics for {len(centrality['degree'])} nodes")
    
    print("✅ Integration tests completed successfully")


def run_performance_tests():
    """Run performance tests with larger datasets."""
    print("Running performance tests...")
    
    import time
    
    # Generate synthetic papers
    num_papers = 100
    papers = {}
    
    topics = [
        "machine learning classification algorithms",
        "computer vision image processing",
        "natural language processing transformers",
        "robotics autonomous navigation", 
        "quantum computing algorithms",
        "bioinformatics genomic analysis"
    ]
    
    for i in range(num_papers):
        topic = topics[i % len(topics)]
        papers[f"paper_{i}"] = {
            "title": f"Research Paper {i}: {topic.title()}",
            "abstract": f"This paper explores {topic} and related methodologies.",
            "full_text": f"{topic} research methodology experimental results",
            "authors": [f"Author_{i}", f"Coauthor_{i}"],
            "year": 2020 + (i % 5)
        }
    
    # Test similarity computation performance
    similarity_engine = SimilarityEngine(similarity_threshold=0.5)
    
    start_time = time.time()
    similarity_engine.add_papers(papers)
    add_time = time.time() - start_time
    
    start_time = time.time()
    similarities = similarity_engine.compute_pairwise_similarities(min_similarity=0.5)
    similarity_time = time.time() - start_time
    
    print(f"✓ Added {num_papers} papers in {add_time:.2f} seconds")
    print(f"✓ Computed {len(similarities)} similarities in {similarity_time:.2f} seconds")
    
    # Test graph construction performance
    start_time = time.time()
    graph = CrossRefGraph()
    graph.add_papers(papers)
    graph.add_similarity_relationships(similarities)
    graph_time = time.time() - start_time
    
    print(f"✓ Built graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges in {graph_time:.2f} seconds")
    
    print("✅ Performance tests completed")


if __name__ == "__main__":
    # Run all tests
    print("🧪 Running Cross-Reference Module Test Suite")
    print("=" * 50)
    
    # Unit tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print(f"\n📊 Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Integration tests
    print(f"\n🔗 Integration Tests:")
    try:
        run_integration_tests()
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
    
    # Performance tests
    print(f"\n⚡ Performance Tests:")
    try:
        run_performance_tests()
    except Exception as e:
        print(f"❌ Performance tests failed: {e}")
    
    # Summary
    if result.wasSuccessful():
        print(f"\n✅ All tests passed!")
        exit_code = 0
    else:
        print(f"\n❌ Some tests failed!")
        exit_code = 1
    
    print(f"🏁 Test suite completed")
    exit(exit_code)
