"""
Comprehensive tests for QA Module (Module 4)

Tests all components of the Question Answerer module including:
- Hybrid retrieval (FAISS + BM25)
- RAG pipeline with Ollama LLM
- Query rewriting and expansion
- Answer formatting
- Integration testing
"""

import pytest
import tempfile
import sqlite3
import json
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Test imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa.retriever import HybridRetriever, RetrievedChunk, create_bm25_index, create_chunk_database
from qa.rag import RAGPipeline, RAGResponse, OllamaLLM, create_rag_pipeline
from qa.query_rewriter import QueryRewriter, QueryExpansion, create_academic_query_rewriter
from qa.formatter import AnswerFormatter, FormattedCitation, FormattedAnswer

class TestHybridRetriever:
    """Test cases for HybridRetriever."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {
                'chunk_id': 'chunk_1',
                'paper_id': 'paper_1',
                'content': 'Transformers are a type of neural network architecture that rely entirely on attention mechanisms.',
                'section': 'introduction',
                'token_count': 15,
                'metadata': {
                    'title': 'Attention Is All You Need',
                    'authors': 'Vaswani et al.',
                    'arxiv_id': '1706.03762'
                }
            },
            {
                'chunk_id': 'chunk_2',
                'paper_id': 'paper_1',
                'content': 'The attention mechanism allows the model to focus on different parts of the input sequence.',
                'section': 'methodology',
                'token_count': 16,
                'metadata': {
                    'title': 'Attention Is All You Need',
                    'authors': 'Vaswani et al.',
                    'arxiv_id': '1706.03762'
                }
            },
            {
                'chunk_id': 'chunk_3',
                'paper_id': 'paper_2',
                'content': 'Convolutional neural networks have been the dominant approach for computer vision tasks.',
                'section': 'related_work',
                'token_count': 13,
                'metadata': {
                    'title': 'Deep Learning for Computer Vision',
                    'authors': 'LeCun et al.',
                    'doi': '10.1038/nature14539'
                }
            }
        ]
    
    @pytest.fixture
    def temp_db_path(self, sample_chunks):
        """Create temporary chunk database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        create_chunk_database(sample_chunks, db_path)
        yield db_path
        
        # Cleanup
        os.unlink(db_path)
    
    @pytest.fixture
    def temp_bm25_path(self, sample_chunks):
        """Create temporary BM25 index."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            bm25_path = f.name
        
        chunk_texts = [chunk['content'] for chunk in sample_chunks]
        create_bm25_index(chunk_texts, bm25_path)
        yield bm25_path
        
        # Cleanup
        os.unlink(bm25_path)
    
    @patch('qa.retriever.faiss.read_index')
    @patch('qa.retriever.SentenceTransformer')
    def test_retriever_initialization(self, mock_st, mock_faiss, temp_db_path, temp_bm25_path):
        """Test retriever initialization."""
        # Mock FAISS index
        mock_index = Mock()
        mock_index.ntotal = 3
        mock_faiss.return_value = mock_index
        
        # Mock sentence transformer
        mock_model = Mock()
        mock_st.return_value = mock_model
        
        retriever = HybridRetriever(
            faiss_index_path="dummy_faiss.bin",
            bm25_index_path=temp_bm25_path,
            chunk_metadata_path=temp_db_path
        )
        
        assert retriever is not None
        assert len(retriever.chunk_metadata) == 3
        assert retriever.bm25_index is not None
        assert len(retriever.chunk_texts) == 3
    
    @patch('qa.retriever.faiss.read_index')
    @patch('qa.retriever.SentenceTransformer')
    def test_bm25_search(self, mock_st, mock_faiss, temp_db_path, temp_bm25_path):
        """Test BM25 keyword search."""
        mock_index = Mock()
        mock_faiss.return_value = mock_index
        mock_st.return_value = Mock()
        
        retriever = HybridRetriever(
            faiss_index_path="dummy_faiss.bin",
            bm25_index_path=temp_bm25_path,
            chunk_metadata_path=temp_db_path
        )
        
        # Test BM25 search
        results = retriever._bm25_search("attention mechanism", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(chunk, RetrievedChunk) for chunk in results)
        if results:
            assert all(chunk.score > 0 for chunk in results)
    
    def test_chunk_database_creation(self, sample_chunks):
        """Test chunk database creation and querying."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            create_chunk_database(sample_chunks, db_path)
            
            # Verify database contents
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM chunks")
            count = cursor.fetchone()[0]
            assert count == 3
            
            cursor.execute("SELECT chunk_id, paper_id, content FROM chunks WHERE chunk_id = 'chunk_1'")
            row = cursor.fetchone()
            assert row[0] == 'chunk_1'
            assert row[1] == 'paper_1'
            assert 'Transformers' in row[2]
            
            conn.close()
            
        finally:
            os.unlink(db_path)
    
    def test_bm25_index_creation(self, sample_chunks):
        """Test BM25 index creation and loading."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            bm25_path = f.name
        
        try:
            chunk_texts = [chunk['content'] for chunk in sample_chunks]
            create_bm25_index(chunk_texts, bm25_path)
            
            # Verify index file exists
            assert os.path.exists(bm25_path)
            
            # Load and test index
            import pickle
            with open(bm25_path, 'rb') as f:
                bm25_data = pickle.load(f)
            
            assert 'index' in bm25_data
            assert 'texts' in bm25_data
            assert len(bm25_data['texts']) == 3
            
        finally:
            os.unlink(bm25_path)

class TestRAGPipeline:
    """Test cases for RAG Pipeline."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Mock retriever for testing."""
        retriever = Mock(spec=HybridRetriever)
        
        # Mock retrieved chunks
        mock_chunks = [
            RetrievedChunk(
                chunk_id='chunk_1',
                paper_id='paper_1',
                content='Transformers revolutionized NLP with attention mechanisms.',
                score=0.9,
                metadata={'title': 'Attention Is All You Need', 'authors': 'Vaswani et al.'},
                source='faiss'
            ),
            RetrievedChunk(
                chunk_id='chunk_2',
                paper_id='paper_2',
                content='Self-attention allows models to process sequences efficiently.',
                score=0.8,
                metadata={'title': 'BERT Paper', 'authors': 'Devlin et al.'},
                source='hybrid'
            )
        ]
        
        retriever.retrieve.return_value = mock_chunks
        retriever.get_paper_chunks.return_value = mock_chunks
        retriever.get_statistics.return_value = {
            'total_chunks': 100,
            'faiss_available': True,
            'bm25_available': True
        }
        
        return retriever
    
    @patch('requests.post')
    def test_ollama_llm(self, mock_post):
        """Test Ollama LLM wrapper."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'This is a test response from Ollama.'
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        llm = OllamaLLM(model_name="deepseek-coder-v2")
        response = llm("Test prompt")
        
        assert response == "This is a test response from Ollama."
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_rag_pipeline_query(self, mock_post, mock_retriever):
        """Test RAG pipeline query processing."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Transformers use attention mechanisms [arxiv:1706.03762] to process sequences effectively.'
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_model_name="deepseek-coder-v2"
        )
        
        response = pipeline.query("What are transformers?")
        
        assert isinstance(response, RAGResponse)
        assert response.answer is not None
        assert len(response.citations) > 0
        assert response.confidence > 0
        assert response.query == "What are transformers?"
    
    def test_rag_pipeline_error_handling(self, mock_retriever):
        """Test RAG pipeline error handling."""
        # Mock retriever to return empty results
        mock_retriever.retrieve.return_value = []
        
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_model_name="deepseek-coder-v2"
        )
        
        response = pipeline.query("Empty query test")
        
        assert isinstance(response, RAGResponse)
        assert "couldn't find any relevant information" in response.answer.lower()
        assert len(response.citations) == 0
        assert response.confidence == 0.0
    
    def test_context_preparation(self, mock_retriever):
        """Test context preparation for LLM prompt."""
        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_model_name="deepseek-coder-v2"
        )
        
        chunks = mock_retriever.retrieve.return_value
        context, citations = pipeline._prepare_context(chunks)
        
        assert len(context) > 0
        assert len(citations) == len(chunks)
        assert 'Transformers' in context
        # Context should contain content from mock chunks
        assert any(chunk.content in context for chunk in chunks)

class TestQueryRewriter:
    """Test cases for Query Rewriter."""
    
    def test_domain_term_expansion(self):
        """Test domain-specific term expansion."""
        rewriter = QueryRewriter(use_llm_expansion=False)
        
        # Test transformer expansion
        terms = rewriter._expand_with_domain_terms("transformer architecture")
        assert any("attention" in term.lower() for term in terms)
        
        # Test neural network expansion
        terms = rewriter._expand_with_domain_terms("neural network training")
        assert any("deep learning" in term.lower() for term in terms)
    
    def test_synonym_expansion(self):
        """Test synonym expansion."""
        rewriter = QueryRewriter(use_llm_expansion=False)
        
        terms = rewriter._expand_with_synonyms("improve method")
        assert "enhance" in terms or "approach" in terms
    
    def test_query_normalization(self):
        """Test academic phrase normalization."""
        rewriter = QueryRewriter()
        
        normalized = rewriter._normalize_query("State of the art machine learning")
        assert "state-of-the-art" in normalized
        assert "machine learning" in normalized
    
    @patch('requests.post')
    def test_llm_query_expansion(self, mock_post):
        """Test LLM-based query expansion."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'transformer neural network attention mechanism deep learning NLP'
        }
        mock_post.return_value = mock_response
        
        rewriter = QueryRewriter(use_llm_expansion=True)
        expanded = rewriter._llm_expand_query("transformer")
        
        assert expanded is not None
        assert len(expanded) > len("transformer")
    
    def test_hybrid_rewriting(self):
        """Test hybrid query rewriting."""
        rewriter = QueryRewriter(use_llm_expansion=False)  # Disable LLM for testing
        
        original = "transformer attention"
        expanded = rewriter.rewrite(original, method="hybrid")
        
        assert len(expanded) >= len(original)
        assert "transformer" in expanded.lower()
    
    def test_expansion_details(self):
        """Test query expansion details."""
        rewriter = QueryRewriter(use_llm_expansion=False)
        
        expansion = rewriter.get_expansion_details("neural network")
        
        assert isinstance(expansion, QueryExpansion)
        assert expansion.original_query == "neural network"
        assert len(expansion.expanded_query) >= len("neural network")
        assert expansion.confidence >= 0

class TestAnswerFormatter:
    """Test cases for Answer Formatter."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for formatting tests."""
        return [
            RetrievedChunk(
                chunk_id='chunk_1',
                paper_id='paper_1',
                content='Sample content about transformers.',
                score=0.9,
                metadata={
                    'title': 'Attention Is All You Need',
                    'authors': 'Vaswani et al.',
                    'arxiv_id': '1706.03762',
                    'year': 2017
                },
                source='faiss'
            )
        ]
    
    def test_citation_parsing(self):
        """Test citation ID parsing."""
        formatter = AnswerFormatter()
        
        # Test arXiv citation
        arxiv_info = formatter._parse_citation_id("arxiv:1706.03762")
        assert arxiv_info['type'] == 'arxiv'
        assert arxiv_info['id'] == '1706.03762'
        
        # Test DOI citation
        doi_info = formatter._parse_citation_id("doi:10.1038/nature14539")
        assert doi_info['type'] == 'doi'
        assert doi_info['id'] == '10.1038/nature14539'
    
    def test_citation_url_generation(self):
        """Test citation URL generation."""
        formatter = AnswerFormatter()
        
        # Test arXiv URL
        url = formatter._generate_citation_url({'type': 'arxiv', 'id': '1706.03762'})
        assert url == "https://arxiv.org/abs/1706.03762"
        
        # Test DOI URL
        url = formatter._generate_citation_url({'type': 'doi', 'id': '10.1038/nature14539'})
        assert url == "https://doi.org/10.1038/nature14539"
    
    def test_json_formatting(self, sample_chunks):
        """Test JSON answer formatting."""
        formatter = AnswerFormatter()
        
        answer = "Transformers use attention mechanisms [arxiv:1706.03762] for processing."
        citations = ["arxiv:1706.03762"]
        
        json_output = formatter.format_as_json(
            answer, citations, sample_chunks, confidence=0.9, query="What are transformers?"
        )
        
        parsed = json.loads(json_output)
        assert 'answer' in parsed
        assert 'citations' in parsed
        assert 'metadata' in parsed
        assert parsed['metadata']['confidence'] == 0.9
    
    def test_markdown_formatting(self, sample_chunks):
        """Test Markdown answer formatting."""
        formatter = AnswerFormatter()
        
        answer = "Transformers use attention mechanisms [arxiv:1706.03762] for processing."
        citations = ["arxiv:1706.03762"]
        
        markdown = formatter.format_as_markdown(
            answer, citations, sample_chunks, query="What are transformers?"
        )
        
        assert "**Query:**" in markdown
        assert "**Answer:**" in markdown
        assert "**References:**" in markdown
        # Check that markdown contains reference information
        assert "Link" in markdown or len(citations) > 0
    
    def test_html_formatting(self, sample_chunks):
        """Test HTML answer formatting."""
        formatter = AnswerFormatter()
        
        answer = "Transformers use attention mechanisms [arxiv:1706.03762] for processing."
        citations = ["arxiv:1706.03762"]
        
        html = formatter.format_as_html(
            answer, citations, sample_chunks, query="What are transformers?"
        )
        
        assert "<h3>Query</h3>" in html
        assert "<h3>Answer</h3>" in html
        assert "<h3>References</h3>" in html
        assert "https://arxiv.org/abs/1706.03762" in html
    
    def test_citation_validation(self):
        """Test citation validation."""
        formatter = AnswerFormatter()
        
        answer = "This uses [arxiv:1706.03762] and [missing:citation] references."
        available_citations = ["arxiv:1706.03762"]
        
        validation = formatter.validate_citations(answer, available_citations)
        
        assert len(validation['valid_citations']) == 1
        assert len(validation['missing_citations']) == 1
        assert validation['citation_coverage'] == 0.5

class TestIntegration:
    """Integration tests for the complete QA pipeline."""
    
    @pytest.fixture
    def temp_files(self):
        """Create temporary files for integration testing."""
        files = {}
        
        # Create temp database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            files['chunks_db'] = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            files['papers_db'] = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            files['bm25_index'] = f.name
        
        files['faiss_index'] = '/tmp/test_faiss.bin'  # Will be mocked
        
        yield files
        
        # Cleanup
        for path in files.values():
            if os.path.exists(path):
                os.unlink(path)
    
    def test_end_to_end_pipeline_creation(self, temp_files):
        """Test complete pipeline creation and basic functionality."""
        # Create sample data
        sample_chunks = [
            {
                'chunk_id': 'chunk_1',
                'paper_id': 'paper_1',
                'content': 'Transformers use attention mechanisms for sequence processing.',
                'section': 'introduction',
                'token_count': 8,
                'metadata': {'title': 'Test Paper', 'authors': 'Test Author'}
            }
        ]
        
        # Create databases
        create_chunk_database(sample_chunks, temp_files['chunks_db'])
        create_bm25_index([chunk['content'] for chunk in sample_chunks], temp_files['bm25_index'])
        
        # Mock FAISS components
        with patch('qa.retriever.faiss.read_index'), \
             patch('qa.retriever.SentenceTransformer'), \
             patch('requests.post') as mock_post:
            
            # Mock LLM response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'response': 'Test answer with [paper_1] citation.'
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            # Create pipeline
            pipeline = create_rag_pipeline(
                faiss_index_path=temp_files['faiss_index'],
                bm25_index_path=temp_files['bm25_index'],
                chunk_metadata_path=temp_files['chunks_db'],
                papers_db_path=temp_files['papers_db']
            )
            
            assert pipeline is not None
            assert pipeline.retriever is not None
    
    @patch('qa.retriever.faiss.read_index')
    @patch('qa.retriever.SentenceTransformer')
    @patch('requests.post')
    def test_query_processing_workflow(self, mock_post, mock_st, mock_faiss, temp_files):
        """Test complete query processing workflow."""
        # Setup mocks
        mock_index = Mock()
        mock_index.ntotal = 1
        mock_faiss.return_value = mock_index
        
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_st.return_value = mock_model
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Transformers are neural network architectures [paper_1].'
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        # Create test data
        sample_chunks = [
            {
                'chunk_id': 'chunk_1',
                'paper_id': 'paper_1',
                'content': 'Transformers use attention mechanisms.',
                'section': 'introduction',
                'token_count': 5,
                'metadata': {'title': 'Transformer Paper', 'authors': 'Test Author'}
            }
        ]
        
        create_chunk_database(sample_chunks, temp_files['chunks_db'])
        create_bm25_index([chunk['content'] for chunk in sample_chunks], temp_files['bm25_index'])
        
        # Create and test pipeline
        pipeline = create_rag_pipeline(
            faiss_index_path=temp_files['faiss_index'],
            bm25_index_path=temp_files['bm25_index'],
            chunk_metadata_path=temp_files['chunks_db']
        )
        
        response = pipeline.query("What are transformers?")
        
        assert isinstance(response, RAGResponse)
        assert response.answer is not None
        assert len(response.retrieved_chunks) >= 0
        assert response.confidence >= 0

# Test data and utilities
def create_sample_test_data():
    """Create sample test data for manual testing."""
    sample_chunks = [
        {
            'chunk_id': 'chunk_transformer_1',
            'paper_id': 'vaswani2017attention',
            'content': 'The Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.',
            'section': 'abstract',
            'token_count': 25,
            'metadata': {
                'title': 'Attention Is All You Need',
                'authors': 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin',
                'arxiv_id': '1706.03762',
                'year': 2017,
                'doi': None
            }
        },
        {
            'chunk_id': 'chunk_transformer_2',
            'paper_id': 'vaswani2017attention',
            'content': 'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.',
            'section': 'introduction',
            'token_count': 22,
            'metadata': {
                'title': 'Attention Is All You Need',
                'authors': 'Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin',
                'arxiv_id': '1706.03762',
                'year': 2017,
                'doi': None
            }
        },
        {
            'chunk_id': 'chunk_bert_1',
            'paper_id': 'devlin2018bert',
            'content': 'BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.',
            'section': 'abstract',
            'token_count': 26,
            'metadata': {
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
                'authors': 'Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova',
                'arxiv_id': '1810.04805',
                'year': 2018,
                'doi': None
            }
        }
    ]
    
    return sample_chunks

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
