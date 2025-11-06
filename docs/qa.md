# Module 4: Question Answerer (RAG Pipeline) Documentation

## Overview

The Question Answerer module implements a sophisticated Retrieval-Augmented Generation (RAG) pipeline for querying research papers. It combines semantic search, keyword matching, and large language model generation to provide accurate, cited answers to research questions.

## Architecture

```
User Query → Query Rewriter → Hybrid Retriever → RAG Pipeline → Answer Formatter → Response
                ↓                    ↓                 ↓                ↓
            LLM Expansion      FAISS + BM25      Ollama LLM      JSON/MD/HTML
```

## Components

### 1. HybridRetriever (`qa/retriever.py`)

Combines FAISS semantic similarity search with BM25 keyword search for comprehensive document retrieval.

**Key Features:**
- FAISS vector similarity search using sentence embeddings
- BM25 keyword search for exact term matching
- Hybrid scoring that combines both approaches
- SQLite-based chunk metadata management
- Configurable search weights and parameters

**Usage:**
```python
from qa.retriever import HybridRetriever

retriever = HybridRetriever(
    faiss_index_path="data/processed/faiss_index.bin",
    bm25_index_path="data/processed/bm25_index.pkl",
    chunk_metadata_path="data/processed/chunks.db",
    embedding_model_name="all-MiniLM-L6-v2",
    alpha=0.7  # 70% FAISS, 30% BM25
)

# Retrieve relevant chunks
chunks = retriever.retrieve("transformer attention mechanism", k=5)
```

### 2. RAGPipeline (`qa/rag.py`)

Core RAG implementation that orchestrates retrieval and generation.

**Key Features:**
- Ollama LLM integration with custom wrapper
- Context preparation and prompt engineering
- Citation extraction and validation
- Confidence scoring based on retrieval quality
- Support for multi-hop queries via knowledge graph integration

**Usage:**
```python
from qa.rag import create_rag_pipeline

pipeline = create_rag_pipeline(
    faiss_index_path="data/processed/faiss_index.bin",
    bm25_index_path="data/processed/bm25_index.pkl",
    chunk_metadata_path="data/processed/chunks.db",
    papers_db_path="data/processed/papers.db",
    llm_model="deepseek-coder-v2"
)

# Query the pipeline
response = pipeline.query("What are the main advantages of transformer architectures?")
print(f"Answer: {response.answer}")
print(f"Citations: {response.citations}")
print(f"Confidence: {response.confidence:.2f}")
```

### 3. QueryRewriter (`qa/query_rewriter.py`)

Expands and rewrites user queries to improve retrieval performance.

**Key Features:**
- LLM-based query expansion
- Domain-specific term mapping for ML/AI research
- Synonym expansion for better recall
- Academic phrase normalization
- Configurable expansion strategies

**Usage:**
```python
from qa.query_rewriter import create_academic_query_rewriter

rewriter = create_academic_query_rewriter(
    ollama_model="deepseek-coder-v2",
    enable_llm=True
)

# Expand query
expanded = rewriter.rewrite("transformer architecture")
print(f"Original: transformer architecture")
print(f"Expanded: {expanded}")

# Get detailed expansion info
details = rewriter.get_expansion_details("neural network")
print(f"Added terms: {details.expansion_terms}")
print(f"Confidence: {details.confidence:.2f}")
```

### 4. AnswerFormatter (`qa/formatter.py`)

Formats RAG responses in multiple output formats with proper citations.

**Key Features:**
- JSON, Markdown, and HTML output formats
- Structured citation management with links
- Multiple citation styles (academic, brief, full)
- Citation validation and bibliography generation
- Rich metadata inclusion

**Usage:**
```python
from qa.formatter import AnswerFormatter

formatter = AnswerFormatter(citation_style="academic")

# Format as JSON
json_response = formatter.format_as_json(
    answer, citations, chunks, confidence=0.9, query="What is BERT?"
)

# Format as Markdown
markdown = formatter.format_as_markdown(
    answer, citations, chunks, query="What is BERT?"
)

# Format as HTML with clickable links
html = formatter.format_as_html(
    answer, citations, chunks, query="What is BERT?"
)
```

### 5. Streamlit Frontend (`app.py`)

Interactive web interface for the RAG pipeline.

**Key Features:**
- Real-time query processing with progress tracking
- Configurable retrieval and generation parameters
- Multiple output format display
- Query history and citation management
- Sample queries and help system
- Pipeline status monitoring

## Installation & Setup

### Prerequisites

```bash
# Install Python dependencies
pip install streamlit langchain sentence-transformers faiss-cpu rank-bm25 requests

# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Pull required model
ollama pull deepseek-coder-v2
```

### Data Preparation

1. **Create FAISS Index** (from Module 2 output):
```bash
python -m summarizer.faiss_index --input data/processed/chunks.json --output data/processed/faiss_index.bin
```

2. **Create BM25 Index**:
```python
from qa.retriever import create_bm25_index, create_chunk_database

# Load your chunks data
chunks = load_chunks_from_json("data/processed/chunks.json")

# Create BM25 index
chunk_texts = [chunk['content'] for chunk in chunks]
create_bm25_index(chunk_texts, "data/processed/bm25_index.pkl")

# Create chunk database
create_chunk_database(chunks, "data/processed/chunks.db")
```

3. **Prepare Papers Database** (from Module 1 output):
```sql
-- Ensure papers.db has the required schema
CREATE TABLE IF NOT EXISTS papers_metadata (
    id TEXT PRIMARY KEY,
    title TEXT,
    authors TEXT,
    doi TEXT,
    arxiv_id TEXT,
    published_date DATE,
    abstract TEXT
);
```

## Usage Examples

### Basic Query Processing

```python
from qa.rag import create_rag_pipeline

# Initialize pipeline
pipeline = create_rag_pipeline(
    faiss_index_path="data/processed/faiss_index.bin",
    bm25_index_path="data/processed/bm25_index.pkl",
    chunk_metadata_path="data/processed/chunks.db",
    papers_db_path="data/processed/papers.db"
)

# Process queries
questions = [
    "What are the main advantages of transformer architectures?",
    "How does attention mechanism work in neural networks?",
    "Compare CNNs and transformers for computer vision tasks",
    "What are the latest developments in large language models?"
]

for question in questions:
    response = pipeline.query(question)
    
    print(f"\nQ: {question}")
    print(f"A: {response.answer}")
    print(f"Citations: {response.citations}")
    print(f"Confidence: {response.confidence:.2%}")
    print(f"Processing time: {response.processing_time:.2f}s")
```

### Advanced Features

```python
# Paper-specific queries
paper_summary = pipeline.get_paper_summary("arxiv:1706.03762")
print(f"Summary: {paper_summary.answer}")

# Compare multiple papers
comparison = pipeline.compare_papers(
    ["arxiv:1706.03762", "arxiv:1810.04805"],
    aspect="methodology"
)
print(f"Comparison: {comparison.answer}")

# Batch processing
questions = ["Question 1", "Question 2", "Question 3"]
responses = pipeline.batch_query(questions)
```

### Custom Configuration

```python
from qa.retriever import HybridRetriever
from qa.rag import RAGPipeline
from qa.query_rewriter import QueryRewriter

# Custom retriever with specific weights
retriever = HybridRetriever(
    faiss_index_path="data/processed/faiss_index.bin",
    bm25_index_path="data/processed/bm25_index.pkl",
    chunk_metadata_path="data/processed/chunks.db",
    alpha=0.8,  # More weight on semantic similarity
    embedding_model_name="all-mpnet-base-v2"
)

# Custom RAG pipeline
pipeline = RAGPipeline(
    retriever=retriever,
    llm_model_name="llama2",
    max_chunks=10,
    temperature=0.2
)

# Custom query rewriter
rewriter = QueryRewriter(
    ollama_model="llama2",
    use_llm_expansion=True,
    use_domain_mapping=True
)

# Add custom domain mappings
rewriter.add_domain_mapping("GAN", ["generative adversarial network", "adversarial training"])
rewriter.add_synonym_mapping("robust", ["stable", "reliable", "resilient"])
```

## Web Interface

### Starting the Streamlit App

```bash
# Start the web interface
streamlit run app.py

# Access at http://localhost:8501
```

### Interface Features

1. **Search Interface**:
   - Main query input with search button
   - Advanced options for chunk count and output format
   - Sample queries for quick testing

2. **Configuration Panel**:
   - Model selection (LLM and embedding models)
   - Retrieval settings (hybrid search, max chunks)
   - Citation style configuration
   - Data path configuration

3. **Response Display**:
   - Formatted answers with citations
   - Confidence scores and processing times
   - Expandable retrieved chunks view
   - Multiple output formats (Markdown, JSON, HTML)

4. **Query History**:
   - Recent queries with timestamps
   - Quick access to previous results
   - Confidence tracking

## Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
pytest tests/test_qa.py -v

# Run specific test categories
pytest tests/test_qa.py::TestHybridRetriever -v
pytest tests/test_qa.py::TestRAGPipeline -v
pytest tests/test_qa.py::TestQueryRewriter -v
pytest tests/test_qa.py::TestAnswerFormatter -v
pytest tests/test_qa.py::TestIntegration -v
```

### Test Categories

1. **Unit Tests**:
   - Component initialization
   - Individual method functionality
   - Error handling and edge cases

2. **Integration Tests**:
   - End-to-end pipeline functionality
   - Component interaction
   - Data flow validation

3. **Mock Tests**:
   - LLM API interaction
   - Database operations
   - File system access

## Performance Optimization

### Retrieval Optimization

```python
# Optimize FAISS index
retriever = HybridRetriever(
    faiss_index_path="data/processed/faiss_index.bin",
    bm25_index_path="data/processed/bm25_index.pkl",
    chunk_metadata_path="data/processed/chunks.db",
    alpha=0.7,  # Adjust based on your data
    embedding_model_name="all-MiniLM-L6-v2"  # Faster model
)

# Use focused retrieval
chunks = retriever.retrieve(query, k=5, use_hybrid=True)
```

### Generation Optimization

```python
# Optimize LLM parameters
pipeline = RAGPipeline(
    retriever=retriever,
    llm_model_name="deepseek-coder-v2",
    max_chunks=5,  # Limit context size
    max_context_length=6000,  # Reduce if needed
    temperature=0.1  # Lower for more focused responses
)
```

### Caching

```python
# Implement response caching
import functools
import hashlib

@functools.lru_cache(maxsize=100)
def cached_query(query_hash):
    return pipeline.query(query)

# Use cache
query = "What are transformers?"
query_hash = hashlib.md5(query.encode()).hexdigest()
response = cached_query(query_hash)
```

## Error Handling & Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
```python
# Check Ollama status
import requests
try:
    response = requests.get("http://localhost:11434/api/tags")
    print("Ollama is running")
except:
    print("Start Ollama: ollama serve")
```

2. **Missing Index Files**:
```python
import os

required_files = [
    "data/processed/faiss_index.bin",
    "data/processed/bm25_index.pkl",
    "data/processed/chunks.db"
]

for file in required_files:
    if not os.path.exists(file):
        print(f"Missing: {file}")
```

3. **Memory Issues**:
```python
# Reduce memory usage
pipeline = RAGPipeline(
    retriever=retriever,
    max_chunks=3,  # Reduce chunks
    max_context_length=4000,  # Reduce context
    embedding_model_name="all-MiniLM-L6-v2"  # Smaller model
)
```

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check pipeline statistics
stats = pipeline.get_statistics()
print(f"Pipeline stats: {stats}")

# Validate retrieval
chunks = retriever.retrieve("test query", k=3)
for chunk in chunks:
    print(f"Chunk: {chunk.chunk_id}, Score: {chunk.score}, Source: {chunk.source}")
```

## Best Practices

### Query Design

1. **Specific Questions**: Use specific, focused questions rather than broad topics
2. **Technical Terms**: Include relevant technical terminology
3. **Context**: Provide context when asking comparative questions

### System Configuration

1. **Model Selection**: Choose appropriate models based on your hardware
2. **Index Optimization**: Regularly update and optimize indexes
3. **Caching**: Implement caching for frequently asked questions

### Citation Management

1. **Verification**: Always verify generated citations
2. **Link Generation**: Ensure citation links are accessible
3. **Format Consistency**: Use consistent citation formats across responses

## API Reference

### Core Classes

#### HybridRetriever
- `__init__(faiss_index_path, bm25_index_path, chunk_metadata_path, ...)`
- `retrieve(query, k=5, use_hybrid=True) -> List[RetrievedChunk]`
- `get_chunk_by_id(chunk_id) -> RetrievedChunk`
- `get_paper_chunks(paper_id) -> List[RetrievedChunk]`

#### RAGPipeline
- `__init__(retriever, llm_model_name, ...)`
- `query(question, max_chunks=None, use_query_rewriter=False) -> RAGResponse`
- `batch_query(questions) -> List[RAGResponse]`
- `get_paper_summary(paper_id) -> RAGResponse`
- `compare_papers(paper_ids, aspect) -> RAGResponse`

#### QueryRewriter
- `__init__(ollama_model, use_llm_expansion, ...)`
- `rewrite(query, method="hybrid") -> str`
- `get_expansion_details(query) -> QueryExpansion`
- `add_domain_mapping(term, expansions)`

#### AnswerFormatter
- `__init__(citation_style="academic")`
- `format_as_json(answer, citations, chunks, ...) -> str`
- `format_as_markdown(answer, citations, chunks, ...) -> str`
- `format_as_html(answer, citations, chunks, ...) -> str`

### Data Structures

#### RetrievedChunk
```python
@dataclass
class RetrievedChunk:
    chunk_id: str
    paper_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
```

#### RAGResponse
```python
@dataclass
class RAGResponse:
    answer: str
    citations: List[str]
    retrieved_chunks: List[RetrievedChunk]
    query: str
    timestamp: str
    confidence: float
    processing_time: float
```

## Contributing

1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include unit tests for new features
4. Update documentation for API changes
5. Test with different models and configurations

## License

This module is part of the Research Copilot project and follows the same licensing terms.
