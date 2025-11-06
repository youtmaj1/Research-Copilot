# Module 3: Cross-Referencer

## Overview

The Cross-Referencer module is a comprehensive system for analyzing relationships between research papers through citation extraction, semantic similarity analysis, and knowledge graph construction. It serves as the relationship detection and network analysis component of the Research Copilot project.

## Features

### 🔍 Citation Extraction
- **PDF Processing**: Extract citations from reference sections using PyMuPDF
- **Multi-format Parsing**: Support for various citation formats and styles
- **Smart Matching**: Match citations to known papers using DOI, arXiv ID, and title similarity
- **Confidence Scoring**: Quantify the reliability of citation matches

### 🎯 Semantic Similarity
- **Embedding-based Analysis**: Use sentence transformers for document embeddings
- **FAISS Integration**: Efficient similarity search and clustering
- **Topic Communities**: Detect research topic clusters and communities
- **Configurable Thresholds**: Adjust similarity sensitivity for different use cases

### 🕷️ Knowledge Graphs
- **NetworkX Integration**: Build and analyze research paper networks
- **Centrality Metrics**: Calculate importance scores (PageRank, betweenness, etc.)
- **Community Detection**: Identify research communities and clusters
- **Multiple Export Formats**: JSON, Neo4j CSV, NetworkX pickle, edge lists

### 🗄️ Database Management
- **SQLite Backend**: Persistent storage for relationships and metadata
- **Comprehensive Schema**: Tables for citations, similarities, and paper metadata
- **Query Views**: Pre-built views for common analysis patterns
- **Statistics Tracking**: Automated relationship statistics and triggers

### 🚀 Pipeline Orchestration
- **End-to-end Workflow**: Coordinate all cross-referencing operations
- **Batch Processing**: Handle large paper collections efficiently
- **Configuration Management**: Flexible pipeline configuration options
- **Progress Tracking**: Monitor processing status and performance

### 🖥️ Command Line Interface
- **Comprehensive CLI**: Full-featured command-line operations
- **Paper Processing**: Process individual papers or batch collections
- **Graph Export**: Export knowledge graphs in multiple formats
- **Database Management**: Initialize, validate, and query databases
- **Search and Analysis**: Find papers and analyze relationships

### 🔗 Module Integration
- **Seamless Workflow**: Connect with Modules 1 (Collector) and 2 (Summarizer)
- **Unified Pipeline**: Complete research workflow from collection to analysis
- **Visualization Export**: Generate data for external visualization tools
- **Comprehensive Reporting**: Integrated analysis reports and insights

## Installation

### Dependencies

```bash
pip install PyMuPDF networkx faiss-cpu sentence-transformers pandas numpy sqlite3
```

### Optional Dependencies

```bash
# For advanced similarity analysis
pip install scikit-learn matplotlib seaborn

# For PDF processing enhancements
pip install pdfplumber

# For graph visualization
pip install plotly networkx[default]
```

## Quick Start

### Basic Usage

```python
from crossref import CrossRefPipeline, CrossRefConfig

# Create configuration
config = CrossRefConfig(
    database_path="research.db",
    output_dir="results/",
    similarity_threshold=0.7
)

# Initialize pipeline
pipeline = CrossRefPipeline(config)

# Process papers
papers = {
    "paper1": {
        "title": "Deep Learning for Computer Vision",
        "authors": ["Smith, J.", "Johnson, A."],
        "abstract": "This paper explores...",
        "year": 2020
    }
}

# Build knowledge graph
graph = pipeline.process_papers(papers)
print(f"Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
```

### Command Line Usage

```bash
# Process papers with PDFs
python -m crossref.cli process papers.json --pdf-dir pdfs/ --output-dir results/

# Add a single paper
python -m crossref.cli add-paper paper123 --title "Machine Learning" --authors "Brown,Davis"

# Export knowledge graph
python -m crossref.cli export --format neo4j,json --output-dir exports/

# Show database statistics
python -m crossref.cli stats --verbose

# Find relationships for a paper
python -m crossref.cli relationships paper123 --limit 10

# Search for papers
python -m crossref.cli search "neural networks" --limit 20
```

### Integrated Workflow

```python
from crossref.integration import create_integrated_pipeline

# Create integrated pipeline
pipeline = create_integrated_pipeline(
    output_dir="integrated_results/",
    crossref_config={
        'similarity_threshold': 0.6,
        'citation_confidence_threshold': 0.5
    }
)

# Run complete research workflow
results = pipeline.run_complete_pipeline(
    search_queries=["machine learning", "deep learning"],
    max_papers_per_query=50,
    include_pdfs=True,
    generate_summaries=True,
    build_knowledge_graph=True
)

print(f"Collected {results['papers_collected']} papers")
print(f"Found {results['relationships_found']} relationships")
```

## Components

### CitationExtractor

Extract and match citations from research papers:

```python
from crossref import CitationExtractor

extractor = CitationExtractor()

# Extract from PDF
citations = extractor.extract_from_pdf("paper.pdf")

# Match to known papers
matches = extractor.match_citations_to_papers(citations, "source_paper", known_papers)

# Filter by confidence
high_confidence = [m for m in matches if m.confidence > 0.8]
```

### SimilarityEngine

Compute semantic similarities between papers:

```python
from crossref import SimilarityEngine

engine = SimilarityEngine(
    embedding_model_name="all-MiniLM-L6-v2",
    similarity_threshold=0.7
)

# Add papers
engine.add_papers(papers)

# Find similar papers
similar = engine.find_similar_papers("paper1", k=5)

# Compute pairwise similarities
similarities = engine.compute_pairwise_similarities(min_similarity=0.5)

# Detect topic communities
communities = engine.detect_topic_communities()
```

### CrossRefGraph

Build and analyze knowledge graphs:

```python
from crossref import CrossRefGraph

graph = CrossRefGraph()

# Add papers as nodes
graph.add_papers(papers)

# Add citation relationships
graph.add_citation_relationships(citation_matches)

# Add similarity relationships
graph.add_similarity_relationships(similarity_results)

# Compute centrality metrics
centrality = graph.compute_centrality_metrics()

# Detect communities
communities = graph.detect_communities()

# Export graph
from crossref import GraphExporter
GraphExporter.export_to_json(graph, "graph.json")
```

## Configuration

### Pipeline Configuration

```python
from crossref import CrossRefConfig

config = CrossRefConfig(
    # Database settings
    database_path="crossref.db",
    
    # Processing thresholds
    similarity_threshold=0.7,
    citation_confidence_threshold=0.5,
    
    # Embedding model
    embedding_model="all-MiniLM-L6-v2",
    
    # Output settings
    output_dir="data/crossref",
    graph_export_formats=["json", "neo4j"],
    
    # Performance settings
    batch_size=10,
    save_intermediate=True,
    
    # Graph construction
    include_author_relationships=True,
    include_topic_clusters=True,
    topic_similarity_threshold=0.8
)
```

### Environment Variables

```bash
export CROSSREF_DATABASE_PATH="path/to/database.db"
export CROSSREF_OUTPUT_DIR="path/to/output"
export CROSSREF_SIMILARITY_THRESHOLD="0.7"
export CROSSREF_EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

## Database Schema

The module uses SQLite with a comprehensive schema:

### Core Tables

- **papers_metadata**: Paper information and metadata
- **crossref_relationships**: All cross-reference relationships
- **citations**: Detailed citation information
- **similarities**: Semantic similarity scores
- **author_relationships**: Author collaboration data

### Views and Indexes

- **all_relationships**: Combined view of all relationship types
- **paper_statistics**: Paper-level statistics and metrics
- **Optimized indexes**: For fast querying and analysis

## Export Formats

### JSON Format
```json
{
  "nodes": [
    {
      "id": "paper1",
      "title": "Deep Learning Paper",
      "authors": ["Smith, J."],
      "year": 2020
    }
  ],
  "edges": [
    {
      "source": "paper1",
      "target": "paper2",
      "relationship": "cites",
      "weight": 0.95
    }
  ]
}
```

### Neo4j CSV Format
- `nodes.csv`: Node data with labels and properties
- `relationships.csv`: Edge data with types and properties

### NetworkX Pickle
- Binary format preserving all graph structure and attributes
- Suitable for further analysis with NetworkX

### Edge List CSV
- Simple CSV format with source, target, and relationship columns
- Compatible with most graph analysis tools

## Testing

### Run Tests

```bash
# Run all tests
python crossref/tests.py

# Run specific test categories
python -m unittest crossref.tests.TestCitationExtractor
python -m unittest crossref.tests.TestSimilarityEngine
python -m unittest crossref.tests.TestCrossRefGraph
python -m unittest crossref.tests.TestCrossRefPipeline
```

### Test Coverage

- **Citation Extraction**: PDF parsing, pattern matching, confidence calculation
- **Similarity Analysis**: Embedding generation, clustering, community detection
- **Graph Construction**: Node/edge creation, centrality metrics, export formats
- **Database Operations**: Schema validation, relationship storage, query performance
- **Pipeline Integration**: End-to-end workflow, error handling, configuration

### Performance Tests

```bash
# Run performance benchmarks
python crossref/tests.py --performance

# Test with large datasets
python crossref/tests.py --integration --size=1000
```

## API Reference

### Core Classes

#### CitationExtractor
- `extract_from_pdf(pdf_path)`: Extract citations from PDF
- `match_citations_to_papers(citations, source_id, known_papers)`: Match citations to papers
- `_parse_citation(text)`: Parse individual citation text
- `_calculate_confidence(citation, paper, match_type)`: Calculate match confidence

#### SimilarityEngine  
- `add_papers(papers, text_field)`: Add papers for similarity analysis
- `compute_pairwise_similarities(min_similarity)`: Compute all pairwise similarities
- `find_similar_papers(paper_id, k)`: Find k most similar papers
- `cluster_papers(n_clusters)`: Cluster papers by similarity
- `detect_topic_communities(threshold)`: Detect topic-based communities

#### CrossRefGraph
- `add_papers(papers)`: Add papers as graph nodes
- `add_citation_relationships(matches)`: Add citation edges
- `add_similarity_relationships(similarities)`: Add similarity edges
- `compute_centrality_metrics()`: Calculate centrality measures
- `detect_communities()`: Detect graph communities
- `get_subgraph(node_ids)`: Extract subgraph

#### CrossRefPipeline
- `process_papers(papers, pdf_paths)`: Process papers through complete pipeline
- `process_single_paper(paper_id, data, pdf_path)`: Process individual paper
- `get_paper_relationships(paper_id)`: Get relationships for specific paper
- `get_pipeline_statistics()`: Get processing statistics

## Advanced Usage

### Custom Similarity Models

```python
from crossref import SimilarityEngine

# Use custom embedding model
engine = SimilarityEngine(
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    similarity_threshold=0.6
)

# Add custom similarity computation
def custom_similarity(text1, text2):
    # Custom similarity logic
    return similarity_score

engine.set_custom_similarity_function(custom_similarity)
```

### Graph Analysis

```python
import networkx as nx
from crossref import CrossRefGraph

graph = CrossRefGraph()
# ... populate graph ...

# Convert to NetworkX for advanced analysis
nx_graph = graph.to_networkx()

# Advanced network analysis
clustering_coeff = nx.clustering(nx_graph)
shortest_paths = nx.shortest_path_length(nx_graph)
communities = nx.community.greedy_modularity_communities(nx_graph)

# Custom centrality measures
custom_centrality = nx.eigenvector_centrality(nx_graph)
```

### Database Queries

```python
import sqlite3
from crossref import CrossRefDatabase

db = CrossRefDatabase("crossref.db")

# Custom queries
with sqlite3.connect(db.db_path) as conn:
    cursor = conn.cursor()
    
    # Find most cited papers
    cursor.execute("""
        SELECT target_paper, COUNT(*) as citation_count
        FROM crossref_relationships 
        WHERE relation = 'cites'
        GROUP BY target_paper
        ORDER BY citation_count DESC
        LIMIT 10
    """)
    
    most_cited = cursor.fetchall()
```

### Visualization Integration

```python
import plotly.graph_objects as go
from crossref import GraphExporter

# Export for Plotly
graph_data = GraphExporter.export_to_dict(graph)

# Create interactive visualization
fig = go.Figure(data=[
    go.Scatter(
        x=[node['x'] for node in graph_data['nodes']],
        y=[node['y'] for node in graph_data['nodes']],
        mode='markers+text',
        text=[node['title'][:20] for node in graph_data['nodes']],
        textposition="middle center"
    )
])

fig.show()
```

## Troubleshooting

### Common Issues

1. **PDF Extraction Errors**
   - Ensure PDFs are text-based, not scanned images
   - Check file permissions and paths
   - Try alternative PDF processing libraries

2. **Similarity Computation Slow**
   - Reduce embedding model size
   - Increase similarity threshold
   - Use batch processing for large datasets

3. **Database Lock Errors**
   - Ensure single writer access
   - Use connection pooling for concurrent reads
   - Check database file permissions

4. **Memory Issues**
   - Process papers in smaller batches
   - Use streaming for large datasets
   - Optimize embedding model choice

### Performance Optimization

```python
# Optimized configuration for large datasets
config = CrossRefConfig(
    batch_size=50,
    similarity_threshold=0.8,  # Higher threshold = fewer comparisons
    embedding_model="all-MiniLM-L6-v2",  # Smaller model
    save_intermediate=False  # Skip intermediate saves
)
```

### Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('crossref')

# Add custom logging
pipeline = CrossRefPipeline(config)
pipeline.logger.setLevel(logging.DEBUG)
```

## Contributing

### Development Setup

```bash
git clone https://github.com/your-repo/research-copilot.git
cd research-copilot/crossref
pip install -e .
pip install -r requirements-dev.txt
```

### Code Style

```bash
# Format code
black crossref/
isort crossref/

# Lint code  
flake8 crossref/
mypy crossref/
```

### Testing

```bash
# Run tests with coverage
pytest --cov=crossref tests/

# Run specific test types
pytest tests/test_citation_extractor.py -v
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this module in your research, please cite:

```bibtex
@software{research_copilot_crossref,
  title={Research Copilot Cross-Reference Module},
  author={Research Copilot Team},
  year={2024},
  url={https://github.com/your-repo/research-copilot}
}
```

## Support

- **Documentation**: Full API documentation available in `docs/`
- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join community discussions for help and ideas
- **Email**: Contact the team at research-copilot@example.com

---

**Module 3: Cross-Referencer** - Connecting research through intelligent relationship detection and knowledge graph construction.
