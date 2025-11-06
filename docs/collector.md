# Paper Collector Module

The Paper Collector module is a comprehensive system for collecting research papers from ArXiv and Google Scholar with automatic deduplication, PDF downloading, and metadata management.

## Overview

The Paper Collector provides:
- **ArXiv API Integration**: Fast, reliable access to ArXiv's database
- **Google Scholar Scraping**: Fallback option for broader paper coverage
- **Automatic Deduplication**: Prevents duplicate papers using ID, DOI, and content hashing
- **PDF Management**: Downloads and organizes PDF files locally
- **SQLite Database**: Efficient metadata storage and searching
- **Retry Logic**: Robust error handling with exponential backoff
- **Comprehensive Logging**: Full audit trail of collection activities

## Quick Start

### Installation

```bash
# Install required dependencies
pip install requests scholarly sqlite3

# Optional: For better Scholar scraping with proxy support
pip install scholarly[proxy]
```

### Basic Usage

```python
from collector import PaperCollector

# Initialize the collector
collector = PaperCollector(
    data_dir="data",           # Directory for storing papers and metadata
    db_path="papers.db"        # SQLite database path
)

# Search for papers
results = collector.search(
    query="large language models",
    max_results=50,
    download_pdfs=True
)

print(f"Found {results['total_found']} papers")
print(f"Added {results['papers_added']} new papers")
print(f"Downloaded {results['pdfs_downloaded']} PDFs")

# Get recent papers from ArXiv category
recent_results = collector.update_recent(
    category="cs.AI",          # ArXiv category
    days_back=7,               # Look back 7 days
    download_pdfs=True
)

# Search locally stored papers
local_papers = collector.search_local("transformer models")
for paper in local_papers:
    print(f"Title: {paper['title']}")
    print(f"Authors: {', '.join(paper['authors'])}")
    print(f"Source: {paper['source']}")
    print("---")

# Get collector statistics
stats = collector.get_stats()
print(f"Total papers: {stats['total_papers']}")
print(f"ArXiv papers: {stats['by_source'].get('arxiv', 0)}")
print(f"Scholar papers: {stats['by_source'].get('scholar', 0)}")
```

## Module Components

### 1. PaperCollector (Main Orchestrator)

The main class that coordinates all collection activities.

#### Constructor Parameters

```python
PaperCollector(
    data_dir="data",                    # Base directory for data storage
    db_path="papers.db",               # SQLite database path
    use_scholar_proxy=False,           # Enable proxy rotation for Scholar
    retry_config=None                  # Custom retry configuration
)
```

#### Key Methods

##### `search(query, max_results=100, download_pdfs=True, use_scholar_fallback=True)`

Search for papers using ArXiv API with optional Scholar fallback.

**Parameters:**
- `query` (str): Search query
- `max_results` (int): Maximum number of results
- `download_pdfs` (bool): Whether to download PDF files
- `use_scholar_fallback` (bool): Use Scholar if ArXiv returns insufficient results

**Returns:** Dictionary with search results and statistics

```python
{
    'query': 'machine learning',
    'timestamp': datetime.now(),
    'arxiv_papers': 25,
    'scholar_papers': 15,
    'total_found': 40,
    'papers_added': 35,
    'papers_skipped': 5,  # Duplicates
    'pdfs_downloaded': 30,
    'errors': []
}
```

##### `update_recent(category="cs.AI", days_back=7, download_pdfs=True)`

Fetch latest papers from an ArXiv category.

**Parameters:**
- `category` (str): ArXiv category (e.g., 'cs.AI', 'cs.LG', 'cs.CL')
- `days_back` (int): Number of days to look back
- `download_pdfs` (bool): Whether to download PDF files

**Popular ArXiv Categories:**
- `cs.AI`: Artificial Intelligence
- `cs.LG`: Machine Learning  
- `cs.CL`: Computation and Language (NLP)
- `cs.CV`: Computer Vision and Pattern Recognition
- `cs.NE`: Neural and Evolutionary Computing
- `stat.ML`: Machine Learning (Statistics)

##### `search_local(query, source=None, limit=100)`

Search locally stored papers.

**Parameters:**
- `query` (str): Search query (searches title and abstract)
- `source` (str, optional): Filter by source ('arxiv' or 'scholar')
- `limit` (int): Maximum number of results

### 2. ArxivClient

Direct interface to the ArXiv API.

```python
from collector import ArxivClient

client = ArxivClient()

# Basic search
papers = client.search("quantum computing", max_results=20)

# Search by category and date range
recent_papers = client.get_recent_papers(
    category="quant-ph",
    days_back=14,
    max_results=50
)

# Search by author
author_papers = client.search_by_author("Geoffrey Hinton")

# Advanced search
advanced_papers = client.search_advanced(
    title="transformer",
    author="Vaswani",
    category="cs.CL"
)

# Get specific paper by ID
paper = client.get_by_id("1706.03762")  # "Attention Is All You Need"
```

### 3. ScholarClient

Google Scholar scraping interface (requires `scholarly` package).

```python
from collector import ScholarClient, scholar_available

if scholar_available():
    client = ScholarClient(
        use_proxy=True,           # Enable proxy rotation
        delay_range=(5, 10)       # Random delay between requests
    )
    
    # Search papers
    papers = client.search("deep learning", max_results=30)
    
    # Search by author
    author_papers = client.search_author("Yann LeCun")
    
    # Get recent papers
    recent_papers = client.search_recent(
        query="computer vision",
        start_year=2023
    )
else:
    print("Scholar client not available - install scholarly package")
```

### 4. PaperDatabase

SQLite database interface for metadata storage.

```python
from collector import PaperDatabase

db = PaperDatabase("papers.db")

# Check if paper exists
exists = db.paper_exists(
    paper_id="2301.07041",
    doi="10.1000/example",
    hash_value="content_hash_123"
)

# Add paper
paper_data = {
    'id': 'example_001',
    'title': 'Example Paper',
    'authors': ['Author One', 'Author Two'],
    'abstract': 'Paper abstract...',
    'source': 'arxiv',
    'hash': 'unique_hash'
}
success = db.add_paper(paper_data)

# Search papers
results = db.search_papers(
    query="machine learning",
    source="arxiv",
    limit=50
)

# Get statistics
stats = db.get_stats()
```

## Configuration

### Retry Configuration

Customize retry behavior with exponential backoff:

```python
from collector import PaperCollector, RetryConfig

retry_config = RetryConfig(
    max_retries=5,      # Maximum number of retry attempts
    base_delay=2.0,     # Base delay in seconds
    max_delay=120.0     # Maximum delay in seconds
)

collector = PaperCollector(retry_config=retry_config)
```

### Logging Configuration

Configure logging to monitor collection activities:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('collector.log'),
        logging.StreamHandler()
    ]
)

# Create collector
collector = PaperCollector()
```

### Directory Structure

The collector creates the following directory structure:

```
data/
├── raw/
│   ├── papers/           # Downloaded PDF files
│   │   ├── 2301.07041.pdf
│   │   ├── scholar_abc123.pdf
│   │   └── ...
│   └── metadata/         # JSON metadata files
│       ├── 2301.07041_metadata.json
│       ├── scholar_abc123_metadata.json
│       └── ...
└── papers.db            # SQLite database
```

## Error Handling

The collector includes comprehensive error handling:

### Network Errors
- Automatic retries with exponential backoff
- Graceful fallback between ArXiv and Scholar
- Timeout handling for long-running requests

### Rate Limiting
- Built-in rate limiting for ArXiv API (3-second delays)
- Configurable delays for Scholar scraping (5-10 second range)
- Proxy rotation support for Scholar (optional)

### Data Validation
- PDF content type verification
- Metadata completeness checking
- Database constraint enforcement

### Recovery Strategies
- Continue processing after individual paper failures
- Fallback to metadata-only storage if PDF download fails
- Transaction rollback for database errors

## Performance Optimization

### Batch Processing
```python
# Process papers in batches for better performance
collector = PaperCollector()

# Large-scale collection
results = collector.search(
    query="deep learning OR machine learning OR neural networks",
    max_results=1000,
    download_pdfs=False  # Skip PDFs for faster processing
)

# Download PDFs separately if needed
for paper_id in successful_paper_ids:
    paper = collector.database.get_paper(paper_id)
    if paper['pdf_url']:
        pdf_path = collector._download_pdf(paper['pdf_url'], paper_id)
        if pdf_path:
            collector.database.update_pdf_path(paper_id, pdf_path)
```

### Database Optimization
```python
# Use database indices for faster searches
papers = collector.search_local(
    query="transformer",
    source="arxiv",  # Use indexed column
    limit=100
)

# Regular maintenance
stats = collector.get_stats()
print(f"Database has {stats['total_papers']} papers")
```

## Integration Examples

### Daily Update Script
```python
#!/usr/bin/env python3
"""Daily update script for collecting recent papers."""

import logging
from collector import PaperCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/paper_collector.log'),
        logging.StreamHandler()
    ]
)

def main():
    collector = PaperCollector(
        data_dir="/data/research_papers",
        db_path="/data/papers.db"
    )
    
    # Update multiple categories
    categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV']
    
    for category in categories:
        logging.info(f"Updating category: {category}")
        results = collector.update_recent(
            category=category,
            days_back=1,  # Daily update
            download_pdfs=True
        )
        logging.info(f"Category {category}: {results['papers_added']} new papers")

if __name__ == "__main__":
    main()
```

### Research Query Script
```python
#!/usr/bin/env python3
"""Script for collecting papers on specific research topics."""

from collector import PaperCollector

def collect_research_papers(topics, max_per_topic=50):
    collector = PaperCollector()
    
    all_results = {}
    for topic in topics:
        print(f"Collecting papers on: {topic}")
        results = collector.search(
            query=topic,
            max_results=max_per_topic,
            download_pdfs=True
        )
        all_results[topic] = results
        print(f"  Found: {results['total_found']}")
        print(f"  Added: {results['papers_added']}")
        print(f"  PDFs: {results['pdfs_downloaded']}")
        print()
    
    return all_results

# Example usage
topics = [
    "large language models",
    "diffusion models",
    "reinforcement learning",
    "computer vision transformers"
]

results = collect_research_papers(topics)

# Print summary
total_papers = sum(r['papers_added'] for r in results.values())
print(f"Total papers collected: {total_papers}")
```

## Troubleshooting

### Common Issues

#### Scholar Client Not Available
```
Error: scholarly library not available
Solution: pip install scholarly
```

#### PDF Download Failures
```python
# Check PDF availability before downloading
papers = collector.search("query", download_pdfs=False)
pdf_urls = [p['pdf_url'] for p in papers if p.get('pdf_url')]
print(f"PDFs available: {len(pdf_urls)}/{len(papers)}")
```

#### Rate Limiting Issues
```python
# Increase delays for Scholar scraping
collector = PaperCollector()
collector.scholar_client.delay_range = (10, 20)  # Longer delays
```

#### Database Lock Errors
```python
# Use with statement for proper connection handling
with sqlite3.connect("papers.db") as conn:
    # Database operations
    pass
```

### Performance Monitoring

```python
# Monitor collection performance
import time

start_time = time.time()
results = collector.search("query", max_results=100)
end_time = time.time()

print(f"Collection took {end_time - start_time:.2f} seconds")
print(f"Rate: {results['total_found'] / (end_time - start_time):.2f} papers/second")

# Check database growth
stats = collector.get_stats()
print(f"Database size: {stats['total_papers']} papers")
print(f"Success rate: {stats['papers_with_pdfs'] / stats['total_papers'] * 100:.1f}% PDFs")
```

## API Reference

### Paper Metadata Schema

Each paper in the database contains the following fields:

```python
{
    'id': str,                    # Unique identifier
    'title': str,                 # Paper title
    'authors': List[str],         # List of author names
    'abstract': str,              # Paper abstract
    'doi': Optional[str],         # DOI if available
    'published_date': datetime,   # Publication date
    'updated_date': datetime,     # Last update date (ArXiv)
    'venue': Optional[str],       # Publication venue (Scholar)
    'year': Optional[int],        # Publication year
    'categories': List[str],      # ArXiv categories
    'source': str,                # 'arxiv' or 'scholar'
    'url': str,                   # Paper URL
    'pdf_url': Optional[str],     # PDF download URL
    'pdf_path': Optional[str],    # Local PDF file path
    'citations': int,             # Citation count (Scholar)
    'hash': str,                  # Content hash for deduplication
    'created_at': datetime,       # Database insertion time
    'updated_at': datetime        # Last database update
}
```

## License

This module is part of the Research Copilot project. See the main project license for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test cases for usage examples
3. Open an issue on the project repository
