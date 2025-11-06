# Research Copilot �

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

A production-ready research assistant powered by RAG (Retrieval-Augmented Generation) with microservices architecture, advanced NLP, comprehensive monitoring, and a web interface.

## ✨ Features

- **📚 Paper Collection** - Automated collection from ArXiv & Google Scholar with deduplication
- **🔗 Citation Tracking** - Advanced citation extraction, resolution, and knowledge graphs
- **❓ Question-Answering** - RAG-powered Q&A with semantic and keyword retrieval
- **📄 Summarization** - Intelligent document summarization and key point extraction
- **🏗️ Microservices** - Scalable architecture with independent services (RAG, LLM, Embeddings, Storage)
- **🔒 Enterprise Security** - JWT authentication, rate limiting, input validation
- **⚡ High Performance** - Async processing, Redis caching, streaming responses
- **📊 Monitoring** - Prometheus + Grafana + Loki observability stack
- **🐳 Docker Ready** - Complete containerization with Docker Compose
- **🌐 Web Interface** - Interactive Streamlit web application

## 🚀 Quick Start

### ⚡ Fastest Way (Docker - 5 minutes)

```bash
git clone https://github.com/your-username/research-copilot.git
cd research-copilot
docker-compose up -d
# Open: http://localhost:8501
```

**See [QUICKSTART.md](QUICKSTART.md) for complete setup options!**

### 🌐 Access Points
- **Web UI:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs
- **Monitoring:** http://localhost:3000 (Grafana)
- **Metrics:** http://localhost:9090 (Prometheus)

**Full setup guides:**
- 📖 **[QUICKSTART.md](QUICKSTART.md)** - 30-second quick reference
- 📚 **[UBUNTU_SETUP.md](UBUNTU_SETUP.md)** - Complete Ubuntu/Linux guide
- 🌐 **[RUN_AS_SERVICE.md](RUN_AS_SERVICE.md)** - Run as service for others to access
- 🚀 **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment

## 📁 Project Structure

```
Research-Copilot/
├── collector/                 # Paper collection module (ArXiv & Scholar)
│   ├── arxiv_client.py       # ArXiv API wrapper
│   ├── scholar_client.py     # Google Scholar scraper
│   ├── database.py           # Database operations
│   ├── collector.py          # Main orchestrator
│   └── cli.py               # Command-line interface
│
├── qa/                       # Question-answering system
│   ├── rag.py               # RAG pipeline with Ollama
│   ├── retriever.py         # Hybrid retrieval (FAISS + BM25)
│   ├── query_rewriter.py    # Query enhancement
│   └── formatter.py         # Multi-format output
│
├── services/                 # Microservices (API Gateway pattern)
│   ├── rag_service.py       # Document retrieval service
│   ├── llm_service.py       # Language model service
│   ├── embedding_service.py # Text embedding service
│   ├── storage_service.py   # Database operations
│   └── Dockerfile.*         # Service containers
│
├── citation_tracker/         # Citation extraction & analysis
│   ├── extractor.py         # Citation extraction
│   ├── resolver.py          # Citation resolution
│   ├── cli.py              # CLI interface
│   └── tests/              # Unit tests
│
├── summarizer/               # Document summarization
│   ├── summarizer.py        # Summarization engine
│   └── extractors/          # Text extractors
│
├── monitoring/               # Observability stack
│   ├── prometheus.yml       # Metrics collection
│   ├── grafana.yml          # Dashboard config
│   ├── loki.yml             # Log aggregation
│   └── promtail.yml         # Log forwarding
│
├── config/                   # Configuration
│   ├── production_config.py # Production settings
│   └── ollama_config.py     # Ollama configuration
│
├── examples/                 # Example scripts & demos
│   ├── interactive_demo.py
│   ├── research_demo.py
│   └── simple_demo.py
│
├── tests/                    # Test suite
│   ├── test_collector.py    # Collection tests
│   ├── test_qa.py          # Q&A tests
│   ├── performance/        # Performance tests
│   └── integration/        # Integration tests
│
├── docs/                     # Documentation
│   ├── collector.md         # Collection API
│   └── qa.md               # Q&A system
│
├── data/                     # Data storage
│   ├── raw/papers/         # Downloaded PDFs
│   ├── metadata/           # Paper metadata
│   └── processed/          # Processed data
│
├── app.py                   # Streamlit web interface
├── production_api.py        # FastAPI gateway
├── docker-compose.yml       # Container orchestration
├── requirements.txt         # Dependencies
└── setup.py                # Package setup
```

## 🔧 Installation

### Quick Start (Docker - Recommended)
```bash
docker-compose up -d
# Access: http://localhost:8501
```

### Development Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## � Core Features

### 1. Paper Collection
```python
from collector import PaperCollector

collector = PaperCollector()
results = collector.search('machine learning', max_results=50)
```

### 2. Question-Answering with RAG
```python
from qa import create_rag_pipeline

rag = create_rag_pipeline()
answer = rag.query("What are transformers?")
```

### 3. Citation Analysis
```python
from citation_tracker import CitationExtractor

extractor = CitationExtractor()
citations = extractor.extract("paper.pdf")
```

### 4. Web Interface
```bash
streamlit run app.py
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_collector.py

# With coverage
pytest tests/ --cov=collector --cov=qa
```

## 🚀 Deployment

### Docker Compose
```bash
docker-compose up -d
```

### Production Deployment
```bash
./deploy.sh production your-domain.com
```

See `DEPLOYMENT.md` for comprehensive deployment guide.

## � Documentation

- `docs/collector.md` - Paper collection API
- `docs/qa.md` - Q&A system documentation
- `IMPLEMENTATION_SUMMARY.md` - Architecture overview
- `DEPLOYMENT.md` - Deployment instructions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Add tests for new functionality
4. Run tests: `pytest tests/`
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- ArXiv for research paper access
- Ollama for local LLM capabilities
- FastAPI & Streamlit communities
- Open source research community

## 📞 Support

- Documentation: See `docs/collector.md` for detailed usage
- Examples: Check `examples/usage_examples.py` for code samples
- Issues: Submit bug reports and feature requests via GitHub issues
- Tests: Run the test suite to verify your installation

---

**Happy paper collecting! 📚🤖**
