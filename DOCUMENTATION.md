# Research Copilot 🔬

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

Production-ready research assistant powered by RAG (Retrieval-Augmented Generation) with microservices architecture, advanced NLP, comprehensive monitoring, and a web interface.

## ✨ Features

- **📚 Paper Collection** - Automated collection from ArXiv & Google Scholar
- **🔗 Citation Tracking** - Advanced citation extraction and analysis
- **❓ Question-Answering** - RAG-powered Q&A with semantic + keyword retrieval
- **📄 Summarization** - Intelligent document summarization
- **🏗️ Microservices** - RAG, LLM, Embeddings, and Storage services
- **🔒 Security** - JWT auth, rate limiting, input validation
- **⚡ Performance** - Async processing, Redis caching
- **📊 Monitoring** - Prometheus, Grafana, Loki observability
- **🐳 Docker Ready** - Complete containerization with Docker Compose
- **🌐 Web Interface** - Interactive Streamlit application

## 🚀 Quick Start

### ⚡ Fastest Way - Docker (5 minutes)

```bash
git clone https://github.com/youtmaj1/Research-Copilot.git
cd Research-Copilot
docker-compose up -d
```

Access:
- **Web UI:** http://localhost:8501
- **API:** http://localhost:8000/docs
- **Monitoring:** http://localhost:3000 (Grafana)

### 🐍 Python Setup (Development)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### 🌐 Run on Another Machine

Share with others or run on your Ubuntu server:

```bash
# Option 1: Docker Compose (Recommended)
docker-compose up -d
# Access: http://<your-ip>:8501

# Option 2: ngrok for remote access
pip install pyngrok
python -c "from pyngrok import ngrok; print(ngrok.connect(8501))"
# Share the public URL

# Option 3: systemd service (Ubuntu)
sudo systemctl start research-copilot
```

## 📁 Project Structure

```
├── collector/               # Paper collection (ArXiv, Scholar)
├── qa/                     # RAG Q&A system
├── services/               # Microservices (4 independent services)
├── citation_tracker/       # Citation extraction & analysis
├── summarizer/             # Document summarization
├── app.py                  # Streamlit web UI
├── production_api.py       # FastAPI gateway
├── docker-compose.yml      # Container orchestration
├── examples/               # Demo scripts
├── tests/                  # Test suite
├── monitoring/             # Prometheus, Grafana, Loki configs
└── config/                 # Configuration files
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

## 🔧 Core Usage

### Paper Collection
```python
from collector import PaperCollector

collector = PaperCollector()
results = collector.search('machine learning', max_results=50)
```

### RAG Question-Answering
```python
from qa import create_rag_pipeline

rag = create_rag_pipeline()
answer = rag.query("What are transformers in NLP?")
```

### Extract Citations
```python
from citation_tracker import CitationExtractor

extractor = CitationExtractor()
citations = extractor.extract('paper.pdf')
```

## 🏗️ Architecture

**Service Ports:**
- FastAPI Gateway: 8000
- Streamlit UI: 8501
- RAG Service: 8001
- LLM Service: 8002
- Embedding Service: 8003
- Storage Service: 8004

**Data Layer:**
- PostgreSQL: 5432 (paper metadata, vectors)
- Redis: 6379 (caching, sessions)
- Ollama: 11434 (local LLM - phi4-mini:3.8b)

**Monitoring:**
- Prometheus: 9090
- Grafana: 3000
- Loki: 3100

## 🧪 Testing

Run tests:
```bash
# All tests
python -m pytest tests/

# Specific test suite
python -m pytest tests/test_qa.py -v

# With coverage
pytest --cov=qa tests/
```

## 🔒 Security

- Credentials stored in `.env.production` (git-ignored)
- JWT token-based authentication
- Input validation and sanitization
- Rate limiting on API endpoints
- Circuit breaker for service resilience

## 📦 Dependencies

**Core:**
- FastAPI 0.104.1
- Streamlit
- SQLAlchemy
- asyncpg (PostgreSQL)
- Redis

**ML/NLP:**
- Ollama (phi4-mini:3.8b LLM)
- FAISS (vector search)
- scikit-learn (BM25 keyword search)

**DevOps:**
- Docker & Docker Compose
- Prometheus (metrics)
- Grafana (dashboards)
- Loki (logging)

See `requirements.txt` for full dependency list.

## 🌐 Default LLM

- **Model:** Ollama phi4-mini:3.8b
- **Size:** 3.8 billion parameters
- **Language:** English
- **Type:** Lightweight instruction-tuned model
- **Hardware:** CPU-optimized (no GPU required)

## 📚 Documentation

For detailed setup and deployment guides, refer to:
- `docs/collector.md` - Paper collection API
- `docs/qa.md` - Q&A system documentation
- `examples/` - Demo scripts and usage examples

## 🚀 Production Deployment

```bash
# Using systemd on Ubuntu
sudo cp research-copilot.service /etc/systemd/system/
sudo systemctl enable research-copilot
sudo systemctl start research-copilot

# Using Docker in production
docker-compose -f docker-compose.yml up -d
```

## 📊 Monitoring

Access Grafana dashboard:
```
http://localhost:3000
Username: admin
Password: admin
```

Query metrics in Prometheus:
```
http://localhost:9090
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 👨‍💼 Author

Damian - [GitHub](https://github.com/youtmaj1)

## 🆘 Troubleshooting

**Docker not starting?**
```bash
docker-compose logs -f
```

**Port already in use?**
```bash
# Change ports in docker-compose.yml or:
lsof -i :8501
kill -9 <PID>
```

**Permission denied on /data?**
```bash
sudo chown -R $(whoami):$(whoami) data/
```

---

**Get started now!** → `docker-compose up -d` then open http://localhost:8501

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
