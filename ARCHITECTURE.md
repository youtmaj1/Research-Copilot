# 🎯 Setup Decision Tree & Architecture Guide

## Which Setup Method Should You Choose?

```
                    CHOOSE YOUR SETUP METHOD
                            |
                    ________|________
                   |                 |
            Want Docker?         Want Python Only?
            YES ↓               YES ↓
                |                   |
         Have Docker?          Have Ollama?
         NO → Install first    NO → Install first
             (5 min)               (10 min)
                |                   |
           Run 1 command       Run 1 command
                |                   |
      docker-compose up -d    streamlit run app.py
                |                   |
           DONE!                 DONE!
          (2 min wait)          (Start immediately)
```

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    RESEARCH COPILOT SYSTEM                   │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────┐                                          │
│  │   Web Browser  │                                          │
│  │ (Port 8501)    │                                          │
│  └────────┬───────┘                                          │
│           │                                                  │
│  ┌────────▼──────────────┐                                  │
│  │   Streamlit Frontend   │ (app.py)                         │
│  │ ✓ Search Papers        │                                  │
│  │ ✓ Ask Questions        │                                  │
│  │ ✓ View Results         │                                  │
│  └────────┬──────────────┘                                  │
│           │                                                  │
│  ┌────────▼──────────────────┐                              │
│  │  FastAPI Gateway           │ (Port 8000)                 │
│  │  ✓ Authentication          │ production_api.py           │
│  │  ✓ Rate Limiting           │                             │
│  │  ✓ Request Routing         │                             │
│  └────────┬───────────────────┘                             │
│           │                                                  │
│    ┌──────┴────────────────────────────┬─────────────┐     │
│    │                                    │             │     │
│    ▼                                    ▼             ▼     │
│ ┌──────────┐  ┌──────────┐  ┌────────────────┐  ┌────────┐ │
│ │  RAG     │  │  LLM     │  │  Embedding     │  │Storage │ │
│ │Service   │  │Service   │  │  Service       │  │Service │ │
│ │(8001)    │  │(8002)    │  │  (8003)        │  │(8004)  │ │
│ └────┬─────┘  └────┬─────┘  └────┬───────────┘  └───┬────┘ │
│      │             │             │                   │      │
│      │  ┌─────────────────┐     │                   │      │
│      │  │  OLLAMA LLM     │     │                   │      │
│      │  │  (Port 11434)   │     │                   │      │
│      │  │  phi4-mini:3.8b │     │                   │      │
│      │  └─────────────────┘     │                   │      │
│      │                          │                   │      │
│      └──────────────┬──────────────────────────────┘      │
│                     │                                       │
│      ┌──────────────┴─────────────┐                        │
│      │                            │                        │
│      ▼                            ▼                        │
│ ┌─────────────┐            ┌──────────────┐              │
│ │ PostgreSQL  │            │    Redis     │              │
│ │ (Port 5432) │            │ (Port 6379)  │              │
│ │ Papers DB   │            │  Caching     │              │
│ │ Metadata    │            │  Sessions    │              │
│ └─────────────┘            └──────────────┘              │
│                                                           │
│  ┌────────────────────────────────────────────┐         │
│  │        MONITORING STACK                    │         │
│  │ ✓ Prometheus (Port 9090) - Metrics       │         │
│  │ ✓ Grafana (Port 3000) - Dashboards       │         │
│  │ ✓ Loki - Log Aggregation                 │         │
│  └────────────────────────────────────────────┘         │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## 📊 Setup Comparison

| Feature | Docker | Manual Python | Manual Lite |
|---------|--------|----------------|------------|
| **Setup Time** | 5 min | 20 min | 10 min |
| **All Services** | ✅ Yes | ⚠️ Manual | ❌ Minimal |
| **Database** | ✅ Auto | ⚠️ Manual | ⚠️ SQLite |
| **Monitoring** | ✅ Full | ❌ No | ❌ No |
| **Production Ready** | ✅ Yes | ⚠️ Dev | ❌ Testing |
| **Easy Reset** | ✅ Yes | ❌ Complex | ⚠️ Moderate |
| **Dependency Hell** | ✅ None | ⚠️ Possible | ✅ Minimal |
| **Recommended** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

---

## 🚀 Setup Flow Diagram

### Docker Setup Flow
```
┌─────────────────────────────────────────┐
│  1. Clone Repository                    │
│     git clone ...                       │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  2. Install Docker (if needed)          │
│     curl -fsSL https://get.docker.com   │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  3. Run Docker Compose                  │
│     docker-compose up -d                │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  4. Wait 30-60 seconds                  │
│     Services initialize...              │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  5. Access Web Interface                │
│     http://localhost:8501               │
└─────────────────────────────────────────┘
        ✅ READY TO USE!
```

### Manual Python Setup Flow
```
┌─────────────────────────────────────────┐
│  1. Clone Repository                    │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  2. Create Virtual Environment          │
│     python3 -m venv .venv               │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  3. Activate Environment                │
│     source .venv/bin/activate           │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  4. Install Ollama                      │
│     curl https://ollama.ai/install.sh   │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  5. Pull LLM Model                      │
│     ollama pull phi4-mini:3.8b          │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  6. Install Dependencies                │
│     pip install -r requirements.txt     │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  7. Run Streamlit                       │
│     streamlit run app.py                │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  8. Access Web Interface                │
│     http://localhost:8501               │
└─────────────────────────────────────────┘
        ✅ READY TO USE!
```

---

## 🔍 Component Details

### Frontend (Streamlit - Port 8501)
```
User Interface
├── Search Papers
│   ├── ArXiv Search
│   └── Local Search
├── Ask Questions
│   ├── Query Input
│   ├── Context Preview
│   └── Answer Display
├── Citations
│   ├── Extract Citations
│   └── Citation Network
└── Analytics
    ├── Search History
    └── Performance Metrics
```

### Backend (FastAPI - Port 8000)
```
API Gateway
├── /search - Paper search
├── /ask - Q&A endpoint
├── /citations - Citation extraction
├── /health - System health
├── /metrics - Prometheus metrics
└── /docs - Interactive API docs
```

### Services
```
RAG Service (Port 8001)
├── Document Retrieval
├── Vector Search
├── Keyword Search
└── Context Building

LLM Service (Port 8002)
├── Model Management
├── Prompt Optimization
├── Response Generation
└── Streaming Support

Embedding Service (Port 8003)
├── Text Encoding
├── Similarity Scoring
└── Vector Storage

Storage Service (Port 8004)
├── Database Operations
├── Caching Layer
├── Session Management
└── Data Persistence
```

### Data Layer
```
PostgreSQL (Port 5432)
├── Papers Table
├── Metadata Table
├── Users Table
└── Sessions Table

Redis (Port 6379)
├── Response Cache
├── Session Store
├── Rate Limit Counters
└── Job Queue

Ollama (Port 11434)
└── LLM Models
    └── phi4-mini:3.8b (default)
```

### Monitoring Stack
```
Prometheus (Port 9090)
├── API Metrics
├── Database Metrics
├── Service Health
└── Resource Usage

Grafana (Port 3000)
├── Dashboard 1: API Performance
├── Dashboard 2: System Resources
├── Dashboard 3: Error Rates
└── Dashboard 4: Request Patterns

Loki
├── Log Aggregation
├── Query Interface
└── Log Analysis
```

---

## 📋 Port Reference

```
┌─────────────────────────────────────────────┐
│          ALL PORTS REFERENCE                │
├─────────────────────────────────────────────┤
│  8501   Streamlit Web UI  (MAIN)            │
│  8000   FastAPI Gateway   (API)             │
│  8001   RAG Service                         │
│  8002   LLM Service                         │
│  8003   Embedding Service                   │
│  8004   Storage Service                     │
│  5432   PostgreSQL        (DB)              │
│  6379   Redis             (Cache)           │
│ 11434   Ollama            (LLM)             │
│  9090   Prometheus        (Metrics)         │
│  3000   Grafana           (Dashboards)      │
│  3100   Loki              (Logs)            │
└─────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### User Search Query Flow
```
User Input (Web UI)
    ↓
[Streamlit] Frontend
    ↓
[FastAPI] Gateway (Port 8000)
    ↓
[RAG Service] Retrieve Documents
    ├→ [Embedding Service] Convert to vectors
    ├→ [PostgreSQL] Search metadata
    └→ [Redis] Check cache
    ↓
[LLM Service] Generate Answer
    ├→ [Ollama] Call phi4-mini:3.8b
    └→ [Redis] Cache response
    ↓
Format Response
    ↓
Return to User (Web UI)
    ↓
[Prometheus] Log Metrics
    ↓
[Grafana] Display in Dashboard
```

### Paper Collection Flow
```
Collection Request
    ↓
[Collector] Module
    ├→ [ArXiv Client] API calls
    └→ [Scholar Client] Web scraping
    ↓
Validate & Deduplicate
    ├→ Check duplicates
    └→ Extract metadata
    ↓
[PostgreSQL] Store Papers
    ├→ Insert to database
    └→ Index for search
    ↓
[Embedding Service] Generate vectors
    ↓
[Vector Store] Index embeddings
    ↓
Completion
    ↓
[Grafana] Show collection stats
```

---

## ✅ Quick Verification Checklist

### After Docker Setup:
- [ ] Run `docker-compose ps` - All containers showing "Up"
- [ ] Access http://localhost:8501 - Web UI loads
- [ ] Access http://localhost:8000/docs - API docs visible
- [ ] Access http://localhost:3000 - Grafana dashboard loads
- [ ] Click "Search" in web UI - No errors
- [ ] Check `docker-compose logs api` - No error messages

### After Manual Setup:
- [ ] `source .venv/bin/activate` - Venv activated
- [ ] `ollama serve` running in background
- [ ] `streamlit run app.py` - App starts
- [ ] Browser opens to http://localhost:8501
- [ ] Web UI loads and responds

---

## 🎓 Learning Path

1. **Start:** Docker setup (easiest)
2. **Explore:** Use web interface
3. **Understand:** Read architecture in docs
4. **Develop:** Manual setup for customization
5. **Deploy:** Production setup with DEPLOYMENT.md

---

**Now ready?** Check [QUICKSTART.md](QUICKSTART.md) or [UBUNTU_SETUP.md](UBUNTU_SETUP.md)!
