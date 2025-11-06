# ⚡ QUICK START (30 seconds)

## For GitHub Visitors - Get Running in 5 Minutes!

### Option A: Docker (Recommended)
```bash
git clone https://github.com/your-username/research-copilot.git
cd research-copilot
docker-compose up -d
# Open: http://localhost:8501
```

### Option B: Manual Python Setup
```bash
git clone https://github.com/your-username/research-copilot.git
cd research-copilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
# Open: http://localhost:8501
```

### First Time Setup (Ubuntu/Linux):
```bash
# 1. Install Docker & Docker Compose
curl -fsSL https://get.docker.com | sudo sh
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 2. Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# 3. Clone and run
git clone https://github.com/your-username/research-copilot.git
cd research-copilot
docker-compose up -d

# 4. Wait 30 seconds for services to start
# 5. Open browser to http://localhost:8501
```

## 🌐 Accessing the Application

| Component | URL | Purpose |
|-----------|-----|---------|
| **Web UI** | http://localhost:8501 | Main interface |
| **API Docs** | http://localhost:8000/docs | Interactive API |
| **Monitoring** | http://localhost:3000 | Grafana dashboards (admin/admin) |
| **Metrics** | http://localhost:9090 | Prometheus |

## 🔧 Common Commands

```bash
# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Stop
docker-compose down

# Full reset
docker-compose down -v && docker-compose up -d
```

## 📝 What You Can Do

1. **Search Papers** - Find research papers from ArXiv
2. **Ask Questions** - Use RAG to ask about papers
3. **Track Citations** - Extract and analyze citations
4. **Summarize** - Get paper summaries
5. **Monitor** - View system metrics

## 🚨 Troubleshooting

**Services won't start?**
```bash
docker-compose logs
```

**Port already in use?**
```bash
# Kill process using port
sudo lsof -i :8501
sudo kill -9 <PID>
```

**Out of memory?**
```bash
# Check usage
docker stats
```

For more help, see **UBUNTU_SETUP.md** in the repository.

---

**Ready to try it out?** Start with Docker Compose above! ⬆️
