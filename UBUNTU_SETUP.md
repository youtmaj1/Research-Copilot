# 🚀 How to Run Research Copilot on Ubuntu (Quick Setup Guide)

**For GitHub Visitors: Easy 5-minute setup!**

---

## 📋 PREREQUISITES

Before starting, make sure you have:
- ✅ Ubuntu 20.04+ (or any Linux distro)
- ✅ 8GB RAM minimum (4GB for lightweight mode)
- ✅ 20GB free disk space
- ✅ Internet connection
- ✅ Terminal/SSH access

---

## 🚀 OPTION 1: Docker Compose (RECOMMENDED - 5 minutes)

### **Best for:** Quick demo, full features, production-ready

```bash
# Step 1: Clone the repository
git clone https://github.com/your-username/research-copilot.git
cd research-copilot

# Step 2: Install Docker & Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Step 3: Add user to docker group (so you don't need sudo)
sudo usermod -aG docker $USER
newgrp docker

# Step 4: Copy environment file
cp .env.production .env

# Step 5: Start all services
docker-compose up -d

# Step 6: Wait for services to be ready (30-60 seconds)
docker-compose ps

# Step 7: Access the application
# Web UI: http://localhost:8501
# API Docs: http://localhost:8000/docs
# Grafana (monitoring): http://localhost:3000
# Prometheus: http://localhost:9090
```

### **Verify Everything is Running:**
```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f api

# Test API
curl http://localhost:8000/docs
```

### **Stop Services:**
```bash
docker-compose down
```

---

## 🛠️ OPTION 2: Manual Setup (20 minutes)

### **Best for:** Development, learning, customization

```bash
# Step 1: Clone repository
git clone https://github.com/your-username/research-copilot.git
cd research-copilot

# Step 2: Update system
sudo apt-get update
sudo apt-get upgrade -y

# Step 3: Install Python and dependencies
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
sudo apt-get install -y build-essential libssl-dev libffi-dev
sudo apt-get install -y postgresql postgresql-contrib redis-server

# Step 4: Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Step 5: Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Step 6: Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Step 7: Create database
sudo -u postgres psql -c "CREATE USER research_user WITH PASSWORD 'research_password';"
sudo -u postgres psql -c "ALTER USER research_user CREATEDB;"
sudo -u postgres psql -c "CREATE DATABASE research_copilot OWNER research_user;"

# Step 8: Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Step 9: Install and start Ollama (for LLM)
curl https://ollama.ai/install.sh | sh
ollama serve &

# Step 10: Pull default LLM model (in new terminal)
ollama pull phi4-mini:3.8b

# Step 11: Run the web interface
streamlit run app.py

# Access at: http://localhost:8501
```

---

## 💻 OPTION 3: Quick Lightweight Setup (Testing)

### **Best for:** Quick testing without full infrastructure

```bash
# Step 1: Clone and setup Python
git clone https://github.com/your-username/research-copilot.git
cd research-copilot
python3 -m venv .venv
source .venv/bin/activate

# Step 2: Install core dependencies only
pip install fastapi uvicorn streamlit pydantic requests

# Step 3: Install Ollama
curl https://ollama.ai/install.sh | sh

# Step 4: Start Ollama server (background)
ollama serve &

# Step 5: Pull model
ollama pull phi4-mini:3.8b

# Step 6: Run web interface
streamlit run app.py

# Access: http://localhost:8501
```

---

## 🐳 DOCKER COMPOSE - DETAILED BREAKDOWN

### **What Gets Started:**

```
┌─────────────────────────────────────────────────────┐
│           Docker Compose Services                    │
├─────────────────────────────────────────────────────┤
│ 🗄️  PostgreSQL (Port 5432)                          │
│ 🎛️  Redis (Port 6379)                              │
│ 🤖 Ollama/LLM (Port 11434)                          │
│ 🔍 RAG Service (Port 8001)                          │
│ 🧠 LLM Service (Port 8002)                          │
│ 🔗 Embedding Service (Port 8003)                    │
│ 💾 Storage Service (Port 8004)                      │
│ 📡 API Gateway (Port 8000)                          │
│ 🌐 Web UI - Streamlit (Port 8501)                   │
│ 📊 Prometheus (Port 9090)                           │
│ 📈 Grafana (Port 3000)                              │
│ 📝 Loki (Port 3100)                                 │
└─────────────────────────────────────────────────────┘
```

### **Service Ports Reference:**

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| **Web UI** | 8501 | http://localhost:8501 | Streamlit interface |
| **API Gateway** | 8000 | http://localhost:8000/docs | REST API + docs |
| **RAG Service** | 8001 | Internal only | Document retrieval |
| **LLM Service** | 8002 | Internal only | Language model |
| **Embedding Service** | 8003 | Internal only | Text embeddings |
| **Storage Service** | 8004 | Internal only | Database operations |
| **PostgreSQL** | 5432 | localhost:5432 | Primary database |
| **Redis** | 6379 | localhost:6379 | Caching |
| **Ollama** | 11434 | Internal only | LLM server |
| **Prometheus** | 9090 | http://localhost:9090 | Metrics |
| **Grafana** | 3000 | http://localhost:3000 | Dashboards |

---

## 🌐 ACCESSING THE APPLICATION

### **From the Same Machine:**
```bash
# Web Interface
http://localhost:8501

# API Documentation
http://localhost:8000/docs

# Monitoring Dashboard
http://localhost:3000  # Grafana (admin/admin)
```

### **From Another Machine on Network:**
```bash
# Replace 'localhost' with your Ubuntu machine's IP
# Get IP:
hostname -I

# Then access from other machine:
http://<your-ubuntu-ip>:8501
http://<your-ubuntu-ip>:8000/docs
```

### **From Internet (requires setup):**
Use ngrok or port forwarding:
```bash
# Option 1: Using ngrok (easy)
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update
sudo apt install ngrok
ngrok http 8501

# This gives you a public URL like: https://xxxx-xxx-xxx.ngrok.io
```

---

## 🧪 TESTING THE SETUP

### **Test 1: Check all services are running**
```bash
docker-compose ps
# Should show all services as "Up"
```

### **Test 2: Test the API**
```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy"}
```

### **Test 3: Test Database Connection**
```bash
docker-compose exec postgres psql -U research_user -d research_copilot -c "SELECT version();"
```

### **Test 4: Test Ollama/LLM**
```bash
curl http://localhost:11434/api/tags
# Should return list of available models
```

### **Test 5: Access Web Interface**
```bash
# Open in browser
http://localhost:8501

# Try a search query in the interface
```

---

## 📊 MONITORING & LOGS

### **View Real-time Logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f web  # Streamlit

# Ollama LLM logs
docker-compose logs -f ollama
```

### **Access Monitoring Dashboards:**

**Grafana (Dashboards):**
```
URL: http://localhost:3000
Default Login: admin / admin
Dashboards: Pre-configured for API metrics, DB, and resource usage
```

**Prometheus (Metrics):**
```
URL: http://localhost:9090
Query: Search metrics in PromQL
Examples:
  - http_requests_total
  - http_request_duration_seconds
  - container_memory_usage_bytes
```

---

## ⚙️ CONFIGURATION

### **Environment Variables (.env file):**

Edit `.env` for customization:

```bash
# General
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database
DB_USER=research_user
DB_PASSWORD=research_password
DB_HOST=postgres
DB_PORT=5432
DB_NAME=research_copilot

# Redis
REDIS_URL=redis://:research_redis_pass@redis:6379

# Ollama LLM
OLLAMA_MODEL=phi4-mini:3.8b
OLLAMA_HOST=http://ollama:11434

# CORS (who can access from browser)
CORS_ORIGINS=["http://localhost:8501","http://localhost:3000"]

# Rate limiting
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_PERIOD=60  # seconds
```

---

## 🚨 TROUBLESHOOTING

### **Problem: "Connection refused"**
```bash
# Check if services are running
docker-compose ps

# If not, restart
docker-compose restart

# Check logs
docker-compose logs
```

### **Problem: Port already in use**
```bash
# Find what's using the port
sudo lsof -i :8501

# Change port in docker-compose.yml
# Or kill the process:
sudo kill -9 <PID>
```

### **Problem: Out of memory**
```bash
# Check memory usage
docker stats

# Limit container memory in docker-compose.yml:
services:
  api:
    mem_limit: 512m
```

### **Problem: Ollama model not working**
```bash
# List available models
ollama list

# Pull default model
ollama pull phi4-mini:3.8b

# Or use another model
ollama pull llama2
```

### **Problem: Database connection issues**
```bash
# Check PostgreSQL is running
docker-compose exec postgres pg_isready

# Recreate database
docker-compose down -v
docker-compose up postgres

# Initialize schema
docker-compose exec postgres psql -U research_user -d research_copilot -f init.sql
```

---

## 🔄 STOPPING & RESTARTING

### **Stop All Services:**
```bash
docker-compose down
```

### **Restart Services:**
```bash
docker-compose restart
```

### **Full Reset (WARNING - deletes data!):**
```bash
docker-compose down -v
docker-compose up -d
```

### **View Service Status:**
```bash
docker-compose ps
```

---

## 📈 PERFORMANCE TIPS

### **For Better Performance:**

1. **Increase allocated resources:**
   ```bash
   # Edit docker-compose.yml
   services:
     api:
       deploy:
         resources:
           limits:
             cpus: '2'
             memory: 2G
   ```

2. **Use faster LLM model:**
   ```bash
   # In .env
   OLLAMA_MODEL=neural-chat:7b  # Faster than phi4-mini
   ```

3. **Enable caching:**
   ```bash
   # Already enabled by default with Redis
   REDIS_MAX_CONNECTIONS=100
   ```

4. **Database indexing:**
   ```bash
   docker-compose exec postgres psql -U research_user -d research_copilot << EOF
   CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title);
   CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published_date);
   EOF
   ```

---

## 🔐 SECURITY CONSIDERATIONS

### **For Development (Current Setup):**
- ✅ Suitable for internal networks
- ✅ Good for testing and demos

### **For Production (Before Deploying):**

1. **Change default passwords:**
   ```bash
   # In .env
   DB_PASSWORD=<strong-password>
   REDIS_PASSWORD=<strong-password>
   JWT_SECRET=<strong-secret>
   ```

2. **Set up SSL/HTTPS:**
   ```bash
   # Use Let's Encrypt with nginx
   # See DEPLOYMENT.md for details
   ```

3. **Enable firewall:**
   ```bash
   sudo ufw allow 8501/tcp  # Web UI only
   sudo ufw allow 8000/tcp  # API (optional)
   ```

4. **Set up authentication:**
   - Configure JWT tokens
   - Set rate limiting
   - Enable input validation

---

## 📱 SHARING WITH OTHERS

### **Option 1: On Local Network**
```bash
# Get your IP
hostname -I

# Share with others on same network:
# http://<your-ip>:8501
```

### **Option 2: Using ngrok (Free)**
```bash
# Install ngrok
ngrok http 8501

# Share the public URL it generates
# Lasts 2 hours on free tier
```

### **Option 3: Using Cloudflare Tunnel**
```bash
# More stable than ngrok
# Instructions: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/
```

### **Option 4: Deploy to Cloud**
See `DEPLOYMENT.md` for AWS, DigitalOcean, Heroku options

---

## 📚 NEXT STEPS

### **Once Running:**

1. **Explore the Web Interface:**
   - Search for research papers
   - Ask questions about papers
   - View answers with citations

2. **Test the API:**
   - Visit http://localhost:8000/docs
   - Try API endpoints
   - Generate API tokens

3. **Collect Papers:**
   - Search from ArXiv
   - Import from Google Scholar
   - Build your knowledge base

4. **Monitor Performance:**
   - Check Grafana dashboards
   - View Prometheus metrics
   - Analyze response times

---

## 💬 GETTING HELP

### **If Something Goes Wrong:**

1. **Check logs:**
   ```bash
   docker-compose logs -f
   ```

2. **Review this guide's troubleshooting section**

3. **Check GitHub issues:** https://github.com/your-username/research-copilot/issues

4. **Common issues:**
   - Port conflicts → Change ports in docker-compose.yml
   - Memory issues → Reduce container limits
   - Ollama issues → Check Ollama is running (port 11434)
   - Database issues → Recreate with `docker-compose down -v && docker-compose up`

---

## 🎯 QUICK COMMANDS REFERENCE

```bash
# Start everything
docker-compose up -d

# See status
docker-compose ps

# View logs
docker-compose logs -f

# Stop everything
docker-compose down

# Full reset
docker-compose down -v && docker-compose up -d

# Access web UI
# http://localhost:8501

# Access API docs
# http://localhost:8000/docs

# Access Grafana
# http://localhost:3000
```

---

**Happy Exploring! 🚀**

The application is now ready for your GitHub visitors to test live!
