#!/bin/bash

# Research Copilot - Old Laptop Startup Script
# Usage: bash start-lightweight.sh

set -e

echo "🚀 Starting Research Copilot (Lightweight Mode for Old Laptops)"
echo "================================================================"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Install: sudo apt install docker.io -y"
    exit 1
fi

# Check docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Installing latest version..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
fi

echo "✅ Docker and docker-compose OK"

# Check if already running
if docker-compose ps 2>/dev/null | grep -q "Up"; then
    echo "⚠️  Services already running!"
    echo "Stop with: docker-compose down"
    echo "View status: docker-compose ps"
    exit 0
fi

echo ""
echo "Starting services..."
echo "- PostgreSQL (5432)"
echo "- Redis (6379)"
echo "- Ollama (11434)"
echo "- LLM Service (8002)"
echo "- Embedding Service (8003)"
echo "- RAG Service (8001)"
echo "- Storage Service (8004)"
echo "- API Gateway (8000)"
echo "- Streamlit Web UI (8501)"
echo ""

# Start with lightweight config
docker-compose -f docker-compose.lightweight.yml up -d

echo ""
echo "⏳ Waiting for services to be healthy (60 seconds)..."
sleep 60

echo ""
echo "✅ Services started!"
echo ""
echo "Access points:"
echo "  🌐 Web UI:     http://localhost:8501"
echo "  📚 API Docs:   http://localhost:8000/docs"
echo "  📊 Prometheus: http://localhost:9090"
echo ""
echo "Check status:"
echo "  docker-compose ps"
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "Stop services:"
echo "  docker-compose down"
echo ""
echo "🎉 All systems ready!"
