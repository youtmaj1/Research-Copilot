================================================================================
  RESEARCH COPILOT - OLD LAPTOP TROUBLESHOOTING GUIDE
================================================================================

SCENARIO 1: Services Won't Start
================================================================================

Error: "Cannot connect to Docker daemon"
Fix:
  sudo systemctl start docker
  sudo systemctl enable docker
  sudo usermod -aG docker $USER
  newgrp docker
  docker ps

Error: "Port already in use"
Fix:
  # Find what's using the port
  sudo lsof -i :8501
  
  # Kill process
  sudo kill -9 <PID>
  
  # Or change ports in docker-compose.yml


SCENARIO 2: Embedding Service Crashes
================================================================================

Error: "embedding-service exited with code 1"
Fix Option A (Skip embedding - use lightweight):
  docker-compose -f docker-compose.lightweight.yml down
  docker-compose -f docker-compose.lightweight.yml up -d

Fix Option B (Check logs):
  docker-compose logs embedding-service

Fix Option C (Restart single service):
  docker-compose restart embedding-service
  docker-compose logs embedding-service -f

Fix Option D (Memory too low):
  # If old laptop has <2GB RAM, use lightweight version
  bash start-lightweight.sh


SCENARIO 3: Out of Memory
================================================================================

Error: "Cannot allocate memory" or services crashing randomly
Fix:
  # Use lightweight config (half the memory)
  docker-compose -f docker-compose.lightweight.yml down
  docker-compose -f docker-compose.lightweight.yml up -d

  # Or disable heavy services
  # Edit docker-compose.yml and comment out optional services


SCENARIO 4: Ollama Model Download Issues
================================================================================

Error: "Failed to download model" or "ollama: not running"
Fix:
  # Ollama needs 2-4GB. If laptop is old:
  docker-compose logs ollama
  
  # Reduce model size or use lighter model
  # Edit docker-compose.yml and change model
  
  # Or skip ollama if not needed


SCENARIO 5: Web UI Won't Load (8501)
================================================================================

Problem: http://localhost:8501 shows error or blank page
Fix:
  # Check if streamlit is running
  docker-compose ps | grep streamlit
  
  # View logs
  docker-compose logs streamlit
  
  # Restart streamlit
  docker-compose restart streamlit
  
  # Or rebuild
  docker-compose build streamlit
  docker-compose up -d streamlit


SCENARIO 6: API Won't Respond (8000)
================================================================================

Problem: http://localhost:8000/docs shows error
Fix:
  # Check API gateway
  docker-compose logs api-gateway
  
  # Restart
  docker-compose restart api-gateway
  
  # Rebuild if needed
  docker-compose build --no-cache api-gateway
  docker-compose up -d api-gateway


SCENARIO 7: Database Connection Issues
================================================================================

Error: "Cannot connect to postgres"
Fix:
  # Check postgres is healthy
  docker-compose ps | grep postgres
  
  # View postgres logs
  docker-compose logs postgres
  
  # Restart postgres
  docker-compose down postgres
  docker-compose up -d postgres
  docker-compose ps


SCENARIO 8: Docker Compose Command Not Working
================================================================================

Error: "docker-compose: command not found" or "bash: /usr/bin/docker-compose: No such file"
Fix:
  # Install latest docker-compose
  sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
  
  # Create symlink
  sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
  
  # Verify
  docker-compose --version


================================================================================
QUICK START - OLD LAPTOP (RECOMMENDED)
================================================================================

1. SSH to Ubuntu:
   ssh ubuntu@your-ip

2. Clone and enter directory:
   cd ~/Research-Copilot

3. Make startup script executable:
   chmod +x start-lightweight.sh

4. Start lightweight version:
   bash start-lightweight.sh

5. Wait 60 seconds for all services to start

6. Open in browser:
   http://localhost:8501

7. If issues, check logs:
   docker-compose logs -f


================================================================================
SYSTEM REQUIREMENTS
================================================================================

Minimum for Old Laptop:
  - RAM: 2GB (better with 4GB+)
  - CPU: 2 cores
  - Disk: 20GB free
  - Ubuntu: 18.04 or newer
  - Docker: Latest version

If you have less:
  - Comment out heavy services in docker-compose.yml
  - Use lightweight version (docker-compose.lightweight.yml)
  - Run one service at a time


================================================================================
USEFUL COMMANDS
================================================================================

Start services:
  docker-compose up -d

Stop services:
  docker-compose down

View status:
  docker-compose ps

View logs (all):
  docker-compose logs -f

View logs (single service):
  docker-compose logs -f embedding-service

Restart all:
  docker-compose restart

Restart single service:
  docker-compose restart api-gateway

Remove all data and restart:
  docker-compose down -v
  docker-compose up -d

Check system resources:
  docker stats

Clean up space:
  docker system prune -a -f


================================================================================
PERFORMANCE OPTIMIZATION
================================================================================

For very old laptops:

1. Use lightweight version:
   docker-compose -f docker-compose.lightweight.yml up -d

2. Disable monitoring (remove prometheus, grafana, loki):
   Edit docker-compose.yml and comment out

3. Use CPU-only services:
   Already done - no GPU required

4. Limit memory per service:
   Already configured in lightweight version

5. Reduce model size:
   Use phi2 instead of phi4 in Ollama

6. Disable unused services:
   Comment out in docker-compose.yml


================================================================================
If Still Stuck
================================================================================

1. Check all logs:
   docker-compose logs -f | head -100

2. Share the error in:
   ~/Research-Copilot/error.log

3. Or paste output of:
   docker-compose ps
   docker-compose logs embedding-service --tail 50

4. Describe your laptop:
   - RAM: ___GB
   - CPU: ___
   - Disk: ___GB free
   - Ubuntu version: ___

================================================================================
