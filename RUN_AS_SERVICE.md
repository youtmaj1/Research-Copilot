# 🚀 Run as Service & Share with Others

**Goal:** Run Research Copilot on your Ubuntu laptop so others can access it remotely

---

## 📋 Option 1: Local Network Access (Easiest)

### Step 1: Get Your Ubuntu IP Address
```bash
hostname -I
# You'll see something like: 192.168.1.100
```

### Step 2: Run Docker Compose
```bash
cd research-copilot
docker-compose up -d
```

### Step 3: Share Your IP
Tell others to use: **http://your-ubuntu-ip:8501**

Example: `http://192.168.1.100:8501`

### Access Points for Others:
| Service | URL |
|---------|-----|
| Web UI | http://192.168.1.100:8501 |
| API Docs | http://192.168.1.100:8000/docs |
| Monitoring | http://192.168.1.100:3000 |

---

## 🌐 Option 2: Public Internet Access (Using ngrok)

### Step 1: Install ngrok
```bash
# Download from https://ngrok.com
# Or use apt:
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update
sudo apt install ngrok
```

### Step 2: Sign up ngrok (Free)
- Go to https://ngrok.com
- Create account
- Get your auth token

### Step 3: Setup ngrok
```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

### Step 4: Run Docker Compose
```bash
cd research-copilot
docker-compose up -d
```

### Step 5: Share via ngrok
In a new terminal:
```bash
ngrok http 8501
```

You'll see:
```
Forwarding    https://xxxx-xxx-xxx.ngrok.io -> http://localhost:8501
```

### Step 6: Share the URL
Send others: **https://xxxx-xxx-xxx.ngrok.io**

They can access from anywhere!

---

## 🔧 Option 3: Run as Linux Systemd Service (Always Running)

### Step 1: Create systemd service file
```bash
sudo nano /etc/systemd/system/research-copilot.service
```

### Step 2: Paste this content
```ini
[Unit]
Description=Research Copilot Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/research-copilot
ExecStart=/usr/local/bin/docker-compose up
ExecStop=/usr/local/bin/docker-compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Replace `YOUR_USERNAME` with your Ubuntu username!

### Step 3: Enable and start service
```bash
sudo systemctl daemon-reload
sudo systemctl enable research-copilot
sudo systemctl start research-copilot
```

### Step 4: Check if running
```bash
sudo systemctl status research-copilot
```

### Step 5: View logs
```bash
sudo systemctl logs research-copilot -f
```

### Step 6: Stop/Restart
```bash
sudo systemctl stop research-copilot
sudo systemctl restart research-copilot
```

---

## 📱 Option 4: Cloud Tunnel (Cloudflare Tunnel)

### Step 1: Install Cloudflare CLI
```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

### Step 2: Authenticate
```bash
cloudflared tunnel login
```

### Step 3: Create tunnel
```bash
cloudflared tunnel create research-copilot
```

### Step 4: Configure tunnel
Create `~/.cloudflared/config.yml`:
```yaml
tunnel: research-copilot
credentials-file: /home/YOUR_USERNAME/.cloudflared/research-copilot.json

ingress:
  - hostname: research-copilot.example.com
    service: http://localhost:8501
  - service: http_status:404
```

### Step 5: Run tunnel
```bash
cloudflared tunnel run research-copilot
```

### Step 6: Share URL
Others access: **https://research-copilot.example.com**

---

## 🔐 IMPORTANT: Security for Public Access

If exposing to internet, add authentication:

### Add Password to Streamlit (in docker-compose.yml)
```yaml
services:
  web:
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_SERVER_RUN_ON_SAVE=false
```

### Or use Nginx with Basic Auth
Create `nginx/auth` file:
```bash
sudo apt install apache2-utils
htpasswd -c nginx/auth username
# Enter password when prompted
```

Update `docker-compose.yml` to enable Nginx auth.

---

## 🛑 Keep Service Running

### Auto-restart on reboot
Already configured in systemd service above ✓

### Monitor service health
```bash
# Check if all containers running
docker ps

# View logs
docker-compose logs -f

# Get service status
sudo systemctl status research-copilot
```

### Restart if something breaks
```bash
sudo systemctl restart research-copilot
```

---

## 📊 Share Instructions with Others

**Tell them this:**

```
🚀 Research Copilot is running!

Access URL: http://192.168.1.100:8501

Or public access: https://xxxx-xxx-xxx.ngrok.io

Available services:
- Web UI: Paper search & Q&A
- API Docs: http://192.168.1.100:8000/docs
- Monitoring: http://192.168.1.100:3000

Features you can try:
1. Search for research papers
2. Ask questions about papers
3. View system metrics
```

---

## ✅ Quick Commands Reference

```bash
# Start service
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop service
docker-compose down

# Restart service
docker-compose restart

# Get your IP
hostname -I

# Share with ngrok
ngrok http 8501

# Run as system service
sudo systemctl start research-copilot
sudo systemctl status research-copilot
sudo systemctl logs research-copilot -f
```

---

## 🎯 Recommended Setup

**For friends on same WiFi:**
1. Use Option 1 (Local Network)
2. Share your IP: `hostname -I`
3. They access: `http://your-ip:8501`

**For public internet:**
1. Use Option 2 (ngrok) - easiest
2. Run: `ngrok http 8501`
3. Share the URL

**For always-on service:**
1. Use Option 3 (Systemd)
2. Runs automatically on reboot
3. Others always have access

---

**Now others can test your project live!** 🎉
