#!/bin/bash

# Production Deployment Script for Research Copilot
# This script sets up the complete production environment

set -e

echo "🚀 Research Copilot Production Deployment"
echo "=========================================="

# Configuration
PROJECT_NAME="research-copilot"
ENVIRONMENT="${1:-production}"
DOMAIN="${2:-research-copilot.local}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    if ! command -v openssl &> /dev/null; then
        log_error "OpenSSL is not installed"
        exit 1
    fi
    
    log_success "Prerequisites check completed"
}

# Generate SSL certificates
generate_ssl_certificates() {
    log_info "Generating SSL certificates..."
    
    mkdir -p nginx/ssl
    
    if [ ! -f "nginx/ssl/cert.pem" ]; then
        openssl req -x509 -newkey rsa:4096 -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem \
            -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=${DOMAIN}"
        log_success "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
}

# Create environment configuration
create_environment_config() {
    log_info "Creating environment configuration..."
    
    cat > .env.${ENVIRONMENT} << EOF
# Database Configuration
POSTGRES_DB=research_copilot
POSTGRES_USER=research_user
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Redis Configuration
REDIS_PASSWORD=$(openssl rand -base64 32)

# JWT Configuration
JWT_SECRET_KEY=$(openssl rand -base64 64)

# API Configuration
API_DOMAIN=${DOMAIN}
ENVIRONMENT=${ENVIRONMENT}

# Monitoring
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 32)

# Vector Database
WEAVIATE_API_KEY=$(openssl rand -base64 32)
EOF
    
    log_success "Environment configuration created"
}

# Initialize database
initialize_database() {
    log_info "Initializing database..."
    
    cat > init.sql << EOF
-- Create database extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create full-text search configuration
CREATE TEXT SEARCH CONFIGURATION IF NOT EXISTS english_research (COPY = english);

-- Performance optimizations
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';

-- Security settings
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_duration = on;
ALTER SYSTEM SET log_connections = on;
EOF
    
    log_success "Database initialization script created"
}

# Setup monitoring configuration
setup_monitoring() {
    log_info "Setting up monitoring configuration..."
    
    # Create Grafana dashboards directory
    mkdir -p monitoring/grafana/provisioning/dashboards
    mkdir -p monitoring/grafana/provisioning/datasources
    
    # Grafana datasources configuration
    cat > monitoring/grafana/provisioning/datasources/datasources.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
EOF

    # Grafana dashboards configuration
    cat > monitoring/grafana/provisioning/dashboards/dashboards.yml << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF
    
    log_success "Monitoring configuration created"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    # Stop existing services
    docker-compose down --remove-orphans
    
    # Pull latest images
    docker-compose pull
    
    # Build custom images
    docker-compose build --no-cache
    
    # Start services
    docker-compose --env-file .env.${ENVIRONMENT} up -d
    
    log_success "Services deployed"
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    services=("postgres" "redis" "api-gateway" "storage-service" "rag-service" "llm-service" "embedding-service")
    
    for service in "${services[@]}"; do
        log_info "Waiting for ${service}..."
        timeout=300  # 5 minutes
        elapsed=0
        
        while [ $elapsed -lt $timeout ]; do
            if docker-compose ps ${service} | grep -q "Up"; then
                log_success "${service} is ready"
                break
            fi
            sleep 5
            elapsed=$((elapsed + 5))
        done
        
        if [ $elapsed -ge $timeout ]; then
            log_error "${service} failed to start within ${timeout} seconds"
            exit 1
        fi
    done
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    endpoints=(
        "http://localhost:8000/health"
        "http://localhost:8001/health"
        "http://localhost:8002/health"
        "http://localhost:8003/health"
        "http://localhost:8004/health"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f -s "$endpoint" > /dev/null; then
            log_success "Health check passed: $endpoint"
        else
            log_error "Health check failed: $endpoint"
        fi
    done
}

# Setup log rotation
setup_log_rotation() {
    log_info "Setting up log rotation..."
    
    cat > /etc/logrotate.d/research-copilot << EOF
/var/log/research-copilot/*.log {
    daily
    missingok
    rotate 52
    compress
    notifempty
    create 644 root root
    postrotate
        docker-compose restart promtail
    endscript
}
EOF
    
    log_success "Log rotation configured"
}

# Display deployment information
display_deployment_info() {
    log_success "Deployment completed successfully!"
    echo ""
    echo "🌐 Service URLs:"
    echo "   API: https://${DOMAIN}"
    echo "   Monitoring: https://monitoring.${DOMAIN}"
    echo "   Grafana: http://localhost:3000 (admin/admin)"
    echo "   Prometheus: http://localhost:9090"
    echo ""
    echo "📊 Default credentials:"
    echo "   Grafana: admin/admin (change on first login)"
    echo ""
    echo "🔧 Management commands:"
    echo "   View logs: docker-compose logs -f [service]"
    echo "   Scale service: docker-compose up -d --scale [service]=[count]"
    echo "   Update services: docker-compose pull && docker-compose up -d"
    echo "   Backup database: ./scripts/backup.sh"
    echo ""
    echo "📁 Configuration files:"
    echo "   Environment: .env.${ENVIRONMENT}"
    echo "   SSL Certificates: nginx/ssl/"
    echo "   Monitoring: monitoring/"
}

# Main deployment process
main() {
    log_info "Starting deployment for environment: ${ENVIRONMENT}"
    
    check_prerequisites
    generate_ssl_certificates
    create_environment_config
    initialize_database
    setup_monitoring
    deploy_services
    wait_for_services
    run_health_checks
    setup_log_rotation
    display_deployment_info
}

# Handle script interruption
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"
