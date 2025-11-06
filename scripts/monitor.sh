#!/bin/bash

# Comprehensive System Monitoring and Health Check Script
# Monitors all services, databases, and system metrics

set -e

# Configuration
SERVICES=("api-gateway" "storage-service" "rag-service" "llm-service" "embedding-service")
DATABASES=("postgres" "redis" "weaviate")
API_BASE_URL="http://localhost:8000"
ALERT_WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"  # Set this to your alerting system

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Send alert to webhook if configured
send_alert() {
    local level="$1"
    local message="$2"
    
    if [ -n "$ALERT_WEBHOOK_URL" ]; then
        curl -X POST "$ALERT_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"level\":\"$level\",\"message\":\"$message\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
            --silent --fail || true
    fi
}

# Check Docker service status
check_service_status() {
    local service="$1"
    local status=$(docker-compose ps -q "$service" 2>/dev/null)
    
    if [ -z "$status" ]; then
        log_error "Service $service is not running"
        send_alert "error" "Service $service is down"
        return 1
    fi
    
    local health=$(docker inspect --format='{{.State.Health.Status}}' "$status" 2>/dev/null || echo "none")
    
    case "$health" in
        "healthy")
            log_success "Service $service is healthy"
            return 0
            ;;
        "unhealthy")
            log_error "Service $service is unhealthy"
            send_alert "error" "Service $service is unhealthy"
            return 1
            ;;
        "starting")
            log_warning "Service $service is starting"
            return 2
            ;;
        *)
            log_warning "Service $service health status unknown"
            return 2
            ;;
    esac
}

# Check API endpoint health
check_api_health() {
    local endpoint="$1"
    local service_name="$2"
    
    local response=$(curl -s -w "%{http_code}" -o /tmp/health_response "$endpoint" || echo "000")
    
    if [ "$response" = "200" ]; then
        local status=$(cat /tmp/health_response | jq -r '.status' 2>/dev/null || echo "unknown")
        if [ "$status" = "healthy" ]; then
            log_success "API health check passed: $service_name"
            return 0
        else
            log_warning "API reports unhealthy status: $service_name ($status)"
            return 1
        fi
    else
        log_error "API health check failed: $service_name (HTTP $response)"
        send_alert "error" "API health check failed for $service_name"
        return 1
    fi
}

# Check database connectivity
check_database() {
    local db_type="$1"
    
    case "$db_type" in
        "postgres")
            docker-compose exec -T postgres pg_isready -U research_user -d research_copilot > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                log_success "PostgreSQL is accessible"
                return 0
            else
                log_error "PostgreSQL is not accessible"
                send_alert "error" "PostgreSQL database is not accessible"
                return 1
            fi
            ;;
        "redis")
            docker-compose exec -T redis redis-cli ping > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                log_success "Redis is accessible"
                return 0
            else
                log_error "Redis is not accessible"
                send_alert "error" "Redis cache is not accessible"
                return 1
            fi
            ;;
        "weaviate")
            curl -s -f "http://localhost:8080/v1/meta" > /dev/null
            if [ $? -eq 0 ]; then
                log_success "Weaviate is accessible"
                return 0
            else
                log_error "Weaviate is not accessible"
                send_alert "error" "Weaviate vector database is not accessible"
                return 1
            fi
            ;;
    esac
}

# Check system resources
check_system_resources() {
    log_info "Checking system resources..."
    
    # Memory usage
    local mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    if (( $(echo "$mem_usage > 80.0" | bc -l) )); then
        log_warning "High memory usage: ${mem_usage}%"
        send_alert "warning" "High memory usage: ${mem_usage}%"
    else
        log_success "Memory usage: ${mem_usage}%"
    fi
    
    # Disk usage
    local disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 80 ]; then
        log_warning "High disk usage: ${disk_usage}%"
        send_alert "warning" "High disk usage: ${disk_usage}%"
    else
        log_success "Disk usage: ${disk_usage}%"
    fi
    
    # CPU load
    local cpu_load=$(uptime | awk -F'load average:' '{ print $2 }' | cut -d, -f1 | xargs)
    local cpu_cores=$(nproc)
    local load_threshold=$(echo "$cpu_cores * 0.8" | bc)
    
    if (( $(echo "$cpu_load > $load_threshold" | bc -l) )); then
        log_warning "High CPU load: $cpu_load (threshold: $load_threshold)"
        send_alert "warning" "High CPU load: $cpu_load"
    else
        log_success "CPU load: $cpu_load"
    fi
}

# Check Docker resources
check_docker_resources() {
    log_info "Checking Docker resources..."
    
    # Docker system info
    local docker_info=$(docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}\t{{.Reclaimable}}" | tail -n +2)
    
    echo "$docker_info" | while read line; do
        local size=$(echo "$line" | awk '{print $3}')
        local reclaimable=$(echo "$line" | awk '{print $4}')
        log_info "Docker $line"
    done
    
    # Container resource usage
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | \
    while read line; do
        if [[ "$line" != *"CONTAINER"* ]]; then
            local container=$(echo "$line" | awk '{print $1}')
            local cpu=$(echo "$line" | awk '{print $2}' | sed 's/%//')
            local mem_perc=$(echo "$line" | awk '{print $4}' | sed 's/%//')
            
            if (( $(echo "$cpu > 80.0" | bc -l) )); then
                log_warning "High CPU usage in $container: ${cpu}%"
            fi
            
            if (( $(echo "$mem_perc > 80.0" | bc -l) )); then
                log_warning "High memory usage in $container: ${mem_perc}%"
            fi
        fi
    done
}

# Performance test
run_performance_test() {
    log_info "Running quick performance test..."
    
    local start_time=$(date +%s%N)
    local response=$(curl -s -w "%{http_code},%{time_total}" \
        -H "Content-Type: application/json" \
        -d '{"question":"What is machine learning?","max_results":5}' \
        "${API_BASE_URL}/api/v1/query" || echo "000,999")
    
    local http_code=$(echo "$response" | cut -d',' -f1)
    local response_time=$(echo "$response" | cut -d',' -f2)
    
    if [ "$http_code" = "200" ]; then
        if (( $(echo "$response_time > 5.0" | bc -l) )); then
            log_warning "Slow API response: ${response_time}s"
            send_alert "warning" "API response time is slow: ${response_time}s"
        else
            log_success "API performance test passed: ${response_time}s"
        fi
    else
        log_error "API performance test failed (HTTP $http_code)"
        send_alert "error" "API performance test failed"
    fi
}

# Generate health report
generate_health_report() {
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local report_file="/tmp/health_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "timestamp": "$timestamp",
  "overall_status": "$1",
  "services": {
EOF

    local first=true
    for service in "${SERVICES[@]}"; do
        if [ "$first" = false ]; then
            echo "," >> "$report_file"
        fi
        first=false
        
        local status="unknown"
        check_service_status "$service" >/dev/null && status="healthy" || status="unhealthy"
        
        cat >> "$report_file" << EOF
    "$service": {
      "status": "$status",
      "last_check": "$timestamp"
    }
EOF
    done
    
    cat >> "$report_file" << EOF
  },
  "databases": {
EOF

    first=true
    for db in "${DATABASES[@]}"; do
        if [ "$first" = false ]; then
            echo "," >> "$report_file"
        fi
        first=false
        
        local status="unknown"
        check_database "$db" >/dev/null && status="healthy" || status="unhealthy"
        
        cat >> "$report_file" << EOF
    "$db": {
      "status": "$status",
      "last_check": "$timestamp"
    }
EOF
    done
    
    cat >> "$report_file" << EOF
  },
  "system": {
    "memory_usage": "$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')%",
    "disk_usage": "$(df / | tail -1 | awk '{print $5}')%",
    "cpu_load": "$(uptime | awk -F'load average:' '{ print $2 }' | cut -d, -f1 | xargs)"
  }
}
EOF
    
    log_info "Health report generated: $report_file"
    
    # Send to monitoring system if configured
    if [ -n "$MONITORING_WEBHOOK_URL" ]; then
        curl -X POST "$MONITORING_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d @"$report_file" \
            --silent --fail || true
    fi
}

# Main monitoring function
main() {
    echo "🏥 Research Copilot Health Monitor"
    echo "=================================="
    echo "Timestamp: $(date)"
    echo ""
    
    local overall_status="healthy"
    local failed_checks=0
    
    # Check all services
    log_info "Checking microservices..."
    for service in "${SERVICES[@]}"; do
        if ! check_service_status "$service"; then
            overall_status="unhealthy"
            ((failed_checks++))
        fi
    done
    
    # Check API endpoints
    log_info "Checking API endpoints..."
    local api_endpoints=(
        "http://localhost:8000/health:API Gateway"
        "http://localhost:8001/health:RAG Service"
        "http://localhost:8002/health:LLM Service"
        "http://localhost:8003/health:Embedding Service"
        "http://localhost:8004/health:Storage Service"
    )
    
    for endpoint_info in "${api_endpoints[@]}"; do
        local endpoint=$(echo "$endpoint_info" | cut -d':' -f1)
        local name=$(echo "$endpoint_info" | cut -d':' -f2)
        
        if ! check_api_health "$endpoint" "$name"; then
            overall_status="degraded"
            ((failed_checks++))
        fi
    done
    
    # Check databases
    log_info "Checking databases..."
    for db in "${DATABASES[@]}"; do
        if ! check_database "$db"; then
            overall_status="unhealthy"
            ((failed_checks++))
        fi
    done
    
    # Check system resources
    check_system_resources
    check_docker_resources
    
    # Run performance test
    run_performance_test
    
    # Generate report
    generate_health_report "$overall_status"
    
    # Final status
    echo ""
    echo "===================="
    if [ "$failed_checks" -eq 0 ]; then
        log_success "All systems operational"
    else
        log_warning "$failed_checks checks failed - System status: $overall_status"
    fi
    
    exit $failed_checks
}

# Handle script interruption
trap 'log_error "Monitoring interrupted"; exit 1' INT TERM

# Check if required tools are available
if ! command -v jq &> /dev/null; then
    log_warning "jq not found, JSON parsing may be limited"
fi

if ! command -v bc &> /dev/null; then
    log_warning "bc not found, some calculations may be limited"
fi

# Run main function
main "$@"
