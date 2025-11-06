"""
Production Configuration Management System
=========================================

Comprehensive configuration management for enterprise deployment
of the Research Copilot system with environment-specific settings,
secrets management, and deployment automation.

Features:
- Environment-based configuration (dev, staging, prod)
- Secure secrets management
- Configuration validation and schema
- Hot configuration reloading
- Docker and Kubernetes deployment configs
- Performance tuning presets

Author: Research Copilot System
Version: 2.0 Enterprise
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import subprocess
from enum import Enum
import secrets
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    path: str = "papers.db"
    max_connections: int = 10
    connection_timeout: int = 30
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    maintenance_window: str = "02:00-04:00"

@dataclass
class CacheConfig:
    """Cache configuration"""
    memory_limit_mb: int = 500
    ttl_seconds: int = 3600
    cleanup_interval_seconds: int = 300
    compression_enabled: bool = True
    persistence_enabled: bool = False
    redis_url: Optional[str] = None

@dataclass
class APIConfig:
    """API configuration"""
    arxiv_rate_limit: int = 3  # requests per second
    arxiv_retry_attempts: int = 3
    arxiv_timeout_seconds: int = 30
    
    # LLM Configuration - Default to Ollama
    llm_provider: str = "ollama"  # ollama, openai, anthropic
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi4-mini:3.8b"
    ollama_temperature: float = 0.3
    ollama_max_tokens: int = 2000
    ollama_timeout_seconds: int = 60
    
    # OpenAI Configuration (fallback)
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.3
    openai_max_tokens: int = 2000
    openai_timeout_seconds: int = 60

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    metrics_interval_seconds: int = 30
    health_check_interval_seconds: int = 60
    log_level: str = "INFO"
    log_format: str = "structured"
    prometheus_enabled: bool = True
    prometheus_port: int = 8090
    alertmanager_webhook: Optional[str] = None

@dataclass
class SecurityConfig:
    """Security configuration"""
    encryption_enabled: bool = True
    api_key_rotation_days: int = 90
    session_timeout_minutes: int = 60
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limiting_enabled: bool = True
    max_requests_per_minute: int = 100

@dataclass
class PerformanceConfig:
    """Performance tuning configuration"""
    batch_size: int = 100
    batch_timeout_seconds: float = 5.0
    worker_threads: int = 4
    max_concurrent_requests: int = 50
    request_timeout_seconds: int = 300
    memory_optimization_enabled: bool = True

@dataclass
class ResearchCopilotConfig:
    """Main configuration class"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    version: str = "2.0.0"
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Runtime settings
    config_file: Optional[str] = None
    secrets_file: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)

class SecretsManager:
    """Secure secrets management"""
    
    def __init__(self, secrets_file: str = "secrets.encrypted"):
        self.secrets_file = secrets_file
        self.encryption_key = None
        self.secrets = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption key from environment or generate new one"""
        # Try to get key from environment
        key_env = os.getenv('RESEARCH_COPILOT_SECRET_KEY')
        
        if key_env:
            self.encryption_key = key_env.encode()
        else:
            # Generate key from password (for development)
            password = os.getenv('RESEARCH_COPILOT_PASSWORD', 'development-password-change-in-production')
            salt = b'research-copilot-salt'  # In production, use random salt
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            self.encryption_key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        self.fernet = Fernet(self.encryption_key)
    
    def set_secret(self, key: str, value: str):
        """Set a secret value"""
        self.secrets[key] = value
        self._save_secrets()
        self.logger.info(f"Secret updated: {key}")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value"""
        # First check environment variables
        env_value = os.getenv(key.upper())
        if env_value:
            return env_value
        
        # Then check stored secrets
        return self.secrets.get(key, default)
    
    def load_secrets(self):
        """Load secrets from encrypted file"""
        if not os.path.exists(self.secrets_file):
            self.logger.info("No secrets file found, starting with empty secrets")
            return
        
        try:
            with open(self.secrets_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            self.secrets = json.loads(decrypted_data.decode())
            
            self.logger.info(f"Loaded {len(self.secrets)} secrets")
            
        except Exception as e:
            self.logger.error(f"Failed to load secrets: {e}")
            self.secrets = {}
    
    def _save_secrets(self):
        """Save secrets to encrypted file"""
        try:
            data = json.dumps(self.secrets, indent=2)
            encrypted_data = self.fernet.encrypt(data.encode())
            
            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set secure file permissions
            os.chmod(self.secrets_file, 0o600)
            
        except Exception as e:
            self.logger.error(f"Failed to save secrets: {e}")
    
    def generate_api_key(self, name: str) -> str:
        """Generate a new API key"""
        api_key = secrets.token_urlsafe(32)
        self.set_secret(f"api_key_{name}", api_key)
        return api_key
    
    def rotate_secrets(self):
        """Rotate expiring secrets"""
        # This would implement automatic secret rotation
        self.logger.info("Secret rotation not yet implemented")

class ConfigurationManager:
    """Main configuration management system"""
    
    def __init__(self, config_file: Optional[str] = None, environment: Optional[str] = None):
        self.config_file = config_file or "config.yaml"
        self.environment = Environment(environment or os.getenv('ENVIRONMENT', 'development'))
        self.config = ResearchCopilotConfig(environment=self.environment)
        self.secrets_manager = SecretsManager()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_configuration()
        
        # Apply environment-specific settings
        self._apply_environment_settings()
        
        # Validate configuration
        self._validate_configuration()
    
    def _load_configuration(self):
        """Load configuration from file"""
        # Load secrets first
        self.secrets_manager.load_secrets()
        
        # Load main configuration
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                # Update configuration with loaded data
                self._update_config_from_dict(config_data)
                
                self.logger.info(f"Loaded configuration from {self.config_file}")
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration file: {e}")
        else:
            self.logger.info("No configuration file found, using defaults")
        
        # Override with environment variables
        self._load_environment_variables()
    
    def _update_config_from_dict(self, data: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in data.items():
            if hasattr(self.config, section) and isinstance(values, dict):
                section_config = getattr(self.config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'RC_DB_PATH': ('database', 'path'),
            'RC_DB_MAX_CONNECTIONS': ('database', 'max_connections'),
            'RC_CACHE_MEMORY_LIMIT_MB': ('cache', 'memory_limit_mb'),
            'RC_CACHE_TTL_SECONDS': ('cache', 'ttl_seconds'),
            'RC_API_OPENAI_MODEL': ('api', 'openai_model'),
            'RC_API_OPENAI_TEMPERATURE': ('api', 'openai_temperature'),
            'RC_MONITORING_LOG_LEVEL': ('monitoring', 'log_level'),
            'RC_MONITORING_PROMETHEUS_PORT': ('monitoring', 'prometheus_port'),
            'RC_PERFORMANCE_BATCH_SIZE': ('performance', 'batch_size'),
            'RC_PERFORMANCE_WORKER_THREADS': ('performance', 'worker_threads'),
            'RC_SECURITY_RATE_LIMIT_RPM': ('security', 'max_requests_per_minute'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section_config = getattr(self.config, section)
                
                # Type conversion
                current_value = getattr(section_config, key)
                if isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                
                setattr(section_config, key, value)
                self.logger.debug(f"Set {section}.{key} = {value} from environment")
    
    def _apply_environment_settings(self):
        """Apply environment-specific configurations"""
        if self.environment == Environment.DEVELOPMENT:
            self.config.debug = True
            self.config.monitoring.log_level = "DEBUG"
            self.config.security.encryption_enabled = False
            self.config.database.backup_enabled = False
            
        elif self.environment == Environment.STAGING:
            self.config.debug = False
            self.config.monitoring.log_level = "INFO"
            self.config.security.encryption_enabled = True
            self.config.database.backup_enabled = True
            
        elif self.environment == Environment.PRODUCTION:
            self.config.debug = False
            self.config.monitoring.log_level = "WARNING"
            self.config.security.encryption_enabled = True
            self.config.security.cors_origins = []  # Restrict CORS in production
            self.config.database.backup_enabled = True
            self.config.performance.memory_optimization_enabled = True
            
        elif self.environment == Environment.TESTING:
            self.config.debug = True
            self.config.monitoring.log_level = "DEBUG"
            self.config.database.path = ":memory:"  # In-memory database for tests
            self.config.cache.memory_limit_mb = 50  # Smaller cache for tests
    
    def _validate_configuration(self):
        """Validate configuration values"""
        errors = []
        
        # Database validation
        if self.config.database.max_connections < 1:
            errors.append("Database max_connections must be >= 1")
        
        # Cache validation
        if self.config.cache.memory_limit_mb < 10:
            errors.append("Cache memory_limit_mb must be >= 10")
        
        # API validation
        if self.config.api.arxiv_rate_limit < 1:
            errors.append("ArXiv rate_limit must be >= 1")
        
        # Performance validation
        if self.config.performance.worker_threads < 1:
            errors.append("Performance worker_threads must be >= 1")
        
        # Check required secrets
        required_secrets = ['openai_api_key']
        for secret in required_secrets:
            if not self.secrets_manager.get_secret(secret):
                if self.environment == Environment.PRODUCTION:
                    errors.append(f"Required secret missing: {secret}")
                else:
                    self.logger.warning(f"Required secret missing: {secret}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)
        
        self.logger.info("Configuration validation passed")
    
    def get_config(self) -> ResearchCopilotConfig:
        """Get the current configuration"""
        return self.config
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value"""
        return self.secrets_manager.get_secret(key, default)
    
    def set_secret(self, key: str, value: str):
        """Set a secret value"""
        self.secrets_manager.set_secret(key, value)
    
    def export_config(self, format: str = 'yaml') -> str:
        """Export configuration to string"""
        config_dict = asdict(self.config)
        
        # Remove sensitive fields
        config_dict.pop('secrets_file', None)
        config_dict.pop('start_time', None)
        
        if format.lower() == 'yaml':
            return yaml.dump(config_dict, default_flow_style=False)
        else:
            return json.dumps(config_dict, indent=2, default=str)
    
    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file"""
        file_path = file_path or self.config_file
        
        try:
            config_content = self.export_config('yaml' if file_path.endswith('.yaml') else 'json')
            
            with open(file_path, 'w') as f:
                f.write(config_content)
            
            self.logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def reload_config(self):
        """Reload configuration from file"""
        self.logger.info("Reloading configuration...")
        self._load_configuration()
        self._apply_environment_settings()
        self._validate_configuration()
        self.logger.info("Configuration reloaded successfully")

class DeploymentManager:
    """Deployment configuration and automation"""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.logger = logging.getLogger(__name__)
    
    def generate_docker_config(self) -> str:
        """Generate Dockerfile configuration"""
        dockerfile = f"""# Research Copilot {self.config.version}
FROM python:3.11-slim

# Set environment
ENV ENVIRONMENT={self.config.environment.value}
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    sqlite3 \\
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 research && chown -R research:research /app
USER research

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)"

# Start application
CMD ["python", "-m", "main"]
"""
        return dockerfile
    
    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml configuration"""
        compose = f"""version: '3.8'

services:
  research-copilot:
    build: .
    container_name: research-copilot
    environment:
      - ENVIRONMENT={self.config.environment.value}
      - RC_DB_PATH=/data/papers.db
      - RC_MONITORING_PROMETHEUS_PORT={self.config.monitoring.prometheus_port}
    ports:
      - "8080:8080"
      - "{self.config.monitoring.prometheus_port}:{self.config.monitoring.prometheus_port}"
    volumes:
      - ./data:/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8080/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  prometheus:
    image: prom/prometheus:latest
    container_name: research-copilot-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: research-copilot-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped

volumes:
  grafana-storage:
"""
        return compose
    
    def generate_kubernetes_config(self) -> str:
        """Generate Kubernetes deployment configuration"""
        k8s_config = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: research-copilot
  labels:
    app: research-copilot
    version: "{self.config.version}"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: research-copilot
  template:
    metadata:
      labels:
        app: research-copilot
    spec:
      containers:
      - name: research-copilot
        image: research-copilot:{self.config.version}
        ports:
        - containerPort: 8080
        - containerPort: {self.config.monitoring.prometheus_port}
        env:
        - name: ENVIRONMENT
          value: "{self.config.environment.value}"
        - name: RC_DB_PATH
          value: "/data/papers.db"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: data-volume
          mountPath: /data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: research-copilot-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: research-copilot-service
spec:
  selector:
    app: research-copilot
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: {self.config.monitoring.prometheus_port}
    targetPort: {self.config.monitoring.prometheus_port}
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: research-copilot-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
"""
        return k8s_config
    
    def generate_prometheus_config(self) -> str:
        """Generate Prometheus configuration"""
        prometheus_config = f"""global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'research-copilot'
    static_configs:
      - targets: ['localhost:{self.config.monitoring.prometheus_port}']
    scrape_interval: 10s
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
"""
        return prometheus_config
    
    def create_deployment_files(self, output_dir: str = "deployment"):
        """Create all deployment configuration files"""
        os.makedirs(output_dir, exist_ok=True)
        
        files = {
            'Dockerfile': self.generate_docker_config(),
            'docker-compose.yml': self.generate_docker_compose(),
            'k8s-deployment.yaml': self.generate_kubernetes_config(),
            'prometheus.yml': self.generate_prometheus_config()
        }
        
        for filename, content in files.items():
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'w') as f:
                f.write(content)
            self.logger.info(f"Created deployment file: {file_path}")

def main():
    """Main function for testing configuration management"""
    print("⚙️  Production Configuration Management Test")
    print("=" * 50)
    
    # Test different environments
    environments = ['development', 'staging', 'production']
    
    for env in environments:
        print(f"\n🌍 Testing {env.upper()} environment:")
        
        # Initialize configuration manager
        config_manager = ConfigurationManager(environment=env)
        config = config_manager.get_config()
        
        print(f"  Environment: {config.environment.value}")
        print(f"  Debug: {config.debug}")
        print(f"  Log Level: {config.monitoring.log_level}")
        print(f"  Database Path: {config.database.path}")
        print(f"  Cache Memory: {config.cache.memory_limit_mb}MB")
        print(f"  Worker Threads: {config.performance.worker_threads}")
        print(f"  Encryption: {config.security.encryption_enabled}")
    
    # Test secrets management
    print(f"\n🔐 Testing Secrets Management:")
    config_manager = ConfigurationManager()
    
    # Set a test secret
    config_manager.set_secret("test_api_key", "sk-test-12345")
    retrieved = config_manager.get_secret("test_api_key")
    print(f"  Secret storage/retrieval: {'✅ Working' if retrieved == 'sk-test-12345' else '❌ Failed'}")
    
    # Test configuration export
    print(f"\n📄 Configuration Export:")
    yaml_config = config_manager.export_config('yaml')
    json_config = config_manager.export_config('json')
    print(f"  YAML export: {'✅ Success' if yaml_config else '❌ Failed'}")
    print(f"  JSON export: {'✅ Success' if json_config else '❌ Failed'}")
    
    # Test deployment configuration
    print(f"\n🚀 Deployment Configuration:")
    deployment_manager = DeploymentManager(config_manager)
    
    try:
        deployment_manager.create_deployment_files("test_deployment")
        print("  ✅ Deployment files created successfully")
    except Exception as e:
        print(f"  ❌ Deployment files failed: {e}")
    
    print("\n✅ Production Configuration Management Test Complete!")

if __name__ == "__main__":
    main()
