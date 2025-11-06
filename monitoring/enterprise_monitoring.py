"""
Enterprise Monitoring and Optimization System
============================================

Comprehensive monitoring, logging, health checks, and performance optimization
for the Research Copilot system at enterprise scale.

Features:
- Real-time metrics collection and alerting
- Performance monitoring and optimization
- Health checks and system diagnostics
- Caching and connection pooling
- Batch processing capabilities
- Scalability improvements

Author: Research Copilot System
Version: 2.0 Enterprise
"""

import os
import sys
import time
import json
import sqlite3
import threading
import asyncio
import logging
import psutil
import socket
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import pickle
import hashlib
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import weakref

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

# Configure logging with structured format
class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for enterprise logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)

# Setup enterprise logging
def setup_enterprise_logging(log_level: str = "INFO", log_file: str = "research_copilot.log"):
    """Setup comprehensive logging for enterprise deployment"""
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with structured logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(StructuredFormatter())
    logger.addHandler(file_handler)
    
    return logger

@dataclass
class SystemMetrics:
    """System-level metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int
    process_count: int

@dataclass
class ApplicationMetrics:
    """Application-level metrics"""
    timestamp: datetime
    active_requests: int
    requests_per_second: float
    avg_response_time: float
    error_rate: float
    cache_hit_rate: float
    database_connections: int
    queue_size: int
    memory_usage_mb: float

@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    metric_name: str
    threshold: float
    current_value: float
    alert_type: str  # 'warning', 'critical'
    timestamp: datetime
    message: str

class MetricsCollector:
    """Centralized metrics collection system"""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.system_metrics = deque(maxlen=1000)
        self.app_metrics = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        self.running = False
        self.thread = None
        
        # Prometheus metrics (if available)
        if HAS_PROMETHEUS:
            self.request_count = Counter('research_copilot_requests_total', 'Total requests')
            self.request_duration = Histogram('research_copilot_request_duration_seconds', 'Request duration')
            self.memory_usage = Gauge('research_copilot_memory_usage_bytes', 'Memory usage')
            self.cpu_usage = Gauge('research_copilot_cpu_usage_percent', 'CPU usage')
            self.cache_hits = Counter('research_copilot_cache_hits_total', 'Cache hits')
            self.cache_misses = Counter('research_copilot_cache_misses_total', 'Cache misses')
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': {'warning': 70, 'critical': 90},
            'memory_percent': {'warning': 80, 'critical': 95},
            'disk_usage_percent': {'warning': 85, 'critical': 95},
            'avg_response_time': {'warning': 5.0, 'critical': 10.0},
            'error_rate': {'warning': 0.05, 'critical': 0.1}
        }
        
        logger = logging.getLogger(__name__)
        logger.info("MetricsCollector initialized")
    
    def start_collection(self):
        """Start metrics collection in background thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.thread.start()
        
        logger = logging.getLogger(__name__)
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        logger = logging.getLogger(__name__)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        logger = logging.getLogger(__name__)
        
        while self.running:
            try:
                # Collect system metrics
                sys_metrics = self._collect_system_metrics()
                self.system_metrics.append(sys_metrics)
                
                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                self.app_metrics.append(app_metrics)
                
                # Check for alerts
                self._check_alerts(sys_metrics, app_metrics)
                
                # Update Prometheus metrics
                if HAS_PROMETHEUS:
                    self.memory_usage.set(sys_metrics.memory_used_mb * 1024 * 1024)
                    self.cpu_usage.set(sys_metrics.cpu_percent)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            net_io = psutil.net_io_counters()
            
            # Process info
            process = psutil.Process()
            connections = len(process.connections())
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_sent_mb=net_io.bytes_sent / (1024 * 1024),
                network_recv_mb=net_io.bytes_recv / (1024 * 1024),
                active_connections=connections,
                process_count=len(psutil.pids())
            )
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"System metrics collection failed: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0, memory_percent=0, memory_used_mb=0,
                disk_usage_percent=0, network_sent_mb=0, network_recv_mb=0,
                active_connections=0, process_count=0
            )
    
    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-level metrics"""
        # This would be populated by the application components
        # For now, return basic metrics
        return ApplicationMetrics(
            timestamp=datetime.now(),
            active_requests=0,
            requests_per_second=0.0,
            avg_response_time=0.0,
            error_rate=0.0,
            cache_hit_rate=0.0,
            database_connections=0,
            queue_size=0,
            memory_usage_mb=psutil.Process().memory_info().rss / (1024 * 1024)
        )
    
    def _check_alerts(self, sys_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Check metrics against alert thresholds"""
        current_time = datetime.now()
        
        # Check system metrics
        for metric_name, thresholds in self.alert_thresholds.items():
            if hasattr(sys_metrics, metric_name):
                value = getattr(sys_metrics, metric_name)
                
                if value >= thresholds['critical']:
                    alert = PerformanceAlert(
                        metric_name=metric_name,
                        threshold=thresholds['critical'],
                        current_value=value,
                        alert_type='critical',
                        timestamp=current_time,
                        message=f"CRITICAL: {metric_name} is {value:.2f}, exceeds threshold {thresholds['critical']:.2f}"
                    )
                    self.alerts.append(alert)
                    
                elif value >= thresholds['warning']:
                    alert = PerformanceAlert(
                        metric_name=metric_name,
                        threshold=thresholds['warning'],
                        current_value=value,
                        alert_type='warning',
                        timestamp=current_time,
                        message=f"WARNING: {metric_name} is {value:.2f}, exceeds threshold {thresholds['warning']:.2f}"
                    )
                    self.alerts.append(alert)
        
        # Check application metrics
        if hasattr(app_metrics, 'avg_response_time') and app_metrics.avg_response_time > 0:
            if app_metrics.avg_response_time >= self.alert_thresholds['avg_response_time']['critical']:
                alert = PerformanceAlert(
                    metric_name='avg_response_time',
                    threshold=self.alert_thresholds['avg_response_time']['critical'],
                    current_value=app_metrics.avg_response_time,
                    alert_type='critical',
                    timestamp=current_time,
                    message=f"CRITICAL: Average response time is {app_metrics.avg_response_time:.2f}s"
                )
                self.alerts.append(alert)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        if not self.system_metrics or not self.app_metrics:
            return {}
        
        # Latest metrics
        latest_sys = self.system_metrics[-1]
        latest_app = self.app_metrics[-1]
        
        # Historical averages (last 10 data points)
        recent_sys = list(self.system_metrics)[-10:]
        recent_app = list(self.app_metrics)[-10:]
        
        sys_avg_cpu = sum(m.cpu_percent for m in recent_sys) / len(recent_sys)
        sys_avg_memory = sum(m.memory_percent for m in recent_sys) / len(recent_sys)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'current': asdict(latest_sys),
                'averages': {
                    'cpu_percent': sys_avg_cpu,
                    'memory_percent': sys_avg_memory
                }
            },
            'application': {
                'current': asdict(latest_app)
            },
            'alerts': {
                'total': len(self.alerts),
                'critical': len([a for a in self.alerts if a.alert_type == 'critical']),
                'warning': len([a for a in self.alerts if a.alert_type == 'warning']),
                'recent': [asdict(alert) for alert in list(self.alerts)[-5:]]
            }
        }

class HealthChecker:
    """System health monitoring and diagnostics"""
    
    def __init__(self, db_path: str = "papers.db"):
        self.db_path = db_path
        self.health_checks = {}
        self.last_check_time = None
        self.logger = logging.getLogger(__name__)
        
        # Register default health checks
        self.register_check("database", self._check_database)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory", self._check_memory)
        self.register_check("network", self._check_network)
    
    def register_check(self, name: str, check_func: Callable[[], Tuple[bool, str]]):
        """Register a health check function"""
        self.health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return results"""
        start_time = time.time()
        results = {}
        overall_healthy = True
        
        for check_name, check_func in self.health_checks.items():
            try:
                is_healthy, message = check_func()
                results[check_name] = {
                    'healthy': is_healthy,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                }
                
                if not is_healthy:
                    overall_healthy = False
                    self.logger.warning(f"Health check failed: {check_name} - {message}")
                
            except Exception as e:
                results[check_name] = {
                    'healthy': False,
                    'message': f"Health check error: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                }
                overall_healthy = False
                self.logger.error(f"Health check error for {check_name}: {e}")
        
        self.last_check_time = datetime.now()
        
        return {
            'overall_healthy': overall_healthy,
            'timestamp': self.last_check_time.isoformat(),
            'duration': time.time() - start_time,
            'checks': results
        }
    
    def _check_database(self) -> Tuple[bool, str]:
        """Check database connectivity and basic operations"""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                count = cursor.fetchone()[0]
                return True, f"Database accessible, {count} papers"
        except Exception as e:
            return False, f"Database error: {str(e)}"
    
    def _check_disk_space(self) -> Tuple[bool, str]:
        """Check available disk space"""
        try:
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            
            if free_percent < 5:
                return False, f"Critical: Only {free_percent:.1f}% disk space remaining"
            elif free_percent < 15:
                return False, f"Warning: Only {free_percent:.1f}% disk space remaining"
            else:
                return True, f"Disk space OK: {free_percent:.1f}% free"
        except Exception as e:
            return False, f"Disk check error: {str(e)}"
    
    def _check_memory(self) -> Tuple[bool, str]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 95:
                return False, f"Critical: Memory usage at {memory.percent:.1f}%"
            elif memory.percent > 85:
                return False, f"Warning: Memory usage at {memory.percent:.1f}%"
            else:
                return True, f"Memory OK: {memory.percent:.1f}% used"
        except Exception as e:
            return False, f"Memory check error: {str(e)}"
    
    def _check_network(self) -> Tuple[bool, str]:
        """Check network connectivity"""
        try:
            # Try to connect to a well-known service
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('8.8.8.8', 53))  # Google DNS
            sock.close()
            
            if result == 0:
                return True, "Network connectivity OK"
            else:
                return False, "Network connectivity issues detected"
        except Exception as e:
            return False, f"Network check error: {str(e)}"

class ConnectionPool:
    """Database connection pooling for improved performance"""
    
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self.pool = queue.Queue(maxsize=max_connections)
        self.created_connections = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Pre-create connections
        for _ in range(min(3, max_connections)):  # Start with 3 connections
            self._create_connection()
    
    def _create_connection(self):
        """Create a new database connection"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            self.pool.put(conn)
            self.created_connections += 1
            self.logger.debug(f"Created database connection ({self.created_connections}/{self.max_connections})")
        except Exception as e:
            self.logger.error(f"Failed to create database connection: {e}")
            raise
    
    @contextmanager
    def get_connection(self, timeout: float = 5.0):
        """Get a connection from the pool"""
        conn = None
        try:
            # Try to get existing connection
            conn = self.pool.get(timeout=timeout)
            yield conn
        except queue.Empty:
            # Create new connection if pool is empty and we haven't hit the limit
            with self.lock:
                if self.created_connections < self.max_connections:
                    self._create_connection()
                    conn = self.pool.get(timeout=timeout)
                    yield conn
                else:
                    raise Exception("Connection pool exhausted")
        finally:
            if conn:
                # Return connection to pool
                try:
                    # Test connection before returning
                    conn.execute("SELECT 1")
                    self.pool.put(conn)
                except Exception as e:
                    self.logger.warning(f"Connection test failed, creating new one: {e}")
                    self._create_connection()

class BatchProcessor:
    """Batch processing system for improved throughput"""
    
    def __init__(self, batch_size: int = 100, max_wait_time: float = 5.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_items = []
        self.last_batch_time = time.time()
        self.lock = threading.Lock()
        self.processors = {}
        self.logger = logging.getLogger(__name__)
    
    def register_processor(self, item_type: str, processor_func: Callable[[List[Any]], None]):
        """Register a batch processor function"""
        self.processors[item_type] = processor_func
        self.logger.info(f"Registered batch processor for: {item_type}")
    
    def add_item(self, item_type: str, item: Any):
        """Add item to batch for processing"""
        with self.lock:
            self.pending_items.append((item_type, item))
            
            # Check if we should process the batch
            if (len(self.pending_items) >= self.batch_size or 
                time.time() - self.last_batch_time > self.max_wait_time):
                self._process_batch()
    
    def _process_batch(self):
        """Process current batch of items"""
        if not self.pending_items:
            return
        
        # Group items by type
        batches = defaultdict(list)
        for item_type, item in self.pending_items:
            batches[item_type].append(item)
        
        # Process each batch
        for item_type, items in batches.items():
            if item_type in self.processors:
                try:
                    self.processors[item_type](items)
                    self.logger.debug(f"Processed batch of {len(items)} items for {item_type}")
                except Exception as e:
                    self.logger.error(f"Batch processing failed for {item_type}: {e}")
        
        # Clear processed items
        self.pending_items.clear()
        self.last_batch_time = time.time()
    
    def force_process(self):
        """Force processing of current batch"""
        with self.lock:
            self._process_batch()

class CacheOptimizer:
    """Advanced caching with optimization strategies"""
    
    def __init__(self, max_memory_mb: int = 500):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.caches = {}
        self.cache_stats = defaultdict(lambda: {'hits': 0, 'misses': 0, 'size': 0})
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def register_cache(self, cache_name: str, cache_instance):
        """Register a cache for monitoring and optimization"""
        with self.lock:
            self.caches[cache_name] = weakref.ref(cache_instance)
            self.logger.info(f"Registered cache: {cache_name}")
    
    def optimize_caches(self):
        """Optimize cache performance and memory usage"""
        with self.lock:
            total_memory = 0
            active_caches = {}
            
            # Clean up dead references and calculate memory usage
            for cache_name, cache_ref in list(self.caches.items()):
                cache = cache_ref()
                if cache is None:
                    del self.caches[cache_name]
                    continue
                
                active_caches[cache_name] = cache
                
                # Estimate memory usage
                if hasattr(cache, 'cache') and hasattr(cache.cache, '__sizeof__'):
                    cache_memory = sys.getsizeof(cache.cache)
                    total_memory += cache_memory
                    self.cache_stats[cache_name]['size'] = cache_memory
            
            # If over memory limit, optimize
            if total_memory > self.max_memory_bytes:
                self.logger.warning(f"Cache memory usage ({total_memory / (1024*1024):.1f}MB) exceeds limit")
                self._optimize_memory_usage(active_caches)
    
    def _optimize_memory_usage(self, caches: Dict[str, Any]):
        """Optimize memory usage across caches"""
        # Sort caches by efficiency (hit rate / memory usage)
        cache_efficiency = []
        
        for cache_name, cache in caches.items():
            stats = self.cache_stats[cache_name]
            total_requests = stats['hits'] + stats['misses']
            hit_rate = stats['hits'] / total_requests if total_requests > 0 else 0
            memory_mb = stats['size'] / (1024 * 1024)
            efficiency = hit_rate / max(memory_mb, 0.1)  # Avoid division by zero
            
            cache_efficiency.append((cache_name, cache, efficiency))
        
        # Sort by efficiency (lowest first for cleanup)
        cache_efficiency.sort(key=lambda x: x[2])
        
        # Clear least efficient caches first
        for cache_name, cache, efficiency in cache_efficiency:
            if hasattr(cache, 'clear'):
                cache.clear()
                self.logger.info(f"Cleared cache {cache_name} (efficiency: {efficiency:.2f})")
                
                # Check if we've freed enough memory
                break  # For now, just clear the least efficient cache
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self.lock:
            stats = {}
            total_hits = 0
            total_misses = 0
            total_memory = 0
            
            for cache_name, cache_stats in self.cache_stats.items():
                stats[cache_name] = dict(cache_stats)
                total_hits += cache_stats['hits']
                total_misses += cache_stats['misses']
                total_memory += cache_stats['size']
            
            overall_hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
            
            return {
                'caches': stats,
                'summary': {
                    'total_hits': total_hits,
                    'total_misses': total_misses,
                    'overall_hit_rate': overall_hit_rate,
                    'total_memory_mb': total_memory / (1024 * 1024),
                    'memory_limit_mb': self.max_memory_bytes / (1024 * 1024)
                }
            }

class EnterpriseMonitoringSystem:
    """Main monitoring system orchestrating all components"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            collection_interval=self.config.get('metrics_interval', 30)
        )
        self.health_checker = HealthChecker(
            db_path=self.config.get('db_path', 'papers.db')
        )
        self.connection_pool = ConnectionPool(
            db_path=self.config.get('db_path', 'papers.db'),
            max_connections=self.config.get('max_db_connections', 10)
        )
        self.batch_processor = BatchProcessor(
            batch_size=self.config.get('batch_size', 100),
            max_wait_time=self.config.get('batch_wait_time', 5.0)
        )
        self.cache_optimizer = CacheOptimizer(
            max_memory_mb=self.config.get('cache_memory_limit_mb', 500)
        )
        
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Start Prometheus metrics server if available
        if HAS_PROMETHEUS and self.config.get('prometheus_port'):
            try:
                start_http_server(self.config['prometheus_port'])
                self.logger.info(f"Prometheus metrics server started on port {self.config['prometheus_port']}")
            except Exception as e:
                self.logger.warning(f"Failed to start Prometheus server: {e}")
    
    def start(self):
        """Start all monitoring components"""
        if self.running:
            return
        
        self.running = True
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start periodic optimization
        self._start_optimization_loop()
        
        self.logger.info("Enterprise monitoring system started")
    
    def stop(self):
        """Stop all monitoring components"""
        self.running = False
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        # Force process any pending batches
        self.batch_processor.force_process()
        
        self.logger.info("Enterprise monitoring system stopped")
    
    def start_monitoring(self):
        """Alias for start() method"""
        return self.start()
    
    def stop_monitoring(self):
        """Alias for stop() method"""
        return self.stop()
    
    def health_check(self) -> Dict[str, Any]:
        """Get comprehensive health check status"""
        return self.health_checker.run_health_checks()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return self.metrics_collector.get_metrics_summary()
    
    def _start_optimization_loop(self):
        """Start background optimization loop"""
        def optimization_loop():
            while self.running:
                try:
                    # Optimize caches every 5 minutes
                    self.cache_optimizer.optimize_caches()
                    
                    # Force process any pending batches
                    self.batch_processor.force_process()
                    
                    time.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    self.logger.error(f"Optimization loop error: {e}")
                    time.sleep(60)  # Wait 1 minute on error
        
        thread = threading.Thread(target=optimization_loop, daemon=True)
        thread.start()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        # Run health checks
        health_status = self.health_checker.run_health_checks()
        
        # Get metrics
        metrics_summary = self.metrics_collector.get_metrics_summary()
        
        # Get cache statistics
        cache_stats = self.cache_optimizer.get_cache_stats()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_healthy': health_status['overall_healthy'],
            'uptime_seconds': time.time() - self.config.get('start_time', time.time()),
            'health_checks': health_status,
            'metrics': metrics_summary,
            'cache_performance': cache_stats,
            'configuration': {
                'metrics_interval': self.metrics_collector.collection_interval,
                'max_db_connections': self.connection_pool.max_connections,
                'batch_size': self.batch_processor.batch_size,
                'cache_memory_limit_mb': self.cache_optimizer.max_memory_bytes / (1024 * 1024)
            }
        }
    
    def register_batch_processor(self, item_type: str, processor_func: Callable):
        """Register a batch processor"""
        self.batch_processor.register_processor(item_type, processor_func)
    
    def register_cache(self, cache_name: str, cache_instance):
        """Register a cache for monitoring"""
        self.cache_optimizer.register_cache(cache_name, cache_instance)
    
    def get_database_connection(self):
        """Get a database connection from the pool"""
        return self.connection_pool.get_connection()

def main():
    """Main function for testing the monitoring system"""
    print("📊 Enterprise Monitoring System Test")
    print("=" * 50)
    
    # Setup logging
    logger = setup_enterprise_logging()
    
    # Initialize monitoring system
    config = {
        'metrics_interval': 10,
        'db_path': 'papers.db',
        'max_db_connections': 5,
        'batch_size': 50,
        'cache_memory_limit_mb': 100,
        'start_time': time.time()
    }
    
    monitoring = EnterpriseMonitoringSystem(config)
    
    try:
        # Start monitoring
        monitoring.start()
        
        # Simulate some activity
        logger.info("Simulating system activity...")
        
        # Wait for metrics collection
        time.sleep(15)
        
        # Get system status
        status = monitoring.get_system_status()
        
        print("\n📊 System Status:")
        print(f"System Healthy: {status['system_healthy']}")
        print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
        
        print("\n💾 Health Checks:")
        for check_name, check_result in status['health_checks']['checks'].items():
            status_icon = "✅" if check_result['healthy'] else "❌"
            print(f"  {status_icon} {check_name}: {check_result['message']}")
        
        if 'metrics' in status and 'system' in status['metrics']:
            sys_metrics = status['metrics']['system']['current']
            print(f"\n📈 Current Metrics:")
            print(f"  CPU: {sys_metrics['cpu_percent']:.1f}%")
            print(f"  Memory: {sys_metrics['memory_percent']:.1f}%")
            print(f"  Active Connections: {sys_metrics['active_connections']}")
        
        if 'alerts' in status['metrics']:
            alerts = status['metrics']['alerts']
            print(f"\n🚨 Alerts: {alerts['total']} total ({alerts['critical']} critical, {alerts['warning']} warnings)")
        
        print("\n✅ Enterprise Monitoring System Test Complete!")
        
    except KeyboardInterrupt:
        print("\nStopping monitoring system...")
    finally:
        monitoring.stop()

if __name__ == "__main__":
    main()
