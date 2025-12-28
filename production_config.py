# production_config.py - Production configuration and utilities
import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class ProductionConfig(BaseModel):
    """Production-ready configuration with validation"""

    # API Configuration
    groq_api_key: str = Field(..., description="Groq API key")
    model_name: str = Field(default="llama-3.3-70b-versatile")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)

    # Processing Configuration
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_file_size_mb: int = Field(default=50, ge=1, le=500)

    # Vector Store Configuration
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    vector_store_type: str = Field(default="faiss", pattern="^(faiss|chroma|pinecone)$")
    retrieval_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    # Caching Configuration
    enable_cache: bool = Field(default=True)
    cache_ttl_hours: int = Field(default=24, ge=1, le=168)  # Max 1 week
    redis_url: Optional[str] = Field(default=None, description="Redis URL for distributed caching")

    # File System Configuration
    data_dir: Path = Field(default=Path("./data"))
    cache_dir: Path = Field(default=Path("./cache"))
    vector_store_dir: Path = Field(default=Path("./vector_stores"))
    logs_dir: Path = Field(default=Path("./logs"))

    # Performance Configuration
    max_concurrent_downloads: int = Field(default=3, ge=1, le=10)
    request_timeout: int = Field(default=60, ge=10, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)

    # Logging Configuration
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_to_file: bool = Field(default=True)
    max_log_size_mb: int = Field(default=10, ge=1, le=100)

    # Security Configuration
    allowed_file_types: list = Field(default=[".pdf", ".txt", ".docx"])
    max_url_redirects: int = Field(default=5, ge=0, le=10)

    @validator('data_dir', 'cache_dir', 'vector_store_dir', 'logs_dir')
    def create_directories(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v

    @validator('max_file_size_mb')
    def convert_to_bytes(cls, v):
        return v * 1024 * 1024

    @classmethod
    def from_yaml(cls, config_path: Path) -> 'ProductionConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    @classmethod
    def from_env(cls) -> 'ProductionConfig':
        """Load configuration from environment variables"""
        config_data = {}

        env_mapping = {
            'GROQ_API_KEY': 'groq_api_key',
            'MODEL_NAME': 'model_name',
            'TEMPERATURE': 'temperature',
            'CHUNK_SIZE': 'chunk_size',
            'CHUNK_OVERLAP': 'chunk_overlap',
            'MAX_FILE_SIZE_MB': 'max_file_size_mb',
            'EMBEDDING_MODEL': 'embedding_model',
            'ENABLE_CACHE': 'enable_cache',
            'CACHE_TTL_HOURS': 'cache_ttl_hours',
            'REDIS_URL': 'redis_url',
            'LOG_LEVEL': 'log_level',
        }

        for env_var, config_field in env_mapping.items():
            if env_value := os.getenv(env_var):
                if config_field in ['temperature', 'score_threshold']:
                    config_data[config_field] = float(env_value)
                elif config_field in ['chunk_size', 'chunk_overlap', 'max_file_size_mb', 'cache_ttl_hours']:
                    config_data[config_field] = int(env_value)
                elif config_field in ['enable_cache', 'log_to_file']:
                    config_data[config_field] = env_value.lower() in ['true', '1', 'yes']
                else:
                    config_data[config_field] = env_value

        return cls(**config_data)


# --------------------------------------------------------------------
# Logging Setup
# --------------------------------------------------------------------
import logging.config

def setup_logging(config: ProductionConfig) -> None:
    """Setup comprehensive logging configuration"""
    log_config: Dict[str, Any] = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(levelname)s - %(message)s'
            },
            'json': {
                'format': '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s", "file": "%(filename)s", "line": %(lineno)d}',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': config.log_level.value,
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            }
        },
        'root': {
            'level': config.log_level.value,
            'handlers': ['console']
        },
        'loggers': {
            'rag_system': {
                'level': config.log_level.value,
                'handlers': ['console'],
                'propagate': False
            }
        }
    }

    if config.log_to_file:
        log_file = config.logs_dir / 'rag_system.log'
        log_config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': config.log_level.value,
            'formatter': 'detailed',
            'filename': str(log_file),
            'maxBytes': config.max_log_size_mb * 1024 * 1024,
            'backupCount': 5,
            'encoding': 'utf-8'
        }
        log_config['root']['handlers'].append('file')
        log_config['loggers']['rag_system']['handlers'].append('file')

    logging.config.dictConfig(log_config)


# --------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------
import time
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class PerformanceMetrics:
    documents_processed: int = 0
    total_processing_time: float = 0.0
    questions_answered: int = 0
    total_qa_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: list = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def avg_processing_time(self) -> float:
        return self.total_processing_time / self.documents_processed if self.documents_processed else 0.0

    @property
    def avg_qa_time(self) -> float:
        return self.total_qa_time / self.questions_answered if self.questions_answered else 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0

    @property
    def uptime(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    def to_dict(self) -> dict:
        return {
            'documents_processed': self.documents_processed,
            'avg_processing_time': self.avg_processing_time,
            'questions_answered': self.questions_answered,
            'avg_qa_time': self.avg_qa_time,
            'cache_hit_rate': self.cache_hit_rate,
            'error_count': len(self.errors),
            'uptime_seconds': self.uptime
        }

def monitor_performance(metrics: PerformanceMetrics, operation: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start
                if operation == 'document_processing':
                    metrics.documents_processed += 1
                    metrics.total_processing_time += elapsed
                elif operation == 'question_answering':
                    metrics.questions_answered += 1
                    metrics.total_qa_time += elapsed
                return result
            except Exception as e:
                metrics.errors.append({'operation': operation, 'error': str(e), 'timestamp': datetime.now().isoformat()})
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                if operation == 'question_answering':
                    metrics.questions_answered += 1
                    metrics.total_qa_time += elapsed
                return result
            except Exception as e:
                metrics.errors.append({'operation': operation, 'error': str(e), 'timestamp': datetime.now().isoformat()})
                raise

        import asyncio
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# --------------------------------------------------------------------
# Health Check
# --------------------------------------------------------------------
from typing import Dict

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheck:
    def __init__(self, config: ProductionConfig, metrics: PerformanceMetrics):
        self.config = config
        self.metrics = metrics

    async def check_system_health(self) -> Dict[str, Any]:
        checks = {
            'api_connectivity': await self._check_api_connectivity(),
            'disk_space': self._check_disk_space(),
            'memory_usage': self._check_memory_usage(),
            'error_rate': self._check_error_rate(),
            'response_times': self._check_response_times(),
            'cache_performance': self._check_cache_performance()
        }

        unhealthy = [n for n, s in checks.items() if s['status'] == HealthStatus.UNHEALTHY.value]
        degraded = [n for n, s in checks.items() if s['status'] == HealthStatus.DEGRADED.value]

        if unhealthy:
            overall = HealthStatus.UNHEALTHY
        elif degraded:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return {
            'overall_status': overall.value,
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': self.metrics.uptime,
            'checks': checks,
            'issues': unhealthy + degraded
        }

    async def _check_api_connectivity(self) -> Dict[str, Any]:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                groq_api_key=self.config.groq_api_key,
                model_name=self.config.model_name,
                temperature=self.config.temperature
            )
            start = time.time()
            _ = llm.invoke("Test connection")
            elapsed = time.time() - start
            status = HealthStatus.HEALTHY if elapsed < 10 else HealthStatus.DEGRADED
            return {'status': status.value, 'response_time': elapsed, 'message': 'API connectivity OK'}
        except Exception as e:
            return {'status': HealthStatus.UNHEALTHY.value, 'error': str(e), 'message': 'API connectivity failed'}

    def _check_disk_space(self) -> Dict[str, Any]:
        import shutil
        try:
            total, used, free = shutil.disk_usage(self.config.data_dir)
            free_pct = (free / total) * 100
            if free_pct < 5:
                status, msg = HealthStatus.UNHEALTHY, "Critical: <5% disk space"
            elif free_pct < 15:
                status, msg = HealthStatus.DEGRADED, "Warning: <15% disk space"
            else:
                status, msg = HealthStatus.HEALTHY, "Disk space OK"
            return {'status': status.value, 'free_space_gb': free // (1024**3), 'free_percent': round(free_pct, 2), 'message': msg}
        except Exception as e:
            return {'status': HealthStatus.UNHEALTHY.value, 'error': str(e), 'message': 'Disk check failed'}

    def _check_memory_usage(self) -> Dict[str, Any]:
        try:
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                status, msg = HealthStatus.UNHEALTHY, "Critical: >90% memory"
            elif mem.percent > 75:
                status, msg = HealthStatus.DEGRADED, "Warning: >75% memory"
            else:
                status, msg = HealthStatus.HEALTHY, "Memory OK"
            return {'status': status.value, 'memory_percent': mem.percent, 'available_gb': mem.available // (1024**3), 'message': msg}
        except ImportError:
            return {'status': HealthStatus.DEGRADED.value, 'message': 'psutil not available'}
        except Exception as e:
            return {'status': HealthStatus.UNHEALTHY.value, 'error': str(e), 'message': 'Memory check failed'}

    def _check_error_rate(self) -> Dict[str, Any]:
        total_ops = self.metrics.documents_processed + self.metrics.questions_answered
        error_rate = len(self.metrics.errors) / total_ops if total_ops else 0
        if error_rate > 0.1:
            status, msg = HealthStatus.UNHEALTHY, "Critical: >10% error rate"
        elif error_rate > 0.05:
            status, msg = HealthStatus.DEGRADED, "Warning: >5% error rate"
        else:
            status, msg = HealthStatus.HEALTHY, "Error rate OK"
        return {'status': status.value, 'error_rate': round(error_rate * 100, 2), 'total_errors': len(self.metrics.errors), 'message': msg}

    def _check_response_times(self) -> Dict[str, Any]:
        avg_proc = self.metrics.avg_processing_time
        avg_qa = self.metrics.avg_qa_time
        if avg_proc > 120:
            status, msg = HealthStatus.UNHEALTHY, "Critical: avg proc >120s"
        elif avg_proc > 60:
            status, msg = HealthStatus.DEGRADED, "Warning: avg proc >60s"
        elif avg_qa > 30:
            status, msg = HealthStatus.DEGRADED, "Warning: avg QA >30s"
        else:
            status, msg = HealthStatus.HEALTHY, "Response times OK"
        return {'status': status.value, 'avg_processing_time': round(avg_proc, 2), 'avg_qa_time': round(avg_qa, 2), 'message': msg}

    def _check_cache_performance(self) -> Dict[str, Any]:
        hit_rate = self.metrics.cache_hit_rate
        if hit_rate < 0.3:
            status, msg = HealthStatus.DEGRADED, "Warning: low cache hit rate"
        else:
            status, msg = HealthStatus.HEALTHY, "Cache OK"
        return {'status': status.value, 'hit_rate': round(hit_rate * 100, 2), 'cache_hits': self.metrics.cache_hits, 'cache_misses': self.metrics.cache_misses, 'message': msg}
