import logging
import logging.config
import time
from functools import wraps
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from .models import PerformanceMetrics, LogLevel

def setup_logging(log_level: LogLevel = LogLevel.INFO, 
                 log_to_file: bool = True, 
                 logs_dir: Path = Path("./logs"), 
                 max_log_size_mb: int = 10) -> None:
    """Setup comprehensive logging configuration"""
    
    # Ensure logs directory exists
    if log_to_file:
        logs_dir.mkdir(parents=True, exist_ok=True)

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
                'level': log_level.value,
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            }
        },
        'root': {
            'level': log_level.value,
            'handlers': ['console']
        },
        'loggers': {
            'rag_system': {
                'level': log_level.value,
                'handlers': ['console'],
                'propagate': False
            }
        }
    }

    if log_to_file:
        log_file = logs_dir / 'rag_system.log'
        log_config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level.value,
            'formatter': 'detailed',
            'filename': str(log_file),
            'maxBytes': max_log_size_mb * 1024 * 1024,
            'backupCount': 5,
            'encoding': 'utf-8'
        }
        log_config['root']['handlers'].append('file')
        log_config['loggers']['rag_system']['handlers'].append('file')

    logging.config.dictConfig(log_config)

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
