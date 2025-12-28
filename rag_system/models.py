from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from datetime import datetime

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class ProcessingResult:
    """Result of document processing"""
    success: bool
    message: str
    chunk_count: int = 0
    processing_time: float = 0.0
    source: str = ""

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
