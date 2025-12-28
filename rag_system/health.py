import time
import shutil
from typing import Dict, Any
from datetime import datetime

from .config import RAGConfig
from .models import HealthStatus, PerformanceMetrics

class HealthCheck:
    def __init__(self, config: RAGConfig, metrics: PerformanceMetrics):
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
