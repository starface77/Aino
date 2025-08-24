#!/usr/bin/env python3
"""
Metrics - Система метрик и мониторинга AION
"""

import time
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import statistics

@dataclass
class MetricPoint:
    """Точка метрики"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }

@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    average_execution_time: float = 0.0
    operations_per_second: float = 0.0
    success_rate: float = 0.0
    
    def update(self, execution_time: float, success: bool = True):
        """Обновление метрик"""
        self.total_operations += 1
        self.total_execution_time += execution_time
        
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        
        # Обновляем min/max
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)
        
        # Пересчитываем средние значения
        if self.total_operations > 0:
            self.average_execution_time = self.total_execution_time / self.total_operations
            self.success_rate = self.successful_operations / self.total_operations
            
            # Скорость операций (приблизительная)
            if self.average_execution_time > 0:
                self.operations_per_second = 1.0 / self.average_execution_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'total_execution_time': self.total_execution_time,
            'min_execution_time': self.min_execution_time if self.min_execution_time != float('inf') else 0.0,
            'max_execution_time': self.max_execution_time,
            'average_execution_time': self.average_execution_time,
            'operations_per_second': self.operations_per_second,
            'success_rate': self.success_rate
        }

class MetricsCollector:
    """
    Коллектор метрик AION
    """
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.performance_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
        
        # Системные метрики
        self.start_time = datetime.now()
        self.last_reset_time = datetime.now()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """
        Записать метрику
        
        Args:
            name: Имя метрики
            value: Значение
            tags: Теги (опционально)
        """
        
        with self._lock:
            point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            
            self.metrics[name].append(point)
    
    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """
        Увеличить счетчик
        
        Args:
            name: Имя счетчика
            value: Значение для увеличения
            tags: Теги
        """
        
        with self._lock:
            self.counters[name] += value
            self.record_metric(f"{name}_counter", self.counters[name], tags)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """
        Установить значение gauge
        
        Args:
            name: Имя gauge
            value: Значение
            tags: Теги
        """
        
        with self._lock:
            self.gauges[name] = value
            self.record_metric(f"{name}_gauge", value, tags)
    
    def record_execution_time(self, operation: str, execution_time: float, 
                            success: bool = True, tags: Dict[str, str] = None):
        """
        Записать время выполнения операции
        
        Args:
            operation: Название операции
            execution_time: Время выполнения в секундах
            success: Успешность операции
            tags: Теги
        """
        
        with self._lock:
            # Обновляем метрики производительности
            self.performance_metrics[operation].update(execution_time, success)
            
            # Записываем как обычную метрику
            self.record_metric(f"{operation}_execution_time", execution_time, tags)
            
            # Записываем успешность
            status = "success" if success else "failure"
            combined_tags = (tags or {}).copy()
            combined_tags['status'] = status
            self.record_metric(f"{operation}_status", 1.0, combined_tags)
    
    def get_metrics(self, name: str, 
                   since: datetime = None, 
                   limit: int = None) -> List[MetricPoint]:
        """
        Получить метрики по имени
        
        Args:
            name: Имя метрики
            since: Получить метрики с определенного времени
            limit: Лимит количества точек
        
        Returns:
            List[MetricPoint]: Список точек метрик
        """
        
        with self._lock:
            points = list(self.metrics[name])
            
            # Фильтрация по времени
            if since:
                points = [p for p in points if p.timestamp >= since]
            
            # Лимит
            if limit:
                points = points[-limit:]
            
            return points
    
    def get_performance_metrics(self, operation: str = None) -> Dict[str, PerformanceMetrics]:
        """
        Получить метрики производительности
        
        Args:
            operation: Операция (если None, то все)
        
        Returns:
            Dict: Метрики производительности
        """
        
        with self._lock:
            if operation:
                return {operation: self.performance_metrics[operation]}
            else:
                return dict(self.performance_metrics)
    
    def get_counters(self) -> Dict[str, int]:
        """Получить все счетчики"""
        with self._lock:
            return dict(self.counters)
    
    def get_gauges(self) -> Dict[str, float]:
        """Получить все gauge'ы"""
        with self._lock:
            return dict(self.gauges)
    
    def get_statistics(self, name: str, 
                      since: datetime = None) -> Dict[str, float]:
        """
        Получить статистику по метрике
        
        Args:
            name: Имя метрики
            since: С какого времени
        
        Returns:
            Dict: Статистика (min, max, mean, median, std)
        """
        
        points = self.get_metrics(name, since)
        
        if not points:
            return {}
        
        values = [p.value for p in points]
        
        stats = {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'sum': sum(values)
        }
        
        if len(values) > 1:
            stats['median'] = statistics.median(values)
            stats['stdev'] = statistics.stdev(values)
        
        return stats
    
    def reset_metrics(self):
        """Сброс всех метрик"""
        with self._lock:
            self.metrics.clear()
            self.performance_metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.last_reset_time = datetime.now()
    
    def export_metrics(self, format: str = 'json') -> str:
        """
        Экспорт метрик
        
        Args:
            format: Формат экспорта ('json', 'prometheus')
        
        Returns:
            str: Экспортированные метрики
        """
        
        if format == 'json':
            return self._export_json()
        elif format == 'prometheus':
            return self._export_prometheus()
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}")
    
    def _export_json(self) -> str:
        """Экспорт в JSON"""
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'counters': self.get_counters(),
            'gauges': self.get_gauges(),
            'performance_metrics': {
                name: metrics.to_dict() 
                for name, metrics in self.performance_metrics.items()
            }
        }
        
        return json.dumps(data, indent=2, default=str)
    
    def _export_prometheus(self) -> str:
        """Экспорт в формате Prometheus"""
        
        lines = []
        
        # Счетчики
        for name, value in self.counters.items():
            lines.append(f"aion_{name}_total {value}")
        
        # Gauge'ы
        for name, value in self.gauges.items():
            lines.append(f"aion_{name} {value}")
        
        # Метрики производительности
        for operation, metrics in self.performance_metrics.items():
            prefix = f"aion_{operation}"
            lines.extend([
                f"{prefix}_operations_total {metrics.total_operations}",
                f"{prefix}_operations_successful_total {metrics.successful_operations}",
                f"{prefix}_operations_failed_total {metrics.failed_operations}",
                f"{prefix}_execution_time_seconds_total {metrics.total_execution_time}",
                f"{prefix}_execution_time_seconds_min {metrics.min_execution_time}",
                f"{prefix}_execution_time_seconds_max {metrics.max_execution_time}",
                f"{prefix}_execution_time_seconds_avg {metrics.average_execution_time}",
                f"{prefix}_operations_per_second {metrics.operations_per_second}",
                f"{prefix}_success_rate {metrics.success_rate}"
            ])
        
        return '\n'.join(lines)

class PerformanceMonitor:
    """
    Монитор производительности AION
    """
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.monitoring_active = False
        self.monitor_thread = None
        self.superhuman_thresholds = {
            'speed_multiplier': 1000.0,  # 1000x быстрее человека
            'accuracy_threshold': 0.999,  # 99.9% точность
            'max_response_time': 0.01,    # 10ms максимум
            'min_operations_per_second': 100  # 100 ops/sec минимум
        }
    
    def start_monitoring(self, interval: float = 1.0):
        """
        Запуск мониторинга
        
        Args:
            interval: Интервал мониторинга в секундах
        """
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitoring_loop(self, interval: float):
        """Основной цикл мониторинга"""
        
        import psutil
        
        while self.monitoring_active:
            try:
                # Системные метрики
                self.collector.set_gauge('system_cpu_percent', psutil.cpu_percent())
                self.collector.set_gauge('system_memory_percent', psutil.virtual_memory().percent)
                
                # Метрики процесса
                process = psutil.Process()
                self.collector.set_gauge('process_cpu_percent', process.cpu_percent())
                self.collector.set_gauge('process_memory_mb', process.memory_info().rss / 1024 / 1024)
                
                # Проверка сверхчеловеческих показателей
                self._check_superhuman_performance()
                
                time.sleep(interval)
                
            except Exception as e:
                # Логируем ошибку и продолжаем
                pass
    
    def _check_superhuman_performance(self):
        """Проверка сверхчеловеческих показателей"""
        
        performance_metrics = self.collector.get_performance_metrics()
        
        for operation, metrics in performance_metrics.items():
            if metrics.total_operations == 0:
                continue
            
            # Проверка скорости
            if metrics.operations_per_second >= self.superhuman_thresholds['min_operations_per_second']:
                self.collector.increment_counter('superhuman_speed_achievements')
            
            # Проверка точности
            if metrics.success_rate >= self.superhuman_thresholds['accuracy_threshold']:
                self.collector.increment_counter('superhuman_accuracy_achievements')
            
            # Проверка времени отклика
            if metrics.average_execution_time <= self.superhuman_thresholds['max_response_time']:
                self.collector.increment_counter('superhuman_response_achievements')
    
    def get_superhuman_report(self) -> Dict[str, Any]:
        """Получить отчет о сверхчеловеческих достижениях"""
        
        counters = self.collector.get_counters()
        performance_metrics = self.collector.get_performance_metrics()
        
        total_achievements = sum([
            counters.get('superhuman_speed_achievements', 0),
            counters.get('superhuman_accuracy_achievements', 0),
            counters.get('superhuman_response_achievements', 0)
        ])
        
        report = {
            'total_superhuman_achievements': total_achievements,
            'speed_achievements': counters.get('superhuman_speed_achievements', 0),
            'accuracy_achievements': counters.get('superhuman_accuracy_achievements', 0),
            'response_achievements': counters.get('superhuman_response_achievements', 0),
            'performance_summary': {}
        }
        
        # Сводка производительности
        for operation, metrics in performance_metrics.items():
            if metrics.total_operations > 0:
                human_baseline_time = 60.0  # 1 минута для человека
                speed_multiplier = human_baseline_time / max(metrics.average_execution_time, 0.001)
                
                report['performance_summary'][operation] = {
                    'speed_multiplier': f"{speed_multiplier:.1f}x",
                    'accuracy': f"{metrics.success_rate:.1%}",
                    'operations_per_second': f"{metrics.operations_per_second:.1f}",
                    'is_superhuman': speed_multiplier >= self.superhuman_thresholds['speed_multiplier']
                }
        
        return report

# Декоратор для автоматического измерения производительности
def measure_performance(collector: MetricsCollector, operation_name: str = None):
    """
    Декоратор для измерения производительности функций
    
    Args:
        collector: Коллектор метрик
        operation_name: Имя операции (по умолчанию имя функции)
    """
    
    def decorator(func):
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                execution_time = time.time() - start_time
                collector.record_execution_time(op_name, execution_time, success)
        
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            start_time = time.time()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                execution_time = time.time() - start_time
                collector.record_execution_time(op_name, execution_time, success)
        
        # Определяем, асинхронная ли функция
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Глобальный коллектор метрик
_global_collector: Optional[MetricsCollector] = None
_global_monitor: Optional[PerformanceMonitor] = None

def get_global_collector() -> MetricsCollector:
    """Получить глобальный коллектор метрик"""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector

def get_global_monitor() -> PerformanceMonitor:
    """Получить глобальный монитор производительности"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(get_global_collector())
    return _global_monitor

if __name__ == "__main__":
    # Демонстрация системы метрик
    
    print("📊 Тестирование системы метрик AION")
    
    # Создаем коллектор
    collector = MetricsCollector()
    
    # Записываем тестовые метрики
    collector.record_metric("test_metric", 42.0, {"component": "demo"})
    collector.increment_counter("operations", 5)
    collector.set_gauge("cpu_usage", 15.5)
    collector.record_execution_time("test_operation", 0.005, True)
    
    # Создаем монитор
    monitor = PerformanceMonitor(collector)
    
    # Получаем статистику
    stats = collector.get_statistics("test_metric")
    print(f"Статистика метрик: {stats}")
    
    performance = collector.get_performance_metrics()
    print(f"Метрики производительности: {performance}")
    
    # Экспорт метрик
    json_export = collector.export_metrics('json')
    print(f"JSON экспорт: {json_export[:200]}...")
    
    # Отчет о сверхчеловеческих достижениях
    superhuman_report = monitor.get_superhuman_report()
    print(f"Сверхчеловеческий отчет: {superhuman_report}")
    
    print("✅ Демонстрация системы метрик завершена")
