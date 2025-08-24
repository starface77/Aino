#!/usr/bin/env python3
"""
AION Monitoring System
Система мониторинга производительности и метрик AION
"""

import psutil
import json
import time
import os
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading

class AIONMonitoring:
    """Система мониторинга AION"""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.start_time = datetime.now()
        self.running = False
        self.monitor_thread = None
        
        # Метрики производительности
        self.performance_metrics = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'disk_usage': deque(maxlen=100),
            'process_count': deque(maxlen=100)
        }
        
        # Метрики AION
        self.aion_metrics = {
            'analyses_completed': 0,
            'ai_requests': 0,
            'fixes_applied': 0,
            'errors_encountered': 0,
            'average_response_time': 0,
            'uptime': 0
        }
        
        # Лог событий
        self.events_log = deque(maxlen=500)
        
        self._load_historical_data()
    
    def start_monitoring(self, interval=5):
        """Запуск мониторинга"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.log_event('system', 'Мониторинг запущен')
    
    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        self.log_event('system', 'Мониторинг остановлен')
        self._save_historical_data()
    
    def _monitor_loop(self, interval):
        """Главный цикл мониторинга"""
        while self.running:
            try:
                self._collect_system_metrics()
                self._update_aion_metrics()
                time.sleep(interval)
            except Exception as e:
                self.log_event('error', f'Ошибка мониторинга: {e}')
    
    def _collect_system_metrics(self):
        """Сбор системных метрик"""
        try:
            # CPU использование
            cpu_percent = psutil.cpu_percent(interval=1)
            self.performance_metrics['cpu_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'value': cpu_percent
            })
            
            # Память
            memory = psutil.virtual_memory()
            self.performance_metrics['memory_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'value': memory.percent,
                'used_gb': round(memory.used / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2)
            })
            
            # Диск
            disk = psutil.disk_usage('.')
            self.performance_metrics['disk_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'value': (disk.used / disk.total) * 100,
                'used_gb': round(disk.used / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2)
            })
            
            # Количество процессов
            process_count = len(psutil.pids())
            self.performance_metrics['process_count'].append({
                'timestamp': datetime.now().isoformat(),
                'value': process_count
            })
            
        except Exception as e:
            self.log_event('error', f'Ошибка сбора системных метрик: {e}')
    
    def _update_aion_metrics(self):
        """Обновление метрик AION"""
        try:
            # Обновляем uptime
            uptime = (datetime.now() - self.start_time).total_seconds()
            self.aion_metrics['uptime'] = uptime
            
            # Читаем логи для обновления счетчиков
            self._update_from_logs()
            
        except Exception as e:
            self.log_event('error', f'Ошибка обновления AION метрик: {e}')
    
    def _update_from_logs(self):
        """Обновление метрик из логов"""
        try:
            # Анализы
            if os.path.exists('aion_log.json'):
                with open('aion_log.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.aion_metrics['analyses_completed'] = len(data.get('runs', []))
            
            # AI запросы
            if os.path.exists('ai_analysis_log.json'):
                with open('ai_analysis_log.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'latest_analysis' in data:
                        self.aion_metrics['ai_requests'] += 1
            
        except Exception as e:
            self.log_event('error', f'Ошибка чтения логов: {e}')
    
    def log_event(self, event_type, message):
        """Логирование события"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'message': message
        }
        self.events_log.append(event)
        
        # Обновляем счетчики
        if event_type == 'error':
            self.aion_metrics['errors_encountered'] += 1
        elif event_type == 'analysis':
            self.aion_metrics['analyses_completed'] += 1
        elif event_type == 'ai_request':
            self.aion_metrics['ai_requests'] += 1
        elif event_type == 'fix_applied':
            self.aion_metrics['fixes_applied'] += 1
    
    def get_system_status(self):
        """Получить текущий статус системы"""
        try:
            current_cpu = psutil.cpu_percent(interval=1)
            current_memory = psutil.virtual_memory()
            current_disk = psutil.disk_usage('.')
            
            return {
                'status': 'healthy' if current_cpu < 80 and current_memory.percent < 80 else 'warning',
                'uptime': self.aion_metrics['uptime'],
                'cpu_percent': current_cpu,
                'memory_percent': current_memory.percent,
                'disk_percent': (current_disk.used / current_disk.total) * 100,
                'processes': len(psutil.pids()),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_aion_stats(self):
        """Получить статистику AION"""
        return {
            'analyses_completed': self.aion_metrics['analyses_completed'],
            'ai_requests': self.aion_metrics['ai_requests'],
            'fixes_applied': self.aion_metrics['fixes_applied'],
            'errors_encountered': self.aion_metrics['errors_encountered'],
            'uptime_seconds': self.aion_metrics['uptime'],
            'uptime_human': self._format_uptime(self.aion_metrics['uptime']),
            'average_response_time': self.aion_metrics['average_response_time'],
            'events_count': len(self.events_log)
        }
    
    def get_performance_metrics(self, metric_type='all', limit=50):
        """Получить метрики производительности"""
        if metric_type == 'all':
            return {
                key: list(values)[-limit:] 
                for key, values in self.performance_metrics.items()
            }
        elif metric_type in self.performance_metrics:
            return list(self.performance_metrics[metric_type])[-limit:]
        else:
            return []
    
    def get_recent_events(self, limit=20, event_type=None):
        """Получить последние события"""
        events = list(self.events_log)[-limit:]
        
        if event_type:
            events = [e for e in events if e['type'] == event_type]
        
        return events
    
    def get_health_report(self):
        """Получить отчет о здоровье системы"""
        system_status = self.get_system_status()
        aion_stats = self.get_aion_stats()
        recent_errors = self.get_recent_events(10, 'error')
        
        # Определяем общее здоровье
        health_score = 100
        warnings = []
        
        # Проверки системы
        if system_status['cpu_percent'] > 80:
            health_score -= 20
            warnings.append('Высокое использование CPU')
        
        if system_status['memory_percent'] > 80:
            health_score -= 20
            warnings.append('Высокое использование памяти')
        
        if system_status['disk_percent'] > 90:
            health_score -= 15
            warnings.append('Мало места на диске')
        
        # Проверки AION
        if len(recent_errors) > 5:
            health_score -= 25
            warnings.append('Много ошибок в последнее время')
        
        if aion_stats['uptime_seconds'] < 300:  # Меньше 5 минут
            health_score -= 10
            warnings.append('Недавний перезапуск')
        
        return {
            'health_score': max(0, health_score),
            'status': 'excellent' if health_score >= 90 else 'good' if health_score >= 70 else 'warning' if health_score >= 50 else 'critical',
            'warnings': warnings,
            'system': system_status,
            'aion': aion_stats,
            'recent_errors': recent_errors,
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_uptime(self, seconds):
        """Форматирование времени работы"""
        if seconds < 60:
            return f"{int(seconds)} сек"
        elif seconds < 3600:
            return f"{int(seconds // 60)} мин {int(seconds % 60)} сек"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours} ч {minutes} мин"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            return f"{days} дн {hours} ч"
    
    def _save_historical_data(self):
        """Сохранение исторических данных"""
        try:
            data = {
                'performance_metrics': {
                    key: list(values) for key, values in self.performance_metrics.items()
                },
                'aion_metrics': self.aion_metrics,
                'events_log': list(self.events_log),
                'saved_at': datetime.now().isoformat()
            }
            
            with open('monitoring_data.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Ошибка сохранения данных мониторинга: {e}")
    
    def _load_historical_data(self):
        """Загрузка исторических данных"""
        try:
            if os.path.exists('monitoring_data.json'):
                with open('monitoring_data.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Восстанавливаем метрики
                for key, values in data.get('performance_metrics', {}).items():
                    if key in self.performance_metrics:
                        self.performance_metrics[key].extend(values[-50:])  # Последние 50
                
                # Восстанавливаем AION метрики
                saved_aion = data.get('aion_metrics', {})
                for key in self.aion_metrics:
                    if key in saved_aion and key != 'uptime':  # uptime пересчитываем
                        self.aion_metrics[key] = saved_aion[key]
                
                # Восстанавливаем события
                saved_events = data.get('events_log', [])
                self.events_log.extend(saved_events[-100:])  # Последние 100
                
        except Exception as e:
            print(f"Ошибка загрузки данных мониторинга: {e}")

# Глобальный экземпляр мониторинга
_monitor_instance = None

def get_monitor():
    """Получить экземпляр мониторинга (singleton)"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = AIONMonitoring()
    return _monitor_instance

def start_monitoring(interval=5):
    """Запустить мониторинг"""
    monitor = get_monitor()
    monitor.start_monitoring(interval)
    return monitor

def stop_monitoring():
    """Остановить мониторинг"""
    global _monitor_instance
    if _monitor_instance:
        _monitor_instance.stop_monitoring()

def log_event(event_type, message):
    """Логировать событие"""
    monitor = get_monitor()
    monitor.log_event(event_type, message)

def get_health_status():
    """Получить статус здоровья"""
    monitor = get_monitor()
    return monitor.get_health_report()

def show_monitoring_dashboard():
    """Показать дашборд мониторинга"""
    monitor = get_monitor()
    
    print("📊 AION MONITORING DASHBOARD")
    print("=" * 40)
    
    # Статус системы
    system_status = monitor.get_system_status()
    print(f"🖥️  СИСТЕМА:")
    print(f"   CPU: {system_status['cpu_percent']:.1f}%")
    print(f"   Память: {system_status['memory_percent']:.1f}%")
    print(f"   Диск: {system_status['disk_percent']:.1f}%")
    print(f"   Процессы: {system_status['processes']}")
    
    # Статистика AION
    aion_stats = monitor.get_aion_stats()
    print(f"\n⚡ AION:")
    print(f"   Время работы: {aion_stats['uptime_human']}")
    print(f"   Анализов: {aion_stats['analyses_completed']}")
    print(f"   AI запросов: {aion_stats['ai_requests']}")
    print(f"   Исправлений: {aion_stats['fixes_applied']}")
    print(f"   Ошибок: {aion_stats['errors_encountered']}")
    
    # Отчет о здоровье
    health = monitor.get_health_report()
    print(f"\n💚 ЗДОРОВЬЕ: {health['health_score']}/100 ({health['status']})")
    
    if health['warnings']:
        print("⚠️  ПРЕДУПРЕЖДЕНИЯ:")
        for warning in health['warnings']:
            print(f"   • {warning}")
    
    # Последние события
    recent_events = monitor.get_recent_events(5)
    if recent_events:
        print(f"\n📝 ПОСЛЕДНИЕ СОБЫТИЯ:")
        for event in recent_events:
            timestamp = datetime.fromisoformat(event['timestamp']).strftime("%H:%M:%S")
            print(f"   [{timestamp}] {event['type']}: {event['message']}")

if __name__ == "__main__":
    # Запуск мониторинга в интерактивном режиме
    try:
        monitor = start_monitoring(interval=2)
        print("🔍 Мониторинг запущен. Нажмите Ctrl+C для остановки")
        
        while True:
            time.sleep(10)
            os.system('cls' if os.name == 'nt' else 'clear')
            show_monitoring_dashboard()
            
    except KeyboardInterrupt:
        print("\n⏹️  Остановка мониторинга...")
        stop_monitoring()
        print("✅ Мониторинг остановлен")
