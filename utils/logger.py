#!/usr/bin/env python3
"""
Logger - Система логирования AION
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

class ColoredFormatter(logging.Formatter):
    """Цветной форматтер для консольного вывода"""
    
    # Цветовые коды ANSI
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Добавляем цвет к уровню логирования
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class AIONLogger:
    """
    Система логирования AION с поддержкой:
    - Цветного вывода
    - Структурированного логирования
    - Множественных обработчиков
    - Фильтрации по модулям
    """
    
    def __init__(self, name: str = "AION", config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
        self.setup_logging()
    
    def setup_logging(self):
        """Настройка логирования"""
        
        # Очищаем существующие обработчики
        self.logger.handlers.clear()
        
        # Устанавливаем уровень логирования
        log_level = self.config.get('log_level', 'INFO')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Консольный обработчик
        if self.config.get('enable_console_logging', True):
            self._setup_console_handler()
        
        # Файловый обработчик
        if self.config.get('enable_file_logging', True):
            self._setup_file_handler()
        
        # Предотвращаем дублирование сообщений
        self.logger.propagate = False
    
    def _setup_console_handler(self):
        """Настройка консольного обработчика"""
        
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Цветной форматтер для консоли
        console_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        console_formatter = ColoredFormatter(console_format)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Настройка файлового обработчика"""
        
        # Создаем папку для логов
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Файл с датой
        log_file = self.config.get('log_file')
        if not log_file:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = log_dir / f"aion_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        
        # Детальный форматтер для файла
        file_format = "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Debug сообщение"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Info сообщение"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning сообщение"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Error сообщение"""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Critical сообщение"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def superhuman(self, message: str, **kwargs):
        """Специальный уровень для сверхчеловеческих достижений"""
        enhanced_message = f"🏆 SUPERHUMAN: {message}"
        self._log_with_context(logging.INFO, enhanced_message, **kwargs)
    
    def performance(self, operation: str, execution_time: float, **kwargs):
        """Логирование производительности"""
        message = f"⚡ PERFORMANCE: {operation} completed in {execution_time:.6f}s"
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def task_completed(self, task_id: str, confidence: float, execution_time: float, **kwargs):
        """Логирование завершения задачи"""
        message = f"✅ TASK: {task_id} | Confidence: {confidence:.1%} | Time: {execution_time:.6f}s"
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Логирование с контекстом"""
        
        # Добавляем контекстную информацию
        if kwargs:
            context_str = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {context_str}"
        
        self.logger.log(level, message)

# Глобальные логгеры для разных компонентов
_loggers: Dict[str, AIONLogger] = {}

def setup_logging(config: Dict[str, Any] = None) -> AIONLogger:
    """
    Настройка глобального логирования AION
    
    Args:
        config: Конфигурация логирования
    
    Returns:
        AIONLogger: Главный логгер
    """
    
    main_logger = AIONLogger("AION", config)
    _loggers["AION"] = main_logger
    
    # Создаем специализированные логгеры
    component_configs = {
        "AION.Core": config,
        "AION.Engine": config,
        "AION.Intelligence": config,
        "AION.Reasoning": config,
        "AION.Agents": config,
        "AION.Utils": config
    }
    
    for component, component_config in component_configs.items():
        _loggers[component] = AIONLogger(component, component_config)
    
    main_logger.info("🚀 AION Logging System initialized")
    return main_logger

def get_logger(name: str = "AION") -> AIONLogger:
    """
    Получение логгера по имени
    
    Args:
        name: Имя логгера
    
    Returns:
        AIONLogger: Логгер
    """
    
    if name not in _loggers:
        # Создаем логгер с базовой конфигурацией
        _loggers[name] = AIONLogger(name)
    
    return _loggers[name]

def log_system_info():
    """Логирование информации о системе"""
    
    logger = get_logger("AION.System")
    
    logger.info("📋 System Information:")
    logger.info(f"   🐍 Python: {sys.version}")
    logger.info(f"   💻 Platform: {sys.platform}")
    logger.info(f"   📁 Working Directory: {os.getcwd()}")
    logger.info(f"   🕐 Timestamp: {datetime.now().isoformat()}")

def log_config(config: Dict[str, Any]):
    """Логирование конфигурации"""
    
    logger = get_logger("AION.Config")
    
    logger.info("⚙️ Configuration loaded:")
    
    # Безопасное логирование конфигурации (скрываем секреты)
    safe_config = {}
    sensitive_keys = ['password', 'secret', 'key', 'token']
    
    for key, value in config.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            safe_config[key] = "***HIDDEN***"
        else:
            safe_config[key] = value
    
    logger.info(f"   📋 Config: {json.dumps(safe_config, indent=2, default=str)}")

class LoggingMixin:
    """Миксин для добавления логирования в классы"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(f"AION.{self.__class__.__name__}")
    
    def log_debug(self, message: str, **kwargs):
        """Debug логирование"""
        self.logger.debug(f"[{self.__class__.__name__}] {message}", **kwargs)
    
    def log_info(self, message: str, **kwargs):
        """Info логирование"""
        self.logger.info(f"[{self.__class__.__name__}] {message}", **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Warning логирование"""
        self.logger.warning(f"[{self.__class__.__name__}] {message}", **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Error логирование"""
        self.logger.error(f"[{self.__class__.__name__}] {message}", **kwargs)
    
    def log_performance(self, operation: str, execution_time: float, **kwargs):
        """Логирование производительности"""
        self.logger.performance(f"[{self.__class__.__name__}] {operation}", execution_time, **kwargs)
    
    def log_superhuman(self, message: str, **kwargs):
        """Логирование сверхчеловеческих достижений"""
        self.logger.superhuman(f"[{self.__class__.__name__}] {message}", **kwargs)

# Декоратор для логирования выполнения функций
def log_execution(logger_name: str = "AION"):
    """
    Декоратор для автоматического логирования выполнения функций
    
    Args:
        logger_name: Имя логгера
    """
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start_time = datetime.now()
            
            logger.debug(f"🚀 Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.performance(f"{func.__name__}", execution_time)
                return result
            
            except Exception as e:
                logger.error(f"❌ Error in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator

if __name__ == "__main__":
    # Демонстрация системы логирования
    
    config = {
        'log_level': 'DEBUG',
        'enable_console_logging': True,
        'enable_file_logging': True
    }
    
    # Настройка логирования
    main_logger = setup_logging(config)
    
    # Тестирование различных уровней
    main_logger.debug("🔍 Debug message")
    main_logger.info("ℹ️ Info message")
    main_logger.warning("⚠️ Warning message")
    main_logger.error("❌ Error message")
    main_logger.superhuman("🏆 Superhuman achievement!")
    main_logger.performance("test_operation", 0.001234)
    main_logger.task_completed("test_001", 0.95, 0.00567)
    
    # Тестирование специализированных логгеров
    core_logger = get_logger("AION.Core")
    core_logger.info("Core component message")
    
    agent_logger = get_logger("AION.Agents")
    agent_logger.info("Agent component message")
    
    # Логирование системной информации
    log_system_info()
    
    print("✅ Демонстрация логирования завершена")
    print("📁 Проверьте файл logs/aion_*.log для файлового вывода")
