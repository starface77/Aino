#!/usr/bin/env python3
"""
Configuration System - Система конфигурации AION
"""

import os
import yaml
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AIONConfig:
    """Главная конфигурация AION"""
    
    # Основные параметры
    version: str = "1.0.0"
    environment: str = "development"  # development, testing, production
    debug: bool = True
    
    # Производительность
    processing_speed_multiplier: float = 1000.0
    memory_capacity: int = 1_000_000
    parallel_threads: int = 100
    accuracy_target: float = 0.999
    
    # Модель
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    max_seq_length: int = 8192
    vocab_size: int = 50000
    
    # Система
    device: str = "auto"
    enable_gpu: bool = True
    enable_cuda: bool = True
    
    # Логирование
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    enable_cors: bool = True
    api_timeout: int = 300
    
    # База данных
    database_url: Optional[str] = None
    db_pool_size: int = 10
    db_max_overflow: int = 20
    
    # Redis/Cache
    redis_url: Optional[str] = None
    cache_ttl: int = 3600
    enable_caching: bool = True
    
    # Безопасность
    secret_key: Optional[str] = None
    enable_authentication: bool = False
    jwt_expiration: int = 3600
    
    # Агенты
    enable_search_agent: bool = True
    enable_code_agent: bool = True
    enable_ecommerce_agent: bool = True
    enable_logistics_agent: bool = True
    
    # Superhuman Intelligence
    superhuman_mode: bool = True
    creativity_level: float = 0.95
    pattern_recognition_level: float = 0.99
    emotional_intelligence_level: float = 0.92
    
    # Мониторинг
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_prometheus: bool = False
    
    # Экспериментальные функции
    enable_quantum_computing: bool = False
    enable_consciousness_simulation: bool = False
    enable_time_travel_reasoning: bool = False

@dataclass 
class DatabaseConfig:
    """Конфигурация базы данных"""
    driver: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "aion"
    username: str = "aion_user"
    password: str = "secure_password"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False

@dataclass
class RedisConfig:
    """Конфигурация Redis"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    max_connections: int = 50

@dataclass
class AgentConfig:
    """Конфигурация агента"""
    name: str
    enabled: bool = True
    max_concurrent_tasks: int = 10
    timeout: int = 300
    retry_attempts: int = 3
    superhuman_enhancements: bool = True
    custom_settings: Dict[str, Any] = field(default_factory=dict)

class Config:
    """
    Менеджер конфигурации AION
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.aion_config = AIONConfig()
        self.database_config = DatabaseConfig()
        self.redis_config = RedisConfig()
        self.agent_configs = {}
        
        # Автоматическая загрузка конфигурации
        if config_path:
            self.load_from_file(config_path)
        else:
            self.load_from_environment()
    
    def load_from_file(self, config_path: str):
        """Загрузка конфигурации из файла"""
        
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"Конфигурационный файл {config_path} не найден, используем значения по умолчанию")
            return
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Неподдерживаемый формат файла: {config_file.suffix}")
            
            self._update_config_from_dict(config_data)
            logger.info(f"✅ Конфигурация загружена из {config_path}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки конфигурации из {config_path}: {e}")
            logger.info("Используем конфигурацию по умолчанию")
    
    def load_from_environment(self):
        """Загрузка конфигурации из переменных окружения"""
        
        env_mappings = {
            'AION_ENVIRONMENT': 'environment',
            'AION_DEBUG': 'debug',
            'AION_LOG_LEVEL': 'log_level',
            'AION_API_HOST': 'api_host',
            'AION_API_PORT': 'api_port',
            'AION_DATABASE_URL': 'database_url',
            'AION_REDIS_URL': 'redis_url',
            'AION_SECRET_KEY': 'secret_key',
            'AION_ENABLE_GPU': 'enable_gpu',
            'AION_SUPERHUMAN_MODE': 'superhuman_mode'
        }
        
        updated_count = 0
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Конвертация типов
                if config_key in ['debug', 'enable_gpu', 'superhuman_mode']:
                    value = value.lower() in ['true', '1', 'yes', 'on']
                elif config_key == 'api_port':
                    value = int(value)
                
                setattr(self.aion_config, config_key, value)
                updated_count += 1
        
        if updated_count > 0:
            logger.info(f"✅ Обновлено {updated_count} параметров из переменных окружения")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Обновление конфигурации из словаря"""
        
        # Обновление основной конфигурации
        if 'aion' in config_data:
            aion_data = config_data['aion']
            for key, value in aion_data.items():
                if hasattr(self.aion_config, key):
                    setattr(self.aion_config, key, value)
        
        # Обновление конфигурации базы данных
        if 'database' in config_data:
            db_data = config_data['database']
            for key, value in db_data.items():
                if hasattr(self.database_config, key):
                    setattr(self.database_config, key, value)
        
        # Обновление конфигурации Redis
        if 'redis' in config_data:
            redis_data = config_data['redis']
            for key, value in redis_data.items():
                if hasattr(self.redis_config, key):
                    setattr(self.redis_config, key, value)
        
        # Обновление конфигурации агентов
        if 'agents' in config_data:
            agents_data = config_data['agents']
            for agent_name, agent_data in agents_data.items():
                self.agent_configs[agent_name] = AgentConfig(
                    name=agent_name,
                    **agent_data
                )
    
    def save_to_file(self, config_path: str, format: str = 'yaml'):
        """Сохранение конфигурации в файл"""
        
        config_data = {
            'aion': self._dataclass_to_dict(self.aion_config),
            'database': self._dataclass_to_dict(self.database_config),
            'redis': self._dataclass_to_dict(self.redis_config),
            'agents': {name: self._dataclass_to_dict(config) 
                      for name, config in self.agent_configs.items()}
        }
        
        try:
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                if format.lower() in ['yml', 'yaml']:
                    yaml.safe_dump(config_data, f, default_flow_style=False, 
                                  allow_unicode=True, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Неподдерживаемый формат: {format}")
            
            logger.info(f"✅ Конфигурация сохранена в {config_path}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения конфигурации: {e}")
    
    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Конвертация dataclass в словарь"""
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith('_'):
                result[key] = value
        return result
    
    def get_database_url(self) -> str:
        """Получение URL базы данных"""
        if self.aion_config.database_url:
            return self.aion_config.database_url
        
        db = self.database_config
        return f"{db.driver}://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}"
    
    def get_redis_url(self) -> str:
        """Получение URL Redis"""
        if self.aion_config.redis_url:
            return self.aion_config.redis_url
        
        redis = self.redis_config
        auth_part = f":{redis.password}@" if redis.password else ""
        return f"redis://{auth_part}{redis.host}:{redis.port}/{redis.database}"
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Получение конфигурации агента"""
        return self.agent_configs.get(agent_name)
    
    def is_production(self) -> bool:
        """Проверка production окружения"""
        return self.aion_config.environment == 'production'
    
    def is_development(self) -> bool:
        """Проверка development окружения"""
        return self.aion_config.environment == 'development'
    
    def get_device(self) -> str:
        """Получение устройства для вычислений"""
        if self.aion_config.device == "auto":
            if self.aion_config.enable_gpu:
                try:
                    import torch
                    return "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    return "cpu"
            else:
                return "cpu"
        return self.aion_config.device
    
    def validate(self) -> List[str]:
        """Валидация конфигурации"""
        errors = []
        
        # Проверка обязательных параметров для production
        if self.is_production():
            if not self.aion_config.secret_key:
                errors.append("SECRET_KEY обязателен для production")
            
            if not self.aion_config.database_url and not all([
                self.database_config.host,
                self.database_config.database,
                self.database_config.username,
                self.database_config.password
            ]):
                errors.append("Настройки базы данных обязательны для production")
        
        # Проверка диапазонов значений
        if not 0.0 <= self.aion_config.accuracy_target <= 1.0:
            errors.append("accuracy_target должен быть в диапазоне [0.0, 1.0]")
        
        if self.aion_config.api_port < 1024 or self.aion_config.api_port > 65535:
            errors.append("api_port должен быть в диапазоне [1024, 65535]")
        
        return errors
    
    def __str__(self) -> str:
        """Строковое представление конфигурации"""
        return f"""
AION Configuration:
├── Environment: {self.aion_config.environment}
├── Version: {self.aion_config.version}
├── Debug: {self.aion_config.debug}
├── Device: {self.get_device()}
├── API: {self.aion_config.api_host}:{self.aion_config.api_port}
├── Database: {self.get_database_url()}
├── Redis: {self.get_redis_url()}
├── Superhuman Mode: {self.aion_config.superhuman_mode}
└── Agents: {len(self.agent_configs)} configured
"""

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Загрузка конфигурации AION
    
    Args:
        config_path: Путь к файлу конфигурации (опционально)
    
    Returns:
        Config: Объект конфигурации
    """
    
    # Поиск конфигурационного файла
    if not config_path:
        search_paths = [
            "config/aion.yaml",
            "config/aion.yml", 
            "aion.yaml",
            "aion.yml",
            "config.yaml",
            "config.yml"
        ]
        
        for path in search_paths:
            if Path(path).exists():
                config_path = path
                break
    
    config = Config(config_path)
    
    # Валидация конфигурации
    errors = config.validate()
    if errors:
        logger.warning("⚠️ Найдены ошибки в конфигурации:")
        for error in errors:
            logger.warning(f"   - {error}")
    
    logger.info("✅ Конфигурация AION загружена")
    logger.debug(str(config))
    
    return config

def create_default_config(config_path: str):
    """Создание конфигурационного файла по умолчанию"""
    
    config = Config()
    config.save_to_file(config_path)
    
    logger.info(f"✅ Создан файл конфигурации по умолчанию: {config_path}")

if __name__ == "__main__":
    # Демонстрация работы с конфигурацией
    
    # Создание конфигурации по умолчанию
    config = load_config()
    
    print("📋 Загруженная конфигурация:")
    print(config)
    
    # Сохранение конфигурации
    config.save_to_file("example_config.yaml")
    
    print("✅ Пример конфигурации сохранен в example_config.yaml")
