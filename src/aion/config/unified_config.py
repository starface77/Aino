#!/usr/bin/env python3
"""
UnifiedAIONConfig - Centralized configuration for all AION components
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import yaml
import logging

logger = logging.getLogger(__name__)

@dataclass
class UnifiedAIONConfig:
    """
    Unified configuration for all AION components
    """
    # Meta-Planning Configuration
    strategic_population_size: int = 1000
    tactical_evolution_generations: int = 500
    operational_episodes: int = 10000
    genetic_mutation_rate: float = 0.15
    neuroevolution_learning_rate: float = 0.001
    rl_discount_factor: float = 0.99
    quantum_entanglement_threshold: float = 0.85
    
    # Hierarchical Reasoning Configuration
    abstract_hidden_size: int = 512
    concrete_hidden_size: int = 256
    symbolic_hidden_size: int = 128
    quantum_attention_heads: int = 16
    temporal_window_size: int = 100
    cross_modal_fusion_dim: int = 256
    adaptive_learning_rate: float = 0.001
    
    # Memory System Configuration
    episodic_capacity: int = 1000000
    semantic_capacity: int = 500000
    procedural_capacity: int = 100000
    quantum_capacity: int = 10000
    holographic_dim: int = 1024
    associative_network_size: int = 10000
    pattern_recognition_threshold: float = 0.85
    
    # Bridge Configuration
    sync_interval: float = 0.1
    compression_ratio: float = 0.8
    validation_threshold: float = 0.95
    retry_attempts: int = 3
    timeout_seconds: float = 5.0
    max_queue_size: int = 10000
    
    # System Performance Configuration
    max_parallel_threads: int = 100
    processing_timeout: float = 30.0
    memory_limit_gb: float = 16.0
    gpu_memory_limit_gb: float = 8.0
    cache_size_mb: int = 1024
    
    # Security Configuration
    encryption_enabled: bool = True
    authentication_required: bool = True
    audit_logging: bool = True
    rate_limiting: bool = True
    max_requests_per_minute: int = 1000
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "aion.log"
    log_rotation: bool = True
    log_max_size_mb: int = 100
    
    # Database Configuration
    database_url: str = "sqlite:///aion.db"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    database_timeout: float = 30.0
    
    # Cache Configuration
    cache_type: str = "redis"
    cache_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 10000
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_timeout: float = 30.0
    cors_enabled: bool = True
    
    # Agent Configuration
    search_agent_enabled: bool = True
    code_agent_enabled: bool = True
    ecommerce_agent_enabled: bool = True
    logistics_agent_enabled: bool = True
    agent_timeout: float = 60.0
    
    # Optimization Configuration
    auto_optimization: bool = True
    optimization_interval: float = 300.0  # 5 minutes
    performance_threshold: float = 0.8
    resource_optimization: bool = True
    
    # Monitoring Configuration
    monitoring_enabled: bool = True
    metrics_collection: bool = True
    health_check_interval: float = 30.0
    alert_threshold: float = 0.7
    
    # Development Configuration
    debug_mode: bool = False
    development_mode: bool = False
    hot_reload: bool = False
    profiling_enabled: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'meta_planning': {
                'strategic_population_size': self.strategic_population_size,
                'tactical_evolution_generations': self.tactical_evolution_generations,
                'operational_episodes': self.operational_episodes,
                'genetic_mutation_rate': self.genetic_mutation_rate,
                'neuroevolution_learning_rate': self.neuroevolution_learning_rate,
                'rl_discount_factor': self.rl_discount_factor,
                'quantum_entanglement_threshold': self.quantum_entanglement_threshold
            },
            'reasoning': {
                'abstract_hidden_size': self.abstract_hidden_size,
                'concrete_hidden_size': self.concrete_hidden_size,
                'symbolic_hidden_size': self.symbolic_hidden_size,
                'quantum_attention_heads': self.quantum_attention_heads,
                'temporal_window_size': self.temporal_window_size,
                'cross_modal_fusion_dim': self.cross_modal_fusion_dim,
                'adaptive_learning_rate': self.adaptive_learning_rate
            },
            'memory': {
                'episodic_capacity': self.episodic_capacity,
                'semantic_capacity': self.semantic_capacity,
                'procedural_capacity': self.procedural_capacity,
                'quantum_capacity': self.quantum_capacity,
                'holographic_dim': self.holographic_dim,
                'associative_network_size': self.associative_network_size,
                'pattern_recognition_threshold': self.pattern_recognition_threshold
            },
            'bridge': {
                'sync_interval': self.sync_interval,
                'compression_ratio': self.compression_ratio,
                'validation_threshold': self.validation_threshold,
                'retry_attempts': self.retry_attempts,
                'timeout_seconds': self.timeout_seconds,
                'max_queue_size': self.max_queue_size
            },
            'system': {
                'max_parallel_threads': self.max_parallel_threads,
                'processing_timeout': self.processing_timeout,
                'memory_limit_gb': self.memory_limit_gb,
                'gpu_memory_limit_gb': self.gpu_memory_limit_gb,
                'cache_size_mb': self.cache_size_mb
            },
            'security': {
                'encryption_enabled': self.encryption_enabled,
                'authentication_required': self.authentication_required,
                'audit_logging': self.audit_logging,
                'rate_limiting': self.rate_limiting,
                'max_requests_per_minute': self.max_requests_per_minute
            },
            'logging': {
                'log_level': self.log_level,
                'log_file': self.log_file,
                'log_rotation': self.log_rotation,
                'log_max_size_mb': self.log_max_size_mb
            },
            'database': {
                'database_url': self.database_url,
                'database_pool_size': self.database_pool_size,
                'database_max_overflow': self.database_max_overflow,
                'database_timeout': self.database_timeout
            },
            'cache': {
                'cache_type': self.cache_type,
                'cache_url': self.cache_url,
                'cache_ttl_seconds': self.cache_ttl_seconds,
                'cache_max_size': self.cache_max_size
            },
            'api': {
                'api_host': self.api_host,
                'api_port': self.api_port,
                'api_workers': self.api_workers,
                'api_timeout': self.api_timeout,
                'cors_enabled': self.cors_enabled
            },
            'agents': {
                'search_agent_enabled': self.search_agent_enabled,
                'code_agent_enabled': self.code_agent_enabled,
                'ecommerce_agent_enabled': self.ecommerce_agent_enabled,
                'logistics_agent_enabled': self.logistics_agent_enabled,
                'agent_timeout': self.agent_timeout
            },
            'optimization': {
                'auto_optimization': self.auto_optimization,
                'optimization_interval': self.optimization_interval,
                'performance_threshold': self.performance_threshold,
                'resource_optimization': self.resource_optimization
            },
            'monitoring': {
                'monitoring_enabled': self.monitoring_enabled,
                'metrics_collection': self.metrics_collection,
                'health_check_interval': self.health_check_interval,
                'alert_threshold': self.alert_threshold
            },
            'development': {
                'debug_mode': self.debug_mode,
                'development_mode': self.development_mode,
                'hot_reload': self.hot_reload,
                'profiling_enabled': self.profiling_enabled
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UnifiedAIONConfig':
        """Create configuration from dictionary"""
        
        # Flatten nested dictionary
        flattened = {}
        for section, values in config_dict.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flattened[f"{section}_{key}"] = value
            else:
                flattened[section] = values
        
        return cls(**flattened)
    
    def save_to_file(self, filepath: str, format: str = 'json'):
        """Save configuration to file"""
        
        config_dict = self.to_dict()
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif format.lower() == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'UnifiedAIONConfig':
        """Load configuration from file"""
        
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        logger.info(f"Configuration loaded from {filepath}")
        return cls.from_dict(config_dict)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        
        errors = []
        
        # Validate numeric ranges
        if self.strategic_population_size <= 0:
            errors.append("strategic_population_size must be positive")
        
        if self.genetic_mutation_rate < 0 or self.genetic_mutation_rate > 1:
            errors.append("genetic_mutation_rate must be between 0 and 1")
        
        if self.validation_threshold < 0 or self.validation_threshold > 1:
            errors.append("validation_threshold must be between 0 and 1")
        
        if self.memory_limit_gb <= 0:
            errors.append("memory_limit_gb must be positive")
        
        if self.api_port < 1 or self.api_port > 65535:
            errors.append("api_port must be between 1 and 65535")
        
        # Validate logical constraints
        if self.cache_size_mb > self.memory_limit_gb * 1024:
            errors.append("cache_size_mb cannot exceed memory_limit_gb * 1024")
        
        if self.gpu_memory_limit_gb > self.memory_limit_gb:
            errors.append("gpu_memory_limit_gb cannot exceed memory_limit_gb")
        
        return errors
    
    def optimize_for_performance(self) -> 'UnifiedAIONConfig':
        """Optimize configuration for maximum performance"""
        
        # Increase population sizes for better optimization
        self.strategic_population_size = 2000
        self.tactical_evolution_generations = 1000
        self.operational_episodes = 20000
        
        # Increase memory capacities
        self.episodic_capacity = 2000000
        self.semantic_capacity = 1000000
        self.procedural_capacity = 200000
        
        # Optimize neural network sizes
        self.abstract_hidden_size = 1024
        self.concrete_hidden_size = 512
        self.symbolic_hidden_size = 256
        
        # Increase parallel processing
        self.max_parallel_threads = 200
        self.api_workers = 8
        
        # Optimize caching
        self.cache_size_mb = 2048
        self.cache_ttl_seconds = 7200
        
        logger.info("Configuration optimized for maximum performance")
        return self
    
    def optimize_for_accuracy(self) -> 'UnifiedAIONConfig':
        """Optimize configuration for maximum accuracy"""
        
        # Increase learning rates for better convergence
        self.neuroevolution_learning_rate = 0.0005
        self.adaptive_learning_rate = 0.0005
        
        # Increase validation thresholds
        self.validation_threshold = 0.98
        self.pattern_recognition_threshold = 0.9
        
        # Increase attention heads for better focus
        self.quantum_attention_heads = 32
        
        # Increase temporal window for better context
        self.temporal_window_size = 200
        
        # Increase retry attempts for reliability
        self.retry_attempts = 5
        
        logger.info("Configuration optimized for maximum accuracy")
        return self
    
    def get_component_config(self, component_name: str) -> Dict[str, Any]:
        """Get configuration for specific component"""
        
        config_dict = self.to_dict()
        
        if component_name in config_dict:
            return config_dict[component_name]
        else:
            raise ValueError(f"Unknown component: {component_name}")
    
    def update_component_config(self, component_name: str, updates: Dict[str, Any]):
        """Update configuration for specific component"""
        
        config_dict = self.to_dict()
        
        if component_name not in config_dict:
            raise ValueError(f"Unknown component: {component_name}")
        
        # Update the component configuration
        config_dict[component_name].update(updates)
        
        # Recreate the configuration object
        new_config = self.from_dict(config_dict)
        
        # Copy all attributes
        for attr, value in new_config.__dict__.items():
            setattr(self, attr, value)
        
        logger.info(f"Updated configuration for component: {component_name}")

# Default configurations
def get_default_config() -> UnifiedAIONConfig:
    """Get default configuration"""
    return UnifiedAIONConfig()

def get_performance_config() -> UnifiedAIONConfig:
    """Get performance-optimized configuration"""
    return UnifiedAIONConfig().optimize_for_performance()

def get_accuracy_config() -> UnifiedAIONConfig:
    """Get accuracy-optimized configuration"""
    return UnifiedAIONConfig().optimize_for_accuracy()

def get_development_config() -> UnifiedAIONConfig:
    """Get development configuration"""
    config = UnifiedAIONConfig()
    config.debug_mode = True
    config.development_mode = True
    config.hot_reload = True
    config.profiling_enabled = True
    config.log_level = "DEBUG"
    return config

# Example usage
if __name__ == "__main__":
    # Create default configuration
    config = get_default_config()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid")
    
    # Save configuration
    config.save_to_file("aion_config.json")
    
    # Load configuration
    loaded_config = UnifiedAIONConfig.load_from_file("aion_config.json")
    
    # Get component configuration
    memory_config = config.get_component_config("memory")
    print(f"Memory configuration: {memory_config}")
    
    # Optimize for performance
    perf_config = get_performance_config()
    print(f"Performance config - population size: {perf_config.strategic_population_size}")
    
    # Optimize for accuracy
    acc_config = get_accuracy_config()
    print(f"Accuracy config - validation threshold: {acc_config.validation_threshold}")
