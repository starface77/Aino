#!/usr/bin/env python3
"""
Validators - Валидаторы для AION
"""

import re
from typing import Any, List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum

@dataclass
class ValidationResult:
    """Результат валидации"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def add_error(self, error: str):
        """Добавить ошибку"""
        self.is_valid = False
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Добавить предупреждение"""
        self.warnings.append(warning)

class ValidationSeverity(Enum):
    """Уровни серьезности валидации"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

def validate_problem(problem) -> ValidationResult:
    """
    Валидация объекта Problem
    
    Args:
        problem: Объект Problem для валидации
    
    Returns:
        ValidationResult: Результат валидации
    """
    
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    # Проверка обязательных полей
    if not hasattr(problem, 'id') or not problem.id:
        result.add_error("Problem.id не может быть пустым")
    
    if not hasattr(problem, 'description') or not problem.description:
        result.add_error("Problem.description не может быть пустым")
    
    if not hasattr(problem, 'type'):
        result.add_error("Problem.type обязателен")
    
    # Проверка длины описания
    if hasattr(problem, 'description') and problem.description:
        if len(problem.description) < 10:
            result.add_warning("Описание проблемы слишком короткое (< 10 символов)")
        elif len(problem.description) > 1000:
            result.add_warning("Описание проблемы очень длинное (> 1000 символов)")
    
    # Проверка формата ID
    if hasattr(problem, 'id') and problem.id:
        if not re.match(r'^[a-zA-Z0-9_-]+$', problem.id):
            result.add_error("Problem.id должен содержать только буквы, цифры, _ и -")
    
    # Проверка контекста
    if hasattr(problem, 'context') and problem.context:
        if not isinstance(problem.context, dict):
            result.add_error("Problem.context должен быть словарем")
    
    # Проверка ограничений
    if hasattr(problem, 'constraints') and problem.constraints:
        if not isinstance(problem.constraints, list):
            result.add_error("Problem.constraints должен быть списком")
    
    return result

def validate_solution(solution) -> ValidationResult:
    """
    Валидация объекта Solution
    
    Args:
        solution: Объект Solution для валидации
    
    Returns:
        ValidationResult: Результат валидации
    """
    
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    # Проверка обязательных полей
    if not hasattr(solution, 'problem_id') or not solution.problem_id:
        result.add_error("Solution.problem_id не может быть пустым")
    
    if not hasattr(solution, 'solution'):
        result.add_error("Solution.solution обязателен")
    
    if not hasattr(solution, 'confidence'):
        result.add_error("Solution.confidence обязателен")
    
    # Проверка диапазона уверенности
    if hasattr(solution, 'confidence'):
        if not isinstance(solution.confidence, (int, float)):
            result.add_error("Solution.confidence должен быть числом")
        elif solution.confidence < 0.0 or solution.confidence > 1.0:
            result.add_error("Solution.confidence должен быть в диапазоне [0.0, 1.0]")
        elif solution.confidence < 0.5:
            result.add_warning("Низкая уверенность в решении (< 50%)")
    
    # Проверка времени выполнения
    if hasattr(solution, 'execution_time'):
        if not isinstance(solution.execution_time, (int, float)):
            result.add_error("Solution.execution_time должен быть числом")
        elif solution.execution_time < 0:
            result.add_error("Solution.execution_time не может быть отрицательным")
        elif solution.execution_time > 300:  # 5 минут
            result.add_warning("Долгое время выполнения (> 5 минут)")
    
    # Проверка пути рассуждений
    if hasattr(solution, 'reasoning_path') and solution.reasoning_path:
        if not isinstance(solution.reasoning_path, list):
            result.add_error("Solution.reasoning_path должен быть списком")
        elif len(solution.reasoning_path) == 0:
            result.add_warning("Пустой путь рассуждений")
    
    return result

def validate_config(config) -> ValidationResult:
    """
    Валидация конфигурации AION
    
    Args:
        config: Объект конфигурации
    
    Returns:
        ValidationResult: Результат валидации
    """
    
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    # Проверка обязательных полей для production
    if hasattr(config, 'environment') and config.environment == 'production':
        
        if not hasattr(config, 'secret_key') or not config.secret_key:
            result.add_error("secret_key обязателен для production")
        
        if not hasattr(config, 'database_url') or not config.database_url:
            result.add_error("database_url обязателен для production")
    
    # Проверка числовых параметров
    numeric_fields = {
        'processing_speed_multiplier': (1.0, 10000.0),
        'accuracy_target': (0.0, 1.0),
        'parallel_threads': (1, 1000),
        'api_port': (1024, 65535),
        'hidden_size': (64, 8192),
        'num_heads': (1, 64),
        'num_layers': (1, 48)
    }
    
    for field, (min_val, max_val) in numeric_fields.items():
        if hasattr(config, field):
            value = getattr(config, field)
            if not isinstance(value, (int, float)):
                result.add_error(f"{field} должен быть числом")
            elif value < min_val or value > max_val:
                result.add_error(f"{field} должен быть в диапазоне [{min_val}, {max_val}]")
    
    # Проверка устройства
    if hasattr(config, 'device'):
        valid_devices = ['auto', 'cpu', 'cuda', 'mps']
        if config.device not in valid_devices:
            result.add_error(f"device должен быть одним из: {valid_devices}")
    
    # Проверка уровня логирования
    if hasattr(config, 'log_level'):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.log_level.upper() not in valid_levels:
            result.add_error(f"log_level должен быть одним из: {valid_levels}")
    
    return result

def validate_agent_config(agent_config) -> ValidationResult:
    """
    Валидация конфигурации агента
    
    Args:
        agent_config: Конфигурация агента
    
    Returns:
        ValidationResult: Результат валидации
    """
    
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    # Проверка имени агента
    if not hasattr(agent_config, 'name') or not agent_config.name:
        result.add_error("Имя агента обязательно")
    elif not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', agent_config.name):
        result.add_error("Имя агента должно начинаться с буквы и содержать только буквы, цифры и _")
    
    # Проверка лимитов
    if hasattr(agent_config, 'max_concurrent_tasks'):
        if not isinstance(agent_config.max_concurrent_tasks, int):
            result.add_error("max_concurrent_tasks должен быть целым числом")
        elif agent_config.max_concurrent_tasks < 1 or agent_config.max_concurrent_tasks > 100:
            result.add_error("max_concurrent_tasks должен быть в диапазоне [1, 100]")
    
    if hasattr(agent_config, 'timeout'):
        if not isinstance(agent_config.timeout, (int, float)):
            result.add_error("timeout должен быть числом")
        elif agent_config.timeout < 1 or agent_config.timeout > 3600:
            result.add_error("timeout должен быть в диапазоне [1, 3600] секунд")
    
    return result

def validate_task(task) -> ValidationResult:
    """
    Валидация задачи для агента
    
    Args:
        task: Объект Task
    
    Returns:
        ValidationResult: Результат валидации
    """
    
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    # Проверка обязательных полей
    required_fields = ['task_id', 'description', 'context', 'requirements', 'constraints']
    
    for field in required_fields:
        if not hasattr(task, field):
            result.add_error(f"Task.{field} обязателен")
    
    # Проверка типов
    if hasattr(task, 'context') and not isinstance(task.context, dict):
        result.add_error("Task.context должен быть словарем")
    
    if hasattr(task, 'requirements') and not isinstance(task.requirements, list):
        result.add_error("Task.requirements должен быть списком")
    
    if hasattr(task, 'constraints') and not isinstance(task.constraints, list):
        result.add_error("Task.constraints должен быть списком")
    
    # Проверка длины описания
    if hasattr(task, 'description') and task.description:
        if len(task.description) < 5:
            result.add_warning("Описание задачи слишком короткое")
        elif len(task.description) > 500:
            result.add_warning("Описание задачи очень длинное")
    
    return result

def validate_string(value: str, min_length: int = 0, max_length: int = None, 
                   pattern: str = None, field_name: str = "field") -> ValidationResult:
    """
    Валидация строкового значения
    
    Args:
        value: Значение для валидации
        min_length: Минимальная длина
        max_length: Максимальная длина  
        pattern: Регулярное выражение
        field_name: Имя поля для сообщений об ошибках
    
    Returns:
        ValidationResult: Результат валидации
    """
    
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    if not isinstance(value, str):
        result.add_error(f"{field_name} должен быть строкой")
        return result
    
    if len(value) < min_length:
        result.add_error(f"{field_name} должен быть не менее {min_length} символов")
    
    if max_length and len(value) > max_length:
        result.add_error(f"{field_name} должен быть не более {max_length} символов")
    
    if pattern and not re.match(pattern, value):
        result.add_error(f"{field_name} не соответствует требуемому формату")
    
    return result

def validate_number(value: Union[int, float], min_val: float = None, max_val: float = None,
                   field_name: str = "field") -> ValidationResult:
    """
    Валидация числового значения
    
    Args:
        value: Значение для валидации
        min_val: Минимальное значение
        max_val: Максимальное значение
        field_name: Имя поля для сообщений об ошибках
    
    Returns:
        ValidationResult: Результат валидации
    """
    
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    if not isinstance(value, (int, float)):
        result.add_error(f"{field_name} должен быть числом")
        return result
    
    if min_val is not None and value < min_val:
        result.add_error(f"{field_name} должен быть не менее {min_val}")
    
    if max_val is not None and value > max_val:
        result.add_error(f"{field_name} должен быть не более {max_val}")
    
    return result

def validate_email(email: str) -> ValidationResult:
    """Валидация email адреса"""
    
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        result.add_error("Неверный формат email адреса")
    
    return result

def validate_url(url: str) -> ValidationResult:
    """Валидация URL"""
    
    result = ValidationResult(is_valid=True, errors=[], warnings=[])
    
    url_pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*)?(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?$'
    
    if not re.match(url_pattern, url):
        result.add_error("Неверный формат URL")
    
    return result

class Validator:
    """Универсальный валидатор с цепочкой проверок"""
    
    def __init__(self):
        self.checks = []
    
    def add_check(self, check_func, *args, **kwargs):
        """Добавить проверку"""
        self.checks.append((check_func, args, kwargs))
        return self
    
    def validate(self, value: Any) -> ValidationResult:
        """Выполнить все проверки"""
        
        combined_result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        for check_func, args, kwargs in self.checks:
            result = check_func(value, *args, **kwargs)
            
            if not result.is_valid:
                combined_result.is_valid = False
            
            combined_result.errors.extend(result.errors)
            combined_result.warnings.extend(result.warnings)
        
        return combined_result

# Предустановленные валидаторы
def create_id_validator() -> Validator:
    """Создать валидатор для ID"""
    return (Validator()
            .add_check(validate_string, min_length=1, max_length=100, 
                      pattern=r'^[a-zA-Z0-9_-]+$', field_name="ID"))

def create_confidence_validator() -> Validator:
    """Создать валидатор для уверенности"""
    return (Validator()
            .add_check(validate_number, min_val=0.0, max_val=1.0, field_name="confidence"))

def create_execution_time_validator() -> Validator:
    """Создать валидатор для времени выполнения"""
    return (Validator()
            .add_check(validate_number, min_val=0.0, max_val=3600.0, field_name="execution_time"))

if __name__ == "__main__":
    # Демонстрация валидаторов
    
    print("🧪 Тестирование валидаторов AION")
    
    # Тест валидации строки
    string_result = validate_string("test_id", min_length=5, max_length=20, 
                                   pattern=r'^[a-zA-Z0-9_]+$', field_name="test_field")
    print(f"Валидация строки: {'✅' if string_result.is_valid else '❌'}")
    
    # Тест валидации числа
    number_result = validate_number(0.95, min_val=0.0, max_val=1.0, field_name="confidence")
    print(f"Валидация числа: {'✅' if number_result.is_valid else '❌'}")
    
    # Тест валидации email
    email_result = validate_email("test@example.com")
    print(f"Валидация email: {'✅' if email_result.is_valid else '❌'}")
    
    # Тест цепочки валидаторов
    validator = (Validator()
                .add_check(validate_string, min_length=1, max_length=50, field_name="test")
                .add_check(validate_number, min_val=0, max_val=100, field_name="test"))
    
    chain_result = validator.validate("test_value")
    print(f"Цепочка валидаторов: {'✅' if chain_result.is_valid else '❌'}")
    
    print("✅ Демонстрация валидаторов завершена")
