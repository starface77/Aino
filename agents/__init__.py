"""
AION Agents - Skill Library
Каталог агентов-навыков для различных задач
"""

from .base_agent import BaseAgent, AgentCapability
from .search_agent import SearchAgent
from .code_agent import CodeAgent
from .ecommerce_agent import EcommerceAgent
from .logistics_agent import LogisticsAgent

__all__ = [
    'BaseAgent',
    'AgentCapability',
    'SearchAgent',
    'CodeAgent', 
    'EcommerceAgent',
    'LogisticsAgent'
]
