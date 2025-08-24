#!/usr/bin/env python3
"""
TENDO - AI-driven E-commerce Platform
"""

__version__ = "1.0.0"
__author__ = "TENDO Team"

# Import main components
from .marketplace import IntelligentProductManagementSystem
from .recommendations import AdaptiveRecommendationEngine
from .integration import TendoAionIntegration
from .api import TendoAPI

__all__ = [
    'IntelligentProductManagementSystem',
    'AdaptiveRecommendationEngine', 
    'TendoAionIntegration',
    'TendoAPI'
]
