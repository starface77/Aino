#!/usr/bin/env python3
"""
TENDO Marketplace - Intelligent product management system
"""

# Import from existing core module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'core'))

from intelligent_product_manager import IntelligentProductManagementSystem

__all__ = ['IntelligentProductManagementSystem']
