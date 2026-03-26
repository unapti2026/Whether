"""
Core Interfaces Module

This module contains abstract interfaces that define contracts
for different components of the system.
"""

from .data_processor import DataProcessorInterface
from .visualization_strategy import VisualizationStrategyInterface

__all__ = [
    'DataProcessorInterface',
    'VisualizationStrategyInterface'
] 