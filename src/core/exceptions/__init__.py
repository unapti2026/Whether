"""
Exceptions Module

This module contains custom exception classes for the weather prediction system.
These exceptions provide specific error handling for different scenarios.
"""

from .data_exceptions import DataProcessingError
from .processing_exceptions import ImputationError, ValidationError

__all__ = [
    'DataProcessingError',
    'ImputationError',
    'ValidationError'
] 