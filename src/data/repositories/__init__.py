"""
Data Repositories Module

This module contains repository classes for data access and persistence.
These repositories provide abstracted access to data sources and files.
"""

from .data_repository import DataRepository
from .file_repository import FileRepository

__all__ = [
    'DataRepository',
    'FileRepository'
] 