"""
Pytest Configuration and Fixtures

This module contains pytest configuration, fixtures, and test utilities
for the weather prediction system.
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
import sys

# Add src to Python path for imports
sys.path.append('src')

from src.data.processors.meteorological_processor import MeteorologicalDataProcessor
from src.data.visualizers.meteorological_plotter import MeteorologicalPlotter
from src.data.services.preprocessing_service import PreprocessingService
from src.config.settings import get_config_for_variable, get_plot_config


@pytest.fixture
def sample_meteorological_data() -> pd.DataFrame:
    """
    Create sample meteorological data for testing.
    
    Returns:
        DataFrame with sample meteorological data in monthly format
    """
    data = {
        'Año': [2020, 2020, 2020, 2020],
        'Mes': [1, 1, 2, 2],
        'Estación': ['Test Station', 'Test Station', 'Test Station', 'Test Station'],
        'Código': [123, 123, 123, 123],
        'Día01': [25.5, 26.0, 24.5, 25.0],
        'Día02': [26.0, 26.5, 25.0, 25.5],
        'Día03': [25.8, 26.2, 24.8, 25.2],
        'Día04': [26.2, 26.8, 25.2, 25.8],
        'Día05': [25.9, 26.3, 24.9, 25.3],
        'Día06': [26.1, 26.6, 25.1, 25.6],
        'Día07': [25.7, 26.1, 24.7, 25.1],
        'Día08': [26.3, 26.9, 25.3, 25.9],
        'Día09': [25.6, 26.0, 24.6, 25.0],
        'Día10': [26.0, 26.5, 25.0, 25.5],
        'Día11': [25.8, 26.2, 24.8, 25.2],
        'Día12': [26.2, 26.8, 25.2, 25.8],
        'Día13': [25.9, 26.3, 24.9, 25.3],
        'Día14': [26.1, 26.6, 25.1, 25.6],
        'Día15': [25.7, 26.1, 24.7, 25.1],
        'Día16': [26.3, 26.9, 25.3, 25.9],
        'Día17': [25.6, 26.0, 24.6, 25.0],
        'Día18': [26.0, 26.5, 25.0, 25.5],
        'Día19': [25.8, 26.2, 24.8, 25.2],
        'Día20': [26.2, 26.8, 25.2, 25.8],
        'Día21': [25.9, 26.3, 24.9, 25.3],
        'Día22': [26.1, 26.6, 25.1, 25.6],
        'Día23': [25.7, 26.1, 24.7, 25.1],
        'Día24': [26.3, 26.9, 25.3, 25.9],
        'Día25': [25.6, 26.0, 24.6, 25.0],
        'Día26': [26.0, 26.5, 25.0, 25.5],
        'Día27': [25.8, 26.2, 24.8, 25.2],
        'Día28': [26.2, 26.8, 25.2, 25.8],
        'Día29': [25.9, 26.3, 24.9, 25.3],
        'Día30': [26.1, 26.6, 25.1, 25.6],
        'Día31': [25.7, 26.1, 24.7, 25.1]
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_processed_data() -> pd.DataFrame:
    """
    Create sample processed time series data for testing.
    
    Returns:
        DataFrame with sample processed time series data
    """
    data = {
        'Código': [123, 123, 123, 123, 123, 123],
        'Estación': ['Test Station'] * 6,
        'Fecha': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', 
                                '2020-02-01', '2020-02-02', '2020-02-03']),
        'Temperatura': [25.5, 26.0, 25.8, 24.5, 25.0, 24.8]
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_meteorological_data) -> str:
    """
    Create a temporary CSV file with sample data for testing.
    
    Args:
        sample_meteorological_data: Sample data to write to file
        
    Returns:
        Path to the temporary CSV file
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_meteorological_data.to_csv(f.name, index=False)
        return f.name


@pytest.fixture
def temp_output_dir() -> str:
    """
    Create a temporary output directory for testing.
    
    Returns:
        Path to the temporary output directory
    """
    temp_dir = tempfile.mkdtemp()
    return temp_dir


@pytest.fixture
def meteorological_config():
    """Fixture providing meteorological data configuration."""
    return get_config_for_variable('temp_min')


@pytest.fixture
def plot_config() -> Dict[str, Any]:
    """
    Get configuration for plotting.
    
    Returns:
        Configuration dictionary for plotting
    """
    return get_plot_config('default')


@pytest.fixture
def meteorological_processor(temp_csv_file, meteorological_config) -> MeteorologicalDataProcessor:
    """
    Create a MeteorologicalDataProcessor instance for testing.
    
    Args:
        temp_csv_file: Path to temporary CSV file
        meteorological_config: Configuration for the processor
        
    Returns:
        Configured MeteorologicalDataProcessor instance
    """
    return MeteorologicalDataProcessor(temp_csv_file, meteorological_config)


@pytest.fixture
def meteorological_plotter(plot_config) -> MeteorologicalPlotter:
    """
    Create a MeteorologicalPlotter instance for testing.
    
    Args:
        plot_config: Configuration for the plotter
        
    Returns:
        Configured MeteorologicalPlotter instance
    """
    return MeteorologicalPlotter(plot_config)


@pytest.fixture
def preprocessing_service(temp_csv_file) -> PreprocessingService:
    """
    Create a PreprocessingService instance for testing.
    
    Args:
        temp_csv_file: Path to temporary CSV file
        
    Returns:
        Configured PreprocessingService instance
    """
    return PreprocessingService(temp_csv_file, variable_type='temperature')


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_temp_files(temp_csv_file, temp_output_dir):
    """
    Clean up temporary files after tests.
    
    Args:
        temp_csv_file: Path to temporary CSV file
        temp_output_dir: Path to temporary output directory
    """
    yield
    # Cleanup after test
    if os.path.exists(temp_csv_file):
        os.unlink(temp_csv_file)
    
    if os.path.exists(temp_output_dir):
        import shutil
        shutil.rmtree(temp_output_dir) 