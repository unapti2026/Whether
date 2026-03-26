"""
Settings Configuration Module

This module contains all configuration settings for the weather prediction system,
integrating modular configurations for paths, logging, and validation rules.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import yaml
import pandas as pd

# Import modular configurations
from .paths import (
    PROJECT_ROOT, DATA_DIR, STATIC_DIR, OUTPUT_DIR,
    get_file_paths_for_variable, get_project_structure
)
from .logging_config import setup_logging, get_logger
from .validation_rules import (
    DATA_VALIDATION_RULES, CONFIG_VALIDATION_RULES,
    validate_dataframe_structure, validate_data_values,
    validate_configuration, get_validation_rules
)
from .column_config import (
    ColumnMapping, ColumnType, ColumnDetector, ColumnConfigManager,
    get_column_mapping, get_column_name, validate_data_structure
)

# Meteorological variables configuration
METEOROLOGICAL_VARIABLES = {
    'temp_min': {
        'display_name': 'Temperatura Mínima',
        'file_name': 'temp_min.xlsx',
        'value_column': 'Temperatura',
        'unit': '°C',
        'validation_rules': DATA_VALIDATION_RULES['temp_min'],
        'plot_config': {
            'title': 'Temperatura Mínima',
            'ylabel': 'Temperatura (°C)',
            'color': '#1f77b4'
        }
    },
    'temp_max': {
        'display_name': 'Temperatura Máxima',
        'file_name': 'temp_max.xlsx',
        'value_column': 'Temperatura',
        'unit': '°C',
        'validation_rules': DATA_VALIDATION_RULES['temp_max'],
        'plot_config': {
            'title': 'Temperatura Máxima',
            'ylabel': 'Temperatura (°C)',
            'color': '#ff7f0e'
        }
    },
    'precipitation': {
        'display_name': 'Precipitación',
        'file_name': 'precipitation.xlsx',
        'value_column': 'Precipitación',
        'unit': 'mm',
        'validation_rules': DATA_VALIDATION_RULES['precipitation'],
        'plot_config': {
            'title': 'Precipitación',
            'ylabel': 'Precipitación (mm)',
            'color': '#2ca02c'
        }
    },
    'humidity': {
        'display_name': 'Humedad Relativa',
        'file_name': 'humidity.xlsx',
        'value_column': 'Humedad',
        'unit': '%',
        'validation_rules': DATA_VALIDATION_RULES['humidity'],
        'plot_config': {
            'title': 'Humedad Relativa',
            'ylabel': 'Humedad (%)',
            'color': '#d62728'
        }
    }
}

# Base configuration for all meteorological data
BASE_METEOROLOGICAL_CONFIG = {
    'year_column': 'Año',
    'month_column': 'Mes',
    'station_column': 'Estación',
    'code_column': 'Código',
    'date_column': 'Fecha',
    'day_prefix': 'Día',
    'max_days': 31,
    'clean_values': ['*', 'nan', 'NaN', ''],
    'data_frequency': 'D',  # Default to daily frequency
    'auto_detect_frequency': True,  # Automatically detect frequency from data
    'frequency_validation': True,  # Validate frequency consistency
    'auto_detect_structure': True,  # Automatically detect data structure
    'structure_validation': True,  # Validate structure consistency
    'structure_confidence_threshold': 0.7  # Minimum confidence for structure detection
}

# Plotting configuration
DEFAULT_PLOT_CONFIG = {
    'figure_size': (12, 5),
    'font_family': 'serif',
    'font_size': 10,
    'title_size': 14,
    'label_size': 12,
    'line_width': 0.5,
    'original_data_color': 'k',
    'forecasted_data_color': 'r',
    'grid_enabled': False
}

# Processing pipeline configuration
PROCESSING_PIPELINE = {
    'steps': [
        'load_data',
        'clean_data', 
        'validate_data',
        'restructure_data',
        'save_processed_data'
    ],
    'validation_enabled': True,
    'save_intermediate': False
}

# Initialize logging
setup_logging()
logger = get_logger(__name__)

def get_config_for_variable(variable_type: str, frequency: Optional[str] = None, 
                          structure_config: Optional[Dict[str, Any]] = None,
                          data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific meteorological variable.
    
    Args:
        variable_type: Type of variable ('temp_min', 'temp_max', 'precipitation', 'humidity')
        frequency: Optional frequency override ('H', 'D', 'W', 'M', 'Q', 'Y')
        structure_config: Optional structure configuration override
        data: Optional DataFrame for dynamic column detection
        
    Returns:
        Configuration dictionary for the specified variable
        
    Raises:
        ValueError: If variable type is not supported
    """
    if variable_type not in METEOROLOGICAL_VARIABLES:
        supported_vars = list(METEOROLOGICAL_VARIABLES.keys())
        raise ValueError(f"Unsupported variable type: {variable_type}. Supported: {supported_vars}")
    
    # Get variable-specific config
    var_config = METEOROLOGICAL_VARIABLES[variable_type]
    
    # Merge with base config
    config = BASE_METEOROLOGICAL_CONFIG.copy()
    config.update({
        'value_column': var_config['value_column'],
        'validation_rules': var_config['validation_rules'],
        'display_name': var_config['display_name'],
        'unit': var_config['unit'],
        'file_name': var_config['file_name']
    })
    
    # Use dynamic column detection if data is provided
    if data is not None and config.get('auto_detect_structure', True):
        try:
            column_mapping = get_column_mapping(data, variable_type)
            config.update({
                'date_column': column_mapping.date_column,
                'station_column': column_mapping.station_column,
                'value_column': column_mapping.value_column,
                'year_column': column_mapping.year_column,
                'month_column': column_mapping.month_column,
                'code_column': column_mapping.code_column,
                'day_prefix': column_mapping.day_prefix,
                'column_mapping': column_mapping
            })
        except Exception as e:
            logger.warning(f"Failed to detect column mapping: {e}. Using default configuration.")
    
    # Override frequency if specified
    if frequency is not None:
        config['data_frequency'] = frequency
        config['auto_detect_frequency'] = False  # Disable auto-detection when frequency is explicitly set
    
    # Override structure configuration if specified
    if structure_config is not None:
        config.update(structure_config)
        config['auto_detect_structure'] = False  # Disable auto-detection when structure is explicitly set
    
    return config

def get_file_paths_for_variable(variable_type: str) -> Dict[str, Path]:
    """
    Get file paths configuration for a specific variable.
    
    Args:
        variable_type: Type of variable ('temp_min', 'temp_max', etc.)
        
    Returns:
        Dictionary with file paths for the variable
    """
    if variable_type not in METEOROLOGICAL_VARIABLES:
        raise ValueError(f"Unsupported variable type: {variable_type}")
    
    var_config = METEOROLOGICAL_VARIABLES[variable_type]
    file_name = var_config['file_name']
    
    return {
        'input_excel': DATA_DIR / file_name,
        'input_csv': DATA_DIR / file_name.replace('.xlsx', '.csv'),
        'output_clean': DATA_DIR / "clean",
        'output_processed': DATA_DIR / "clean" / f"restructured_{file_name.replace('.xlsx', '.csv')}",
        'output_plots': STATIC_DIR / variable_type,
        'output_forecasts': OUTPUT_DIR / "forecasts" / variable_type
    }

def get_plot_config(variable_type: Optional[str] = None, plot_type: str = 'default') -> Dict[str, Any]:
    """
    Get plotting configuration for different plot types and variables.
    
    Args:
        variable_type: Type of variable for variable-specific config
        plot_type: Type of plot ('default', 'time_series', 'comparison', 'statistics')
        
    Returns:
        Plot configuration dictionary
    """
    config = DEFAULT_PLOT_CONFIG.copy()
    
    if variable_type and variable_type in METEOROLOGICAL_VARIABLES:
        var_plot_config = METEOROLOGICAL_VARIABLES[variable_type]['plot_config']
        config.update(var_plot_config)
    
    # Add plot-type specific configurations
    if plot_type == 'time_series':
        config.update({
            'show_gaps': True,
            'gap_threshold': 1,  # days
            'interpolation_style': 'linear'
        })
    elif plot_type == 'comparison':
        config.update({
            'show_differences': True,
            'difference_color': 'red',
            'alpha': 0.7
        })
    elif plot_type == 'statistics':
        config.update({
            'bins': 30,
            'show_outliers': True,
            'box_plot_style': 'traditional'
        })
    
    return config

def get_supported_variables() -> List[str]:
    """
    Get list of supported meteorological variables.
    
    Returns:
        List of supported variable types
    """
    return list(METEOROLOGICAL_VARIABLES.keys())

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration using validation rules.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validation results dictionary
    """
    return validate_configuration(config)

def load_config_from_file(file_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a file (JSON or YAML).
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If file format is not supported
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    elif file_path.suffix.lower() in ['.yml', '.yaml']:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Validate loaded configuration
    validation_result = validate_config(config)
    if not validation_result['valid']:
        logger.warning(f"Configuration validation warnings: {validation_result['warnings']}")
    
    return config

def save_config_to_file(config: Dict[str, Any], file_path: Path) -> None:
    """
    Save configuration to a file (JSON or YAML).
    
    Args:
        config: Configuration dictionary to save
        file_path: Path where to save the configuration
    """
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    elif file_path.suffix.lower() in ['.yml', '.yaml']:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Configuration saved to: {file_path}")

# Export main functions for backward compatibility
__all__ = [
    'get_config_for_variable',
    'get_file_paths_for_variable',
    'get_plot_config',
    'get_supported_variables',
    'validate_config',
    'load_config_from_file',
    'save_config_to_file',
    'METEOROLOGICAL_VARIABLES',
    'BASE_METEOROLOGICAL_CONFIG',
    'DEFAULT_PLOT_CONFIG',
    'PROCESSING_PIPELINE'
] 