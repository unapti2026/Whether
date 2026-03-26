"""
Validation Rules

Este módulo centraliza las reglas de validación para datos y configuraciones.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
from pathlib import Path

# Data validation rules
DATA_VALIDATION_RULES = {
    'temp_min': {
        'min_value': -50.0,
        'max_value': 60.0,
        'required_columns': ['Código', 'Estación', 'Año', 'Mes'],
        'date_range': {
            'min_year': 1900,
            'max_year': 2100
        },
        'missing_threshold': 0.3  # 30% missing values allowed
    },
    'temp_max': {
        'min_value': -30.0,
        'max_value': 70.0,
        'required_columns': ['Código', 'Estación', 'Año', 'Mes'],
        'date_range': {
            'min_year': 1900,
            'max_year': 2100
        },
        'missing_threshold': 0.3
    },
    'precipitation': {
        'min_value': 0.0,
        'max_value': 1000.0,
        'required_columns': ['Código', 'Estación', 'Año', 'Mes'],
        'date_range': {
            'min_year': 1900,
            'max_year': 2100
        },
        'missing_threshold': 0.4
    },
    'humidity': {
        'min_value': 0.0,
        'max_value': 100.0,
        'required_columns': ['Código', 'Estación', 'Año', 'Mes'],
        'date_range': {
            'min_year': 1900,
            'max_year': 2100
        },
        'missing_threshold': 0.3
    }
}

# Configuration validation rules
CONFIG_VALIDATION_RULES = {
    'required_fields': [
        'year_column',
        'month_column',
        'station_column',
        'code_column',
        'date_column',
        'value_column'
    ],
    'optional_fields': [
        'day_prefix',
        'max_days',
        'clean_values'
    ],
    'data_types': {
        'year_column': str,
        'month_column': str,
        'station_column': str,
        'code_column': str,
        'date_column': str,
        'value_column': str,
        'day_prefix': str,
        'max_days': int,
        'clean_values': list
    }
}

# File validation rules
FILE_VALIDATION_RULES = {
    'supported_formats': ['.csv', '.xlsx', '.xls'],
    'max_file_size_mb': 100,
    'required_extensions': ['.csv', '.xlsx'],
    'encoding': 'utf-8'
}

def validate_dataframe_structure(data: pd.DataFrame, variable_type: str) -> Dict[str, Any]:
    """
    Validate DataFrame structure according to variable type rules.
    
    Args:
        data: DataFrame to validate
        variable_type: Type of variable ('temp_min', 'temp_max', etc.)
        
    Returns:
        Dictionary with validation results
    """
    if variable_type not in DATA_VALIDATION_RULES:
        raise ValueError(f"Unknown variable type: {variable_type}")
    
    rules = DATA_VALIDATION_RULES[variable_type]
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required columns
    missing_columns = []
    for col in rules['required_columns']:
        if col not in data.columns:
            missing_columns.append(col)
    
    if missing_columns:
        results['valid'] = False
        results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check data types
    if 'Año' in data.columns:
        if not pd.api.types.is_numeric_dtype(data['Año']):
            results['warnings'].append("Year column is not numeric")
    
    if 'Mes' in data.columns:
        if not pd.api.types.is_numeric_dtype(data['Mes']):
            results['warnings'].append("Month column is not numeric")
    
    # Check date range
    if 'Año' in data.columns:
        year_range = rules['date_range']
        min_year = data['Año'].min()
        max_year = data['Año'].max()
        
        if min_year < year_range['min_year'] or max_year > year_range['max_year']:
            results['warnings'].append(f"Year range ({min_year}-{max_year}) outside expected range")
    
    return results

def validate_data_values(data: pd.DataFrame, value_column: str, variable_type: str) -> Dict[str, Any]:
    """
    Validate data values according to variable type rules.
    
    Args:
        data: DataFrame to validate
        value_column: Name of the value column
        variable_type: Type of variable
        
    Returns:
        Dictionary with validation results
    """
    if variable_type not in DATA_VALIDATION_RULES:
        raise ValueError(f"Unknown variable type: {variable_type}")
    
    rules = DATA_VALIDATION_RULES[variable_type]
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    if value_column not in data.columns:
        results['valid'] = False
        results['errors'].append(f"Value column '{value_column}' not found")
        return results
    
    # Convert to numeric, handling errors
    numeric_data = pd.to_numeric(data[value_column], errors='coerce')
    
    # Check for missing values
    missing_count = numeric_data.isna().sum()
    missing_percentage = missing_count / len(numeric_data)
    
    results['statistics']['missing_count'] = int(missing_count)
    results['statistics']['missing_percentage'] = float(missing_percentage)
    
    if missing_percentage > rules['missing_threshold']:
        results['warnings'].append(f"High missing values: {missing_percentage:.2%}")
    
    # Check value range
    valid_data = numeric_data.dropna()
    if len(valid_data) > 0:
        min_val = valid_data.min()
        max_val = valid_data.max()
        
        results['statistics']['min_value'] = float(min_val)
        results['statistics']['max_value'] = float(max_val)
        results['statistics']['mean_value'] = float(valid_data.mean())
        results['statistics']['std_value'] = float(valid_data.std())
        
        if min_val < rules['min_value']:
            results['warnings'].append(f"Values below minimum: {min_val} < {rules['min_value']}")
        
        if max_val > rules['max_value']:
            results['warnings'].append(f"Values above maximum: {max_val} > {rules['max_value']}")
    
    return results

def validate_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    rules = CONFIG_VALIDATION_RULES
    
    # Check required fields
    missing_fields = []
    for field in rules['required_fields']:
        if field not in config:
            missing_fields.append(field)
    
    if missing_fields:
        results['valid'] = False
        results['errors'].append(f"Missing required fields: {missing_fields}")
    
    # Check data types
    for field, expected_type in rules['data_types'].items():
        if field in config:
            if not isinstance(config[field], expected_type):
                results['warnings'].append(f"Field '{field}' should be {expected_type.__name__}")
    
    return results

def validate_file_path(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate file path and format.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        results['valid'] = False
        results['errors'].append(f"File does not exist: {file_path}")
        return results
    
    # Check file extension
    if file_path.suffix not in FILE_VALIDATION_RULES['supported_formats']:
        results['warnings'].append(f"Unsupported file format: {file_path.suffix}")
    
    # Check file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > FILE_VALIDATION_RULES['max_file_size_mb']:
        results['warnings'].append(f"File size ({file_size_mb:.1f}MB) exceeds limit")
    
    return results

def get_validation_rules(variable_type: str) -> Dict[str, Any]:
    """
    Get validation rules for a specific variable type.
    
    Args:
        variable_type: Type of variable
        
    Returns:
        Validation rules dictionary
    """
    if variable_type not in DATA_VALIDATION_RULES:
        raise ValueError(f"Unknown variable type: {variable_type}")
    
    return DATA_VALIDATION_RULES[variable_type].copy() 