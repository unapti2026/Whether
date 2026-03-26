"""
Paths Configuration

Este módulo centraliza la configuración de rutas y directorios del proyecto.
"""

from pathlib import Path
from typing import Dict, Any

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = PROJECT_ROOT / "static"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
TESTS_DIR = PROJECT_ROOT / "tests"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Data subdirectories
CLEAN_DATA_DIR = DATA_DIR / "clean"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"

# Ensure data subdirectories exist
CLEAN_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)

# Output subdirectories
FORECASTS_DIR = OUTPUT_DIR / "forecasts"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Ensure output subdirectories exist
FORECASTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# File paths configuration
FILE_PATHS = {
    'input_data': DATA_DIR / "temp_min.csv",  # Default, will be overridden
    'output_clean': CLEAN_DATA_DIR,
    'output_plots': STATIC_DIR,
    'output_forecasts': FORECASTS_DIR,
    'logs': LOGS_DIR / "weather_prediction.log"
}

def get_file_paths_for_variable(variable_type: str) -> Dict[str, Path]:
    """
    Get file paths configuration for a specific variable.
    
    Args:
        variable_type: Type of variable ('temp_min', 'temp_max', etc.)
        
    Returns:
        Dictionary with file paths for the variable
    """
    from .settings import METEOROLOGICAL_VARIABLES
    
    if variable_type not in METEOROLOGICAL_VARIABLES:
        raise ValueError(f"Unsupported variable type: {variable_type}")
    
    var_config = METEOROLOGICAL_VARIABLES[variable_type]
    file_name = var_config['file_name']
    
    return {
        'input_excel': DATA_DIR / file_name,
        'input_csv': DATA_DIR / file_name.replace('.xlsx', '.csv'),
        'output_clean': CLEAN_DATA_DIR,
        'output_processed': CLEAN_DATA_DIR / f"restructured_{file_name.replace('.xlsx', '.csv')}",
        'output_plots': STATIC_DIR / variable_type,
        'output_forecasts': FORECASTS_DIR / variable_type
    }

def get_project_structure() -> Dict[str, Path]:
    """
    Get the complete project directory structure.
    
    Returns:
        Dictionary with all project paths
    """
    return {
        'project_root': PROJECT_ROOT,
        'data': DATA_DIR,
        'static': STATIC_DIR,
        'output': OUTPUT_DIR,
        'logs': LOGS_DIR,
        'tests': TESTS_DIR,
        'notebooks': NOTEBOOKS_DIR,
        'clean_data': CLEAN_DATA_DIR,
        'processed_data': PROCESSED_DATA_DIR,
        'raw_data': RAW_DATA_DIR,
        'forecasts': FORECASTS_DIR,
        'plots': PLOTS_DIR,
        'reports': REPORTS_DIR
    } 