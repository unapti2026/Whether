"""
Column Configuration Module

This module provides dynamic column configuration and mapping for meteorological data,
eliminating hardcoded column references and providing flexible column detection
and validation capabilities.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class ColumnType(Enum):
    """Enumeration for different column types in meteorological data."""
    DATE = "date"
    STATION = "station"
    VALUE = "value"
    YEAR = "year"
    MONTH = "month"
    CODE = "code"
    DAY = "day"


@dataclass
class ColumnMapping:
    """Configuration for column mapping in meteorological data."""
    date_column: str
    station_column: str
    value_column: str
    year_column: str
    month_column: str
    code_column: str
    day_prefix: Optional[str]
    
    def get_column(self, column_type: ColumnType) -> str:
        """Get column name for a specific column type."""
        mapping = {
            ColumnType.DATE: self.date_column,
            ColumnType.STATION: self.station_column,
            ColumnType.VALUE: self.value_column,
            ColumnType.YEAR: self.year_column,
            ColumnType.MONTH: self.month_column,
            ColumnType.CODE: self.code_column,
            ColumnType.DAY: self.day_prefix
        }
        return mapping[column_type]
    
    def get_required_columns(self) -> List[str]:
        """Get list of required columns for data validation."""
        return [
            self.date_column,
            self.station_column,
            self.value_column,
            self.year_column,
            self.month_column,
            self.code_column
        ]
    
    def get_day_columns(self, max_days: int = 31) -> List[str]:
        """Get list of day column names."""
        return [f"{self.day_prefix}{i}" for i in range(1, max_days + 1)]
    
    def validate_columns(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that all required columns are present in the data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in data.columns]
        return len(missing_columns) == 0, missing_columns


class ColumnDetector:
    """
    Dynamic column detection service for meteorological data.
    
    This class provides intelligent column detection based on data content,
    column names, and patterns to automatically identify the correct column
    mapping for different data structures.
    """
    
    def __init__(self):
        """Initialize the column detector."""
        self._initialize_column_patterns()
    
    def _initialize_column_patterns(self):
        """Initialize patterns for column detection."""
        self.date_patterns = {
            'exact': ['fecha', 'date', 'fecha_medicion', 'fecha_observacion'],
            'partial': ['fecha', 'date', 'fech', 'dat']
        }
        
        self.station_patterns = {
            'exact': ['estación', 'estacion', 'station', 'estacion_meteorologica'],
            'partial': ['estación', 'estacion', 'station', 'est']
        }
        
        self.value_patterns = {
            'temp_max': {
                'exact': ['temperatura', 'temp_max', 'temperatura_maxima', 'tmax'],
                'partial': ['temp', 'temperatura', 'tmax']
            },
            'temp_min': {
                'exact': ['temperatura', 'temp_min', 'temperatura_minima', 'tmin'],
                'partial': ['temp', 'temperatura', 'tmin']
            },
            'precipitation': {
                'exact': ['precipitación', 'precipitacion', 'precipitation', 'lluvia'],
                'partial': ['precip', 'lluvia', 'rain']
            },
            'humidity': {
                'exact': ['humedad', 'humidity', 'humedad_relativa', 'hr'],
                'partial': ['humed', 'humidity', 'hr']
            }
        }
        
        self.year_patterns = {
            'exact': ['año', 'ano', 'year', 'año_observacion'],
            'partial': ['año', 'ano', 'year', 'an']
        }
        
        self.month_patterns = {
            'exact': ['mes', 'month', 'mes_observacion'],
            'partial': ['mes', 'month', 'me']
        }
        
        self.code_patterns = {
            'exact': ['código', 'codigo', 'code', 'codigo_estacion'],
            'partial': ['código', 'codigo', 'code', 'cod']
        }
        
        self.day_prefix_patterns = {
            'exact': ['día', 'dia', 'day', 'd'],
            'partial': ['día', 'dia', 'day', 'd']
        }
    
    def detect_columns(self, data: pd.DataFrame, variable_type: str) -> ColumnMapping:
        """
        Detect column mapping for a given DataFrame and variable type.
        
        Args:
            data: DataFrame to analyze
            variable_type: Type of meteorological variable
            
        Returns:
            ColumnMapping object with detected column names
        """
        columns_lower = [col.lower() for col in data.columns]
        columns_original = list(data.columns)
        
        # Detect each column type
        date_column = self._detect_date_column(columns_lower, columns_original)
        station_column = self._detect_station_column(columns_lower, columns_original)
        value_column = self._detect_value_column(columns_lower, columns_original, variable_type)
        year_column = self._detect_year_column(columns_lower, columns_original)
        month_column = self._detect_month_column(columns_lower, columns_original)
        code_column = self._detect_code_column(columns_lower, columns_original)
        day_prefix = self._detect_day_prefix(columns_lower, columns_original)
        
        return ColumnMapping(
            date_column=date_column,
            station_column=station_column,
            value_column=value_column,
            year_column=year_column,
            month_column=month_column,
            code_column=code_column,
            day_prefix=day_prefix
        )
    
    def _detect_date_column(self, columns_lower: List[str], columns_original: List[str]) -> str:
        """Detect the date column."""
        for pattern in self.date_patterns['exact']:
            for i, col in enumerate(columns_lower):
                if col == pattern:
                    return columns_original[i]
        
        for pattern in self.date_patterns['partial']:
            for i, col in enumerate(columns_lower):
                if pattern in col:
                    return columns_original[i]
        
        return 'Fecha'  # Default fallback
    
    def _detect_station_column(self, columns_lower: List[str], columns_original: List[str]) -> str:
        """Detect the station column."""
        for pattern in self.station_patterns['exact']:
            for i, col in enumerate(columns_lower):
                if col == pattern:
                    return columns_original[i]
        
        for pattern in self.station_patterns['partial']:
            for i, col in enumerate(columns_lower):
                if pattern in col:
                    return columns_original[i]
        
        return 'Estación'  # Default fallback
    
    def _detect_value_column(self, columns_lower: List[str], columns_original: List[str], variable_type: str) -> str:
        """Detect the value column based on variable type."""
        if variable_type not in self.value_patterns:
            return 'Temperatura'  # Default fallback
        
        patterns = self.value_patterns[variable_type]
        
        for pattern in patterns['exact']:
            for i, col in enumerate(columns_lower):
                if col == pattern:
                    return columns_original[i]
        
        for pattern in patterns['partial']:
            for i, col in enumerate(columns_lower):
                if pattern in col:
                    return columns_original[i]
        
        # Return default based on variable type
        defaults = {
            'temp_max': 'Temperatura',
            'temp_min': 'Temperatura',
            'precipitation': 'Precipitación',
            'humidity': 'Humedad'
        }
        return defaults.get(variable_type, 'Temperatura')
    
    def _detect_year_column(self, columns_lower: List[str], columns_original: List[str]) -> str:
        """Detect the year column."""
        for pattern in self.year_patterns['exact']:
            for i, col in enumerate(columns_lower):
                if col == pattern:
                    return columns_original[i]
        
        for pattern in self.year_patterns['partial']:
            for i, col in enumerate(columns_lower):
                if pattern in col:
                    return columns_original[i]
        
        return 'Año'  # Default fallback
    
    def _detect_month_column(self, columns_lower: List[str], columns_original: List[str]) -> str:
        """Detect the month column."""
        for pattern in self.month_patterns['exact']:
            for i, col in enumerate(columns_lower):
                if col == pattern:
                    return columns_original[i]
        
        for pattern in self.month_patterns['partial']:
            for i, col in enumerate(columns_lower):
                if pattern in col:
                    return columns_original[i]
        
        return 'Mes'  # Default fallback
    
    def _detect_code_column(self, columns_lower: List[str], columns_original: List[str]) -> str:
        """Detect the code column."""
        for pattern in self.code_patterns['exact']:
            for i, col in enumerate(columns_lower):
                if col == pattern:
                    return columns_original[i]
        
        for pattern in self.code_patterns['partial']:
            for i, col in enumerate(columns_lower):
                if pattern in col:
                    return columns_original[i]
        
        return 'Código'  # Default fallback
    
    def _detect_day_prefix(self, columns_lower: List[str], columns_original: List[str]) -> Optional[str]:
        """Detect the day prefix (e.g., 'Día', 'Day', etc.) only if columns like 'Día1', 'Day2' exist."""
        # Buscar prefijos válidos en las columnas
        for pattern in self.day_prefix_patterns['exact']:
            for i, col in enumerate(columns_lower):
                # Buscar columnas que empiecen por el patrón y terminen en un número
                for j in range(1, 32):
                    expected = f"{pattern}{j}"
                    if col.startswith(expected):
                        # Devolver el prefijo original correspondiente
                        return columns_original[i][:len(pattern)]
        # Si no se encuentra ningún prefijo válido, retornar None
        return None


class ColumnConfigManager:
    """
    Manager for column configuration across the application.
    
    This class provides a centralized way to manage column configurations,
    including detection, validation, and caching of column mappings.
    """
    
    def __init__(self):
        """Initialize the column configuration manager."""
        self.detector = ColumnDetector()
        self._column_cache: Dict[str, ColumnMapping] = {}
    
    def get_column_mapping(self, data: pd.DataFrame, variable_type: str, 
                          force_detection: bool = False) -> ColumnMapping:
        """
        Get column mapping for data and variable type.
        
        Args:
            data: DataFrame to analyze
            variable_type: Type of meteorological variable
            force_detection: Force re-detection even if cached
            
        Returns:
            ColumnMapping object
        """
        cache_key = f"{variable_type}_{hash(tuple(data.columns))}"
        
        if not force_detection and cache_key in self._column_cache:
            return self._column_cache[cache_key]
        
        # Detect columns
        column_mapping = self.detector.detect_columns(data, variable_type)
        
        # Validate the mapping
        is_valid, missing_columns = column_mapping.validate_columns(data)
        if not is_valid:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Cache the result
        self._column_cache[cache_key] = column_mapping
        
        return column_mapping
    
    def get_column_name(self, data: pd.DataFrame, variable_type: str, 
                       column_type: ColumnType) -> str:
        """
        Get a specific column name for the given data and variable type.
        
        Args:
            data: DataFrame to analyze
            variable_type: Type of meteorological variable
            column_type: Type of column to get
            
        Returns:
            Column name
        """
        mapping = self.get_column_mapping(data, variable_type)
        return mapping.get_column(column_type)
    
    def validate_data_structure(self, data: pd.DataFrame, variable_type: str) -> Dict[str, Any]:
        """
        Validate the data structure against the detected column mapping.
        
        Args:
            data: DataFrame to validate
            variable_type: Type of meteorological variable
            
        Returns:
            Validation result dictionary
        """
        try:
            mapping = self.get_column_mapping(data, variable_type)
            is_valid, missing_columns = mapping.validate_columns(data)
            
            return {
                'valid': is_valid,
                'missing_columns': missing_columns,
                'column_mapping': mapping,
                'total_columns': len(data.columns),
                'required_columns': mapping.get_required_columns(),
                'available_columns': list(data.columns)
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'total_columns': len(data.columns),
                'available_columns': list(data.columns)
            }
    
    def clear_cache(self):
        """Clear the column mapping cache."""
        self._column_cache.clear()


# Global instance for easy access
column_config_manager = ColumnConfigManager()


def get_column_mapping(data: pd.DataFrame, variable_type: str) -> ColumnMapping:
    """
    Convenience function to get column mapping.
    
    Args:
        data: DataFrame to analyze
        variable_type: Type of meteorological variable
        
    Returns:
        ColumnMapping object
    """
    return column_config_manager.get_column_mapping(data, variable_type)


def get_column_name(data: pd.DataFrame, variable_type: str, column_type: ColumnType) -> str:
    """
    Convenience function to get a specific column name.
    
    Args:
        data: DataFrame to analyze
        variable_type: Type of meteorological variable
        column_type: Type of column to get
        
    Returns:
        Column name
    """
    return column_config_manager.get_column_name(data, variable_type, column_type)


def validate_data_structure(data: pd.DataFrame, variable_type: str) -> Dict[str, Any]:
    """
    Convenience function to validate data structure.
    
    Args:
        data: DataFrame to validate
        variable_type: Type of meteorological variable
        
    Returns:
        Validation result dictionary
    """
    return column_config_manager.validate_data_structure(data, variable_type) 