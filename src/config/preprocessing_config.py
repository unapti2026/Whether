"""
Preprocessing Configuration Module

This module provides centralized configuration for preprocessing operations,
including data cleaning, validation, and processing parameters.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .settings import get_config_for_variable, get_file_paths_for_variable


class VariableType(Enum):
    """Enumeration for supported variable types."""
    TEMP_MAX = "temp_max"
    TEMP_MIN = "temp_min"
    PRECIPITATION = "precipitation"
    HUMIDITY = "humidity"


class ProcessingMode(Enum):
    """Enumeration for processing modes."""
    CLEAN_ONLY = "clean_only"
    PROCESS_ONLY = "process_only"
    FULL_PIPELINE = "full_pipeline"


@dataclass
class DataCleaningConfig:
    """Configuration for data cleaning operations."""
    remove_outliers: bool = True
    outlier_threshold: float = 3.0
    fill_missing_with_interpolation: bool = True
    max_consecutive_missing: int = 30
    remove_duplicates: bool = True
    validate_date_range: bool = True
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'remove_outliers': self.remove_outliers,
            'outlier_threshold': self.outlier_threshold,
            'fill_missing_with_interpolation': self.fill_missing_with_interpolation,
            'max_consecutive_missing': self.max_consecutive_missing,
            'remove_duplicates': self.remove_duplicates,
            'validate_date_range': self.validate_date_range,
            'min_date': self.min_date,
            'max_date': self.max_date
        }


@dataclass
class DataProcessingConfig:
    """Configuration for data processing operations."""
    restructure_to_time_series: bool = True
    handle_monthly_data: bool = True
    create_daily_series: bool = True
    interpolate_missing_dates: bool = True
    validate_station_consistency: bool = True
    min_station_data_points: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'restructure_to_time_series': self.restructure_to_time_series,
            'handle_monthly_data': self.handle_monthly_data,
            'create_daily_series': self.create_daily_series,
            'interpolate_missing_dates': self.interpolate_missing_dates,
            'validate_station_consistency': self.validate_station_consistency,
            'min_station_data_points': self.min_station_data_points
        }


@dataclass
class OutputConfig:
    """Configuration for output operations."""
    save_cleaned_data: bool = True
    save_processed_data: bool = True
    save_statistics: bool = True
    output_format: str = "csv"
    include_metadata: bool = True
    compression: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'save_cleaned_data': self.save_cleaned_data,
            'save_processed_data': self.save_processed_data,
            'save_statistics': self.save_statistics,
            'output_format': self.output_format,
            'include_metadata': self.include_metadata,
            'compression': self.compression
        }


@dataclass
class PreprocessingConfig:
    """Complete configuration for preprocessing operations."""
    variable_type: VariableType
    data_path: str
    output_path: Optional[str] = None
    processing_mode: ProcessingMode = ProcessingMode.FULL_PIPELINE
    cleaning_config: Optional[DataCleaningConfig] = None
    processing_config: Optional[DataProcessingConfig] = None
    output_config: Optional[OutputConfig] = None
    custom_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if not isinstance(self.variable_type, VariableType):
            self.variable_type = VariableType(self.variable_type)
        
        if self.cleaning_config is None:
            self.cleaning_config = DataCleaningConfig()
        
        if self.processing_config is None:
            self.processing_config = DataProcessingConfig()
        
        if self.output_config is None:
            self.output_config = OutputConfig()
        
        if self.output_path is None:
            self.output_path = self._generate_output_path()
    
    def _generate_output_path(self) -> str:
        """Generate output path based on variable type."""
        file_paths = get_file_paths_for_variable(self.variable_type.value)
        return str(file_paths['output_clean'])
    
    def get_processing_config_dict(self) -> Dict[str, Any]:
        """Get the complete processing configuration as dictionary."""
        base_config = get_config_for_variable(self.variable_type.value)
        
        if self.custom_config:
            base_config.update(self.custom_config)
        
        # Add our specific configurations
        config_dict = {
            'variable_type': self.variable_type.value,
            'data_path': self.data_path,
            'output_path': self.output_path,
            'processing_mode': self.processing_mode.value,
            'cleaning_config': self.cleaning_config.to_dict() if self.cleaning_config else {},
            'processing_config': self.processing_config.to_dict() if self.processing_config else {},
            'output_config': self.output_config.to_dict() if self.output_config else {}
        }
        
        # Merge with base config
        config_dict.update(base_config)
        
        return config_dict
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not Path(self.data_path).exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        if self.variable_type not in VariableType:
            raise ValueError(f"Invalid variable type: {self.variable_type}")
        
        if self.processing_mode not in ProcessingMode:
            raise ValueError(f"Invalid processing mode: {self.processing_mode}")


class PreprocessingConfigFactory:
    """Factory class for creating preprocessing configurations."""
    
    @staticmethod
    def create_temp_max_config(data_path: str, **kwargs) -> PreprocessingConfig:
        """Create configuration for maximum temperature processing."""
        return PreprocessingConfig(
            variable_type=VariableType.TEMP_MAX,
            data_path=data_path,
            **kwargs
        )
    
    @staticmethod
    def create_temp_min_config(data_path: str, **kwargs) -> PreprocessingConfig:
        """Create configuration for minimum temperature processing."""
        return PreprocessingConfig(
            variable_type=VariableType.TEMP_MIN,
            data_path=data_path,
            **kwargs
        )
    
    @staticmethod
    def create_precipitation_config(data_path: str, **kwargs) -> PreprocessingConfig:
        """Create configuration for precipitation processing."""
        return PreprocessingConfig(
            variable_type=VariableType.PRECIPITATION,
            data_path=data_path,
            **kwargs
        )
    
    @staticmethod
    def create_humidity_config(data_path: str, **kwargs) -> PreprocessingConfig:
        """Create configuration for humidity processing."""
        return PreprocessingConfig(
            variable_type=VariableType.HUMIDITY,
            data_path=data_path,
            **kwargs
        )
    
    @staticmethod
    def create_custom_config(variable_type: str, data_path: str, **kwargs) -> PreprocessingConfig:
        """Create configuration for custom variable type."""
        try:
            var_type = VariableType(variable_type)
        except ValueError:
            raise ValueError(f"Invalid variable type: {variable_type}")
        
        return PreprocessingConfig(
            variable_type=var_type,
            data_path=data_path,
            **kwargs
        )


def get_default_preprocessing_config(variable_type: str, data_path: str) -> PreprocessingConfig:
    """
    Get default preprocessing configuration for a variable type.
    
    Args:
        variable_type: Type of variable to process
        data_path: Path to the data directory
        
    Returns:
        PreprocessingConfig object with default settings
    """
    factory = PreprocessingConfigFactory()
    
    if variable_type == VariableType.TEMP_MAX.value:
        return factory.create_temp_max_config(data_path)
    elif variable_type == VariableType.TEMP_MIN.value:
        return factory.create_temp_min_config(data_path)
    elif variable_type == VariableType.PRECIPITATION.value:
        return factory.create_precipitation_config(data_path)
    elif variable_type == VariableType.HUMIDITY.value:
        return factory.create_humidity_config(data_path)
    else:
        return factory.create_custom_config(variable_type, data_path)


def validate_preprocessing_config(config: PreprocessingConfig) -> bool:
    """
    Validate a preprocessing configuration.
    
    Args:
        config: PreprocessingConfig to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    try:
        config.validate()
        return True
    except ValueError as e:
        raise ValueError(f"Invalid preprocessing configuration: {e}")


def get_supported_variable_types() -> list:
    """Get list of supported variable types."""
    return [var_type.value for var_type in VariableType]


def get_processing_modes() -> list:
    """Get list of available processing modes."""
    return [mode.value for mode in ProcessingMode] 