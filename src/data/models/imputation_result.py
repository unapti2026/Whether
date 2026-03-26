"""
Imputation Result Model

This module defines the ImputationResult model for representing
the results of missing value imputation operations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd


@dataclass
class ImputationResult:
    """
    Model for representing the results of missing value imputation operations.
    
    This model encapsulates imputation results with metadata, statistics,
    and quality metrics.
    """
    
    # Core information
    status: str  # 'success', 'failed', 'partial'
    variable_type: str
    column_name: str
    imputation_date: datetime = field(default_factory=datetime.now)
    
    # Data information
    original_data_shape: tuple = field(default_factory=tuple)
    imputed_data_shape: tuple = field(default_factory=tuple)
    
    # Imputation statistics
    original_missing_count: int = 0
    imputed_count: int = 0
    remaining_missing_count: int = 0
    imputation_percentage: float = 0.0
    
    # Quality metrics
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Imputation details
    imputation_method: str = ""
    imputation_parameters: Dict[str, Any] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    
    # Station-specific results
    station_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and calculate derived fields after creation."""
        self._validate_status()
        self._calculate_imputation_percentage()
    
    def _validate_status(self):
        """Validate the imputation status."""
        valid_statuses = ['success', 'failed', 'partial']
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}. Valid statuses: {valid_statuses}")
    
    def _calculate_imputation_percentage(self):
        """Calculate the percentage of values that were imputed."""
        if self.original_missing_count > 0:
            self.imputation_percentage = (self.imputed_count / self.original_missing_count) * 100
        else:
            self.imputation_percentage = 0.0
    
    def is_successful(self) -> bool:
        """
        Check if imputation was successful.
        
        Returns:
            True if imputation was successful, False otherwise
        """
        return self.status == 'success'
    
    def has_warnings(self) -> bool:
        """
        Check if imputation had warnings.
        
        Returns:
            True if there are warnings, False otherwise
        """
        return len(self.warnings) > 0
    
    def has_errors(self) -> bool:
        """
        Check if imputation had errors.
        
        Returns:
            True if there are errors, False otherwise
        """
        return len(self.errors) > 0
    
    def get_imputation_efficiency(self) -> float:
        """
        Calculate imputation efficiency (imputed / original missing).
        
        Returns:
            Imputation efficiency as a percentage
        """
        return self.imputation_percentage
    
    def get_imputation_rate(self) -> float:
        """
        Calculate imputation rate (imputed values per second).
        
        Returns:
            Imputation rate in values per second
        """
        if self.processing_time_seconds == 0:
            return 0.0
        return self.imputed_count / self.processing_time_seconds
    
    def add_station_result(self, station_name: str, result: Dict[str, Any]):
        """
        Add imputation result for a specific station.
        
        Args:
            station_name: Name of the station
            result: Station-specific imputation result
        """
        self.station_results[station_name] = result
    
    def add_warning(self, warning: str):
        """
        Add a warning message.
        
        Args:
            warning: Warning message to add
        """
        self.warnings.append(f"{datetime.now().isoformat()}: {warning}")
    
    def add_error(self, error: str):
        """
        Add an error message.
        
        Args:
            error: Error message to add
        """
        self.errors.append(f"{datetime.now().isoformat()}: {error}")
        if self.status == 'success':
            self.status = 'partial'
    
    def get_station_summary(self) -> Dict[str, Any]:
        """
        Get a summary of station-specific results.
        
        Returns:
            Dictionary with station summary
        """
        if not self.station_results:
            return {}
        
        total_stations = len(self.station_results)
        successful_stations = sum(1 for result in self.station_results.values() 
                                if result.get('status') == 'success')
        
        station_missing_counts = [result.get('original_missing_count', 0) 
                                for result in self.station_results.values()]
        station_imputed_counts = [result.get('imputed_count', 0) 
                                for result in self.station_results.values()]
        
        return {
            'total_stations': total_stations,
            'successful_stations': successful_stations,
            'failed_stations': total_stations - successful_stations,
            'max_missing_per_station': max(station_missing_counts) if station_missing_counts else 0,
            'min_missing_per_station': min(station_missing_counts) if station_missing_counts else 0,
            'avg_missing_per_station': sum(station_missing_counts) / len(station_missing_counts) if station_missing_counts else 0,
            'max_imputed_per_station': max(station_imputed_counts) if station_imputed_counts else 0,
            'min_imputed_per_station': min(station_imputed_counts) if station_imputed_counts else 0,
            'avg_imputed_per_station': sum(station_imputed_counts) / len(station_imputed_counts) if station_imputed_counts else 0
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the imputation result.
        
        Returns:
            Dictionary with imputation summary
        """
        return {
            'status': self.status,
            'variable_type': self.variable_type,
            'column_name': self.column_name,
            'imputation_date': self.imputation_date.isoformat(),
            'original_data_shape': self.original_data_shape,
            'imputed_data_shape': self.imputed_data_shape,
            'original_missing_count': self.original_missing_count,
            'imputed_count': self.imputed_count,
            'remaining_missing_count': self.remaining_missing_count,
            'imputation_percentage': self.imputation_percentage,
            'imputation_efficiency': self.get_imputation_efficiency(),
            'imputation_rate': self.get_imputation_rate(),
            'imputation_method': self.imputation_method,
            'processing_time_seconds': self.processing_time_seconds,
            'quality_metrics': self.quality_metrics,
            'station_summary': self.get_station_summary(),
            'warnings_count': len(self.warnings),
            'errors_count': len(self.errors)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary representation.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            'status': self.status,
            'variable_type': self.variable_type,
            'column_name': self.column_name,
            'imputation_date': self.imputation_date.isoformat(),
            'original_data_shape': self.original_data_shape,
            'imputed_data_shape': self.imputed_data_shape,
            'original_missing_count': self.original_missing_count,
            'imputed_count': self.imputed_count,
            'remaining_missing_count': self.remaining_missing_count,
            'imputation_percentage': self.imputation_percentage,
            'imputation_method': self.imputation_method,
            'imputation_parameters': self.imputation_parameters,
            'processing_time_seconds': self.processing_time_seconds,
            'quality_metrics': self.quality_metrics,
            'station_results': self.station_results,
            'warnings': self.warnings,
            'errors': self.errors
        }
    
    @classmethod
    def create_success(cls, 
                      variable_type: str, 
                      column_name: str,
                      original_missing_count: int,
                      imputed_count: int,
                      imputation_method: str,
                      processing_time: float,
                      **kwargs) -> 'ImputationResult':
        """
        Create a successful imputation result.
        
        Args:
            variable_type: Type of meteorological variable
            column_name: Name of the column that was imputed
            original_missing_count: Number of missing values before imputation
            imputed_count: Number of values that were imputed
            imputation_method: Method used for imputation
            processing_time: Processing time in seconds
            **kwargs: Additional parameters
            
        Returns:
            ImputationResult instance with success status
        """
        remaining_missing = original_missing_count - imputed_count
        
        return cls(
            status='success',
            variable_type=variable_type,
            column_name=column_name,
            original_missing_count=original_missing_count,
            imputed_count=imputed_count,
            remaining_missing_count=remaining_missing,
            imputation_method=imputation_method,
            processing_time_seconds=processing_time,
            **kwargs
        )
    
    @classmethod
    def create_failure(cls, 
                      variable_type: str, 
                      column_name: str,
                      error_message: str,
                      **kwargs) -> 'ImputationResult':
        """
        Create a failed imputation result.
        
        Args:
            variable_type: Type of meteorological variable
            column_name: Name of the column that was being imputed
            error_message: Error message
            **kwargs: Additional parameters
            
        Returns:
            ImputationResult instance with failed status
        """
        result = cls(
            status='failed',
            variable_type=variable_type,
            column_name=column_name,
            **kwargs
        )
        result.add_error(error_message)
        return result
    
    def __str__(self) -> str:
        """String representation of the result."""
        return (f"ImputationResult(status='{self.status}', "
                f"variable_type='{self.variable_type}', "
                f"column='{self.column_name}', "
                f"imputed={self.imputed_count}/{self.original_missing_count} "
                f"({self.imputation_percentage:.1f}%), "
                f"method='{self.imputation_method}')") 