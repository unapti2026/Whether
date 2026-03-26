"""
Data Validator

This module contains the DataValidator class for validating data integrity
and structure in the weather prediction system.
"""

from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from ..exceptions import ValidationError


class DataValidator:
    """
    Centralized validator for data integrity and structure.
    
    This class provides methods to validate various aspects of data,
    including structure, types, ranges, and completeness.
    """
    
    def __init__(self):
        """Initialize the DataValidator."""
        self.validation_rules = {}
        self.validation_history = []
    
    def validate_dataframe_structure(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that the DataFrame has the required columns.
        
        Args:
            data: DataFrame to validate
            required_columns: List of column names that must be present
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValidationError: If required columns are missing
        """
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Data must be a pandas DataFrame", value=type(data))
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValidationError(
                f"Missing required columns: {missing_columns}",
                field="columns",
                value=missing_columns,
                constraint="required_columns"
            )
        
        return True
    
    def validate_data_types(self, data: pd.DataFrame, column_types: Dict[str, type]) -> bool:
        """
        Validate that DataFrame columns have the expected data types.
        
        Args:
            data: DataFrame to validate
            column_types: Dictionary mapping column names to expected types
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValidationError: If data types don't match expectations
        """
        for column, expected_type in column_types.items():
            if column not in data.columns:
                raise ValidationError(
                    f"Column '{column}' not found in DataFrame",
                    field=column,
                    value=None,
                    constraint="required_column"
                )
            
            actual_type = data[column].dtype
            if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                raise ValidationError(
                    f"Column '{column}' has type {actual_type}, expected {expected_type}",
                    field=column,
                    value=actual_type,
                    constraint=f"type_{expected_type}"
                )
        
        return True
    
    def validate_value_ranges(self, data: pd.DataFrame, column: str, 
                            min_value: Optional[float] = None, 
                            max_value: Optional[float] = None) -> bool:
        """
        Validate that values in a column are within specified ranges.
        
        Args:
            data: DataFrame to validate
            column: Column name to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValidationError: If values are outside the specified range
        """
        if column not in data.columns:
            raise ValidationError(
                f"Column '{column}' not found in DataFrame",
                field=column,
                value=None,
                constraint="required_column"
            )
        
        # Convert to numeric, ignoring errors
        numeric_data = pd.to_numeric(data[column], errors='coerce')
        
        if min_value is not None:
            below_min = numeric_data < min_value
            if below_min.any():
                invalid_values = numeric_data[below_min]
                raise ValidationError(
                    f"Column '{column}' contains values below minimum {min_value}",
                    field=column,
                    value=invalid_values.tolist(),
                    constraint=f"min_{min_value}"
                )
        
        if max_value is not None:
            above_max = numeric_data > max_value
            if above_max.any():
                invalid_values = numeric_data[above_max]
                raise ValidationError(
                    f"Column '{column}' contains values above maximum {max_value}",
                    field=column,
                    value=invalid_values.tolist(),
                    constraint=f"max_{max_value}"
                )
        
        return True
    
    def validate_missing_values(self, data: pd.DataFrame, column: str, 
                              max_missing_ratio: float = 0.3) -> bool:
        """
        Validate that missing values in a column don't exceed the threshold.
        
        Args:
            data: DataFrame to validate
            column: Column name to validate
            max_missing_ratio: Maximum allowed ratio of missing values
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValidationError: If missing value ratio exceeds threshold
        """
        if column not in data.columns:
            raise ValidationError(
                f"Column '{column}' not found in DataFrame",
                field=column,
                value=None,
                constraint="required_column"
            )
        
        missing_count = data[column].isnull().sum()
        total_count = len(data[column])
        missing_ratio = missing_count / total_count if total_count > 0 else 0
        
        if missing_ratio > max_missing_ratio:
            raise ValidationError(
                f"Column '{column}' has {missing_ratio:.2%} missing values, "
                f"exceeds threshold of {max_missing_ratio:.2%}",
                field=column,
                value=missing_ratio,
                constraint=f"max_missing_{max_missing_ratio}"
            )
        
        return True
    
    def validate_date_range(self, data: pd.DataFrame, date_column: str,
                          min_date: Optional[str] = None,
                          max_date: Optional[str] = None) -> bool:
        """
        Validate that dates in a column are within specified range.
        
        Args:
            data: DataFrame to validate
            date_column: Column name containing dates
            min_date: Minimum allowed date (YYYY-MM-DD format)
            max_date: Maximum allowed date (YYYY-MM-DD format)
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValidationError: If dates are outside the specified range
        """
        if date_column not in data.columns:
            raise ValidationError(
                f"Date column '{date_column}' not found in DataFrame",
                field=date_column,
                value=None,
                constraint="required_column"
            )
        
        # Convert to datetime
        try:
            dates = pd.to_datetime(data[date_column])
        except Exception as e:
            raise ValidationError(
                f"Column '{date_column}' contains invalid dates: {e}",
                field=date_column,
                value=data[date_column].head().tolist(),
                constraint="valid_dates"
            )
        
        if min_date is not None:
            min_dt = pd.to_datetime(min_date)
            below_min = dates < min_dt
            if below_min.any():
                invalid_dates = dates[below_min]
                raise ValidationError(
                    f"Column '{date_column}' contains dates before {min_date}",
                    field=date_column,
                    value=invalid_dates.dt.strftime('%Y-%m-%d').tolist(),
                    constraint=f"min_date_{min_date}"
                )
        
        if max_date is not None:
            max_dt = pd.to_datetime(max_date)
            above_max = dates > max_dt
            if above_max.any():
                invalid_dates = dates[above_max]
                raise ValidationError(
                    f"Column '{date_column}' contains dates after {max_date}",
                    field=date_column,
                    value=invalid_dates.dt.strftime('%Y-%m-%d').tolist(),
                    constraint=f"max_date_{max_date}"
                )
        
        return True
    
    def validate_time_series(self, time_series: pd.Series) -> bool:
        """
        Validate that a time series is suitable for processing.
        
        Args:
            time_series: Time series to validate
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check if series is not empty
            if len(time_series) == 0:
                raise ValidationError("Time series is empty")
            
            # Check if series has enough data points
            if len(time_series) < 10:
                raise ValidationError("Time series must have at least 10 data points")
            
            # Check if series has any non-null values
            if time_series.isnull().all():
                raise ValidationError("Time series contains only null values")
            
            # Check if series has sufficient variance
            clean_series = time_series.dropna()
            if len(clean_series) < 5:
                raise ValidationError("Time series must have at least 5 non-null values")
            
            variance = np.var(clean_series)
            if variance == 0:
                raise ValidationError("Time series has zero variance")
            
            return True
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Time series validation failed: {e}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of validation operations performed.
        
        Returns:
            Dictionary containing validation summary
        """
        return {
            'total_validations': len(self.validation_history),
            'validation_rules': self.validation_rules,
            'recent_validations': self.validation_history[-10:] if self.validation_history else []
        } 