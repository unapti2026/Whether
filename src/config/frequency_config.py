"""
Frequency Configuration Module

This module provides configuration and utilities for handling different data frequencies
in meteorological data processing. It supports automatic frequency detection and
parameterization for various time series frequencies.
"""

from typing import Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta


class DataFrequency(Enum):
    """Enumeration for supported data frequencies."""
    HOURLY = "H"
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"
    
    @classmethod
    def from_string(cls, frequency_str: str) -> 'DataFrequency':
        """Create DataFrequency from string representation."""
        frequency_map = {
            'hourly': cls.HOURLY,
            'daily': cls.DAILY,
            'weekly': cls.WEEKLY,
            'monthly': cls.MONTHLY,
            'quarterly': cls.QUARTERLY,
            'yearly': cls.YEARLY,
            'H': cls.HOURLY,
            'D': cls.DAILY,
            'W': cls.WEEKLY,
            'M': cls.MONTHLY,
            'Q': cls.QUARTERLY,
            'Y': cls.YEARLY
        }
        
        if frequency_str not in frequency_map:
            raise ValueError(f"Unsupported frequency: {frequency_str}. Supported: {list(frequency_map.keys())}")
        
        return frequency_map[frequency_str]


@dataclass
class FrequencyConfig:
    """Configuration class for data frequency parameters."""
    frequency: DataFrequency
    pandas_freq: str
    description: str
    max_gap_days: int
    seasonal_period: int
    min_data_points: int
    interpolation_method: str
    validation_rules: Dict[str, Any]
    
    @classmethod
    def create_for_frequency(cls, frequency: Union[DataFrequency, str]) -> 'FrequencyConfig':
        """Create frequency configuration for a specific frequency."""
        if isinstance(frequency, str):
            frequency = DataFrequency.from_string(frequency)
        
        configs = {
            DataFrequency.HOURLY: {
                'pandas_freq': 'H',
                'description': 'Hourly data',
                'max_gap_days': 7,  # 7 days for hourly data
                'seasonal_period': 24,  # 24 hours
                'min_data_points': 168,  # 1 week of hourly data
                'interpolation_method': 'linear',
                'validation_rules': {
                    'max_consecutive_missing': 48,  # 2 days
                    'min_interval_hours': 1,
                    'max_interval_hours': 1
                }
            },
            DataFrequency.DAILY: {
                'pandas_freq': 'D',
                'description': 'Daily data',
                'max_gap_days': 30,  # 30 days
                'seasonal_period': 365,  # 1 year
                'min_data_points': 30,  # 1 month
                'interpolation_method': 'linear',
                'validation_rules': {
                    'max_consecutive_missing': 7,  # 1 week
                    'min_interval_days': 1,
                    'max_interval_days': 1
                }
            },
            DataFrequency.WEEKLY: {
                'pandas_freq': 'W',
                'description': 'Weekly data',
                'max_gap_days': 90,  # 3 months
                'seasonal_period': 52,  # 1 year
                'min_data_points': 12,  # 3 months
                'interpolation_method': 'linear',
                'validation_rules': {
                    'max_consecutive_missing': 4,  # 1 month
                    'min_interval_weeks': 1,
                    'max_interval_weeks': 1
                }
            },
            DataFrequency.MONTHLY: {
                'pandas_freq': 'M',
                'description': 'Monthly data',
                'max_gap_days': 365,  # 1 year
                'seasonal_period': 12,  # 1 year
                'min_data_points': 12,  # 1 year
                'interpolation_method': 'polynomial',
                'validation_rules': {
                    'max_consecutive_missing': 3,  # 3 months
                    'min_interval_months': 1,
                    'max_interval_months': 1
                }
            },
            DataFrequency.QUARTERLY: {
                'pandas_freq': 'Q',
                'description': 'Quarterly data',
                'max_gap_days': 730,  # 2 years
                'seasonal_period': 4,  # 1 year
                'min_data_points': 8,  # 2 years
                'interpolation_method': 'polynomial',
                'validation_rules': {
                    'max_consecutive_missing': 2,  # 6 months
                    'min_interval_quarters': 1,
                    'max_interval_quarters': 1
                }
            },
            DataFrequency.YEARLY: {
                'pandas_freq': 'Y',
                'description': 'Yearly data',
                'max_gap_days': 1825,  # 5 years
                'seasonal_period': 1,  # No seasonality
                'min_data_points': 5,  # 5 years
                'interpolation_method': 'polynomial',
                'validation_rules': {
                    'max_consecutive_missing': 2,  # 2 years
                    'min_interval_years': 1,
                    'max_interval_years': 1
                }
            }
        }
        
        config = configs[frequency]
        return cls(
            frequency=frequency,
            pandas_freq=config['pandas_freq'],
            description=config['description'],
            max_gap_days=config['max_gap_days'],
            seasonal_period=config['seasonal_period'],
            min_data_points=config['min_data_points'],
            interpolation_method=config['interpolation_method'],
            validation_rules=config['validation_rules']
        )


class FrequencyDetector:
    """Utility class for detecting data frequency from time series data."""
    
    @staticmethod
    def detect_frequency(dates: pd.Series) -> DataFrequency:
        """
        Detect the frequency of a time series from its dates.
        
        Args:
            dates: Series of datetime objects
            
        Returns:
            Detected DataFrequency
            
        Raises:
            ValueError: If frequency cannot be determined
        """
        if len(dates) < 2:
            raise ValueError("Need at least 2 dates to detect frequency")
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates)
        
        # Sort dates
        dates = dates.sort_values()
        
        # Calculate time differences
        time_diffs = dates.diff().dropna()
        
        if len(time_diffs) == 0:
            raise ValueError("Cannot detect frequency from single date")
        
        # Get the most common time difference
        # Convert to seconds for easier comparison
        time_diffs_seconds = pd.Series(time_diffs.total_seconds())
        most_common_diff_seconds = time_diffs_seconds.mode().iloc[0]
        total_seconds = most_common_diff_seconds
        
        # Determine frequency based on time difference
        if total_seconds <= 3600:  # <= 1 hour
            return DataFrequency.HOURLY
        elif total_seconds <= 86400:  # <= 1 day
            return DataFrequency.DAILY
        elif total_seconds <= 604800:  # <= 1 week
            return DataFrequency.WEEKLY
        elif 2419200 <= total_seconds <= 2764800:  # 28-32 days (month)
            return DataFrequency.MONTHLY
        elif 2764800 < total_seconds <= 7776000:  # >32 days up to 90 days (quarter)
            return DataFrequency.QUARTERLY
        else:
            return DataFrequency.YEARLY
    
    @staticmethod
    def validate_frequency_consistency(dates: pd.Series, expected_frequency: DataFrequency) -> bool:
        """
        Validate that the dates follow the expected frequency.
        
        Args:
            dates: Series of datetime objects
            expected_frequency: Expected DataFrequency
            
        Returns:
            True if frequency is consistent, False otherwise
        """
        if len(dates) < 2:
            return True
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates)
        
        # Sort dates
        dates = dates.sort_values()
        
        # Calculate time differences
        time_diffs = dates.diff().dropna()
        
        # Get frequency configuration
        freq_config = FrequencyConfig.create_for_frequency(expected_frequency)
        
        # Check if differences are consistent with expected frequency
        expected_seconds = {
            DataFrequency.HOURLY: 3600,
            DataFrequency.DAILY: 86400,
            DataFrequency.WEEKLY: 604800,
            DataFrequency.MONTHLY: 2592000,  # 30 days
            DataFrequency.QUARTERLY: 7776000,  # 90 days
            DataFrequency.YEARLY: 31536000  # 365 days
        }
        
        expected_seconds_value = expected_seconds[expected_frequency]
        tolerance = expected_seconds_value * 0.1  # 10% tolerance
        
        # Check if most differences are within tolerance
        consistent_diffs = time_diffs[
            (time_diffs >= pd.Timedelta(seconds=expected_seconds_value - tolerance)) &
            (time_diffs <= pd.Timedelta(seconds=expected_seconds_value + tolerance))
        ]
        
        consistency_ratio = len(consistent_diffs) / len(time_diffs)
        return consistency_ratio >= 0.8  # 80% consistency threshold


def get_frequency_config(frequency: Union[DataFrequency, str]) -> FrequencyConfig:
    """
    Get frequency configuration for a specific frequency.
    
    Args:
        frequency: DataFrequency enum or string representation
        
    Returns:
        FrequencyConfig object
    """
    return FrequencyConfig.create_for_frequency(frequency)


def detect_and_validate_frequency(dates: pd.Series, expected_frequency: Optional[Union[DataFrequency, str]] = None) -> FrequencyConfig:
    """
    Detect frequency from data and optionally validate against expected frequency.
    
    Args:
        dates: Series of datetime objects
        expected_frequency: Optional expected frequency for validation
        
    Returns:
        FrequencyConfig object for detected frequency
        
    Raises:
        ValueError: If frequency validation fails
    """
    detected_frequency = FrequencyDetector.detect_frequency(dates)
    
    if expected_frequency is not None:
        if isinstance(expected_frequency, str):
            expected_frequency = DataFrequency.from_string(expected_frequency)
        
        if detected_frequency != expected_frequency:
            # Validate consistency with expected frequency
            if not FrequencyDetector.validate_frequency_consistency(dates, expected_frequency):
                raise ValueError(
                    f"Frequency mismatch: detected {detected_frequency.value} but expected {expected_frequency.value}. "
                    f"Data is not consistent with expected frequency."
                )
            else:
                # Use expected frequency if data is consistent
                detected_frequency = expected_frequency
    
    return FrequencyConfig.create_for_frequency(detected_frequency)


def get_supported_frequencies() -> list:
    """Get list of supported frequency strings."""
    return [freq.value for freq in DataFrequency]


def get_frequency_description(frequency: Union[DataFrequency, str]) -> str:
    """Get human-readable description of a frequency."""
    config = get_frequency_config(frequency)
    return config.description 