"""
Meteorological Data Model

This module defines the MeteorologicalData model for representing
structured meteorological data in the weather prediction system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np

from src.config.column_config import ColumnMapping, ColumnType, get_column_mapping


@dataclass
class MeteorologicalData:
    """
    Model for representing meteorological data in a structured format.
    
    This model encapsulates meteorological data with metadata and provides
    methods for data validation and transformation.
    """
    
    # Core data
    data: pd.DataFrame
    variable_type: str
    unit: str
    
    # Metadata
    source_file: str
    processing_date: datetime = field(default_factory=datetime.now)
    
    # Data characteristics
    date_range: Dict[str, Any] = field(default_factory=dict)
    stations: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Column mapping (will be detected automatically)
    column_mapping: Optional[ColumnMapping] = None
    
    def __post_init__(self):
        """Validate and initialize the data model after creation."""
        self._validate_data()
        self._detect_column_mapping()
        self._extract_metadata()
    
    def _validate_data(self):
        """Validate the input data."""
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        
        if self.data.empty:
            raise ValueError("Data cannot be empty")
    
    def _detect_column_mapping(self):
        """Detect column mapping for the data."""
        try:
            self.column_mapping = get_column_mapping(self.data, self.variable_type)
        except Exception as e:
            raise ValueError(f"Failed to detect column mapping: {e}")
    
    def _extract_metadata(self):
        """Extract metadata from the data."""
        if not self.column_mapping:
            raise ValueError("Column mapping not available")
        
        # Extract date range
        date_column = self.column_mapping.date_column
        if date_column in self.data.columns:
            dates = pd.to_datetime(self.data[date_column])
            self.date_range = {
                'start': dates.min(),
                'end': dates.max()
            }
        
        # Extract stations
        station_column = self.column_mapping.station_column
        if station_column in self.data.columns:
            self.stations = self.data[station_column].unique().tolist()
        
        # Extract statistics
        value_column = self.column_mapping.value_column
        if value_column in self.data.columns:
            value_data = pd.to_numeric(self.data[value_column], errors='coerce')
            self.statistics = {
                'min': float(value_data.min()) if not pd.isna(value_data.min()) else None,
                'max': float(value_data.max()) if not pd.isna(value_data.max()) else None,
                'mean': float(value_data.mean()) if not pd.isna(value_data.mean()) else None,
                'std': float(value_data.std()) if not pd.isna(value_data.std()) else None,
                'missing_count': int(value_data.isnull().sum()),
                'total_count': int(len(value_data))
            }
    
    def get_data_by_station(self, station_name: str) -> pd.DataFrame:
        """
        Get data filtered by station name.
        
        Args:
            station_name: Name of the station to filter by
            
        Returns:
            Filtered DataFrame
        """
        if not self.column_mapping:
            raise ValueError("Column mapping not available")
        
        station_column = self.column_mapping.station_column
        return self.data[self.data[station_column] == station_name].copy()
    
    def get_data_by_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get data filtered by date range.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Filtered DataFrame
        """
        if not self.column_mapping:
            raise ValueError("Column mapping not available")
        
        date_column = self.column_mapping.date_column
        dates = pd.to_datetime(self.data[date_column])
        mask = (dates >= start_date) & (dates <= end_date)
        return self.data[mask].copy()
    
    def get_missing_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of missing data.
        
        Returns:
            Dictionary with missing data statistics
        """
        missing_summary = {}
        
        for column in self.data.columns:
            missing_count = self.data[column].isnull().sum()
            total_count = len(self.data[column])
            missing_percentage = (missing_count / total_count) * 100 if total_count > 0 else 0
            
            missing_summary[column] = {
                'missing_count': missing_count,
                'total_count': total_count,
                'missing_percentage': missing_percentage
            }
        
        return missing_summary
    
    def get_station_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for each station.
        
        Returns:
            Dictionary with statistics per station
        """
        if not self.column_mapping:
            raise ValueError("Column mapping not available")
        
        station_stats = {}
        value_column = self.column_mapping.value_column
        
        for station in self.stations:
            station_data = self.get_data_by_station(station)
            value_data = pd.to_numeric(station_data[value_column], errors='coerce')
            
            station_stats[station] = {
                'min': float(value_data.min()) if not pd.isna(value_data.min()) else None,
                'max': float(value_data.max()) if not pd.isna(value_data.max()) else None,
                'mean': float(value_data.mean()) if not pd.isna(value_data.mean()) else None,
                'std': float(value_data.std()) if not pd.isna(value_data.std()) else None,
                'count': int(len(value_data)),
                'missing_count': int(value_data.isnull().sum())
            }
        
        return station_stats
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary representation.
        
        Returns:
            Dictionary representation of the model
        """
        result = {
            'variable_type': self.variable_type,
            'unit': self.unit,
            'source_file': self.source_file,
            'processing_date': self.processing_date.isoformat(),
            'date_range': {
                'start': self.date_range['start'].isoformat() if self.date_range.get('start') else None,
                'end': self.date_range['end'].isoformat() if self.date_range.get('end') else None
            },
            'stations': self.stations,
            'statistics': self.statistics,
            'config': self.config,
            'data_shape': self.data.shape,
            'columns': list(self.data.columns)
        }
        
        # Add column mapping information if available
        if self.column_mapping:
            result['column_mapping'] = {
                'date_column': self.column_mapping.date_column,
                'station_column': self.column_mapping.station_column,
                'value_column': self.column_mapping.value_column,
                'year_column': self.column_mapping.year_column,
                'month_column': self.column_mapping.month_column,
                'code_column': self.column_mapping.code_column,
                'day_prefix': self.column_mapping.day_prefix
            }
        
        return result
    
    @classmethod
    def from_dataframe(cls, 
                      data: pd.DataFrame, 
                      variable_type: str, 
                      unit: str, 
                      source_file: str,
                      config: Optional[Dict[str, Any]] = None) -> 'MeteorologicalData':
        """
        Create a MeteorologicalData instance from a DataFrame.
        
        Args:
            data: Input DataFrame
            variable_type: Type of meteorological variable
            unit: Unit of measurement
            source_file: Source file path
            config: Optional configuration dictionary
            
        Returns:
            MeteorologicalData instance
        """
        return cls(
            data=data,
            variable_type=variable_type,
            unit=unit,
            source_file=source_file,
            config=config or {}
        )
    
    def __len__(self) -> int:
        """Return the number of rows in the data."""
        return len(self.data)
    
    def __str__(self) -> str:
        """String representation of the model."""
        return (f"MeteorologicalData(variable_type='{self.variable_type}', "
                f"unit='{self.unit}', shape={self.data.shape}, "
                f"stations={len(self.stations)}, date_range={self.date_range})") 