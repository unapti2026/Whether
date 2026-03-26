"""
Data Repository

This module defines the DataRepository class for handling
data access operations for meteorological data.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import pandas as pd
import logging

from ..models.meteorological_data import MeteorologicalData
from ..models.processing_result import ProcessingResult
from ...core.exceptions import DataProcessingError, ValidationError


class DataRepository(ABC):
    """
    Abstract repository for data access operations.
    
    This repository provides an abstraction layer for accessing
    meteorological data from various sources.
    """
    
    def __init__(self):
        """Initialize the data repository."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def load_data(self, source: str, **kwargs) -> MeteorologicalData:
        """
        Load meteorological data from a source.
        
        Args:
            source: Data source identifier
            **kwargs: Additional loading parameters
            
        Returns:
            MeteorologicalData instance
            
        Raises:
            DataProcessingError: If loading fails
        """
        pass
    
    @abstractmethod
    def save_data(self, data: MeteorologicalData, destination: str, **kwargs) -> bool:
        """
        Save meteorological data to a destination.
        
        Args:
            data: MeteorologicalData to save
            destination: Destination identifier
            **kwargs: Additional saving parameters
            
        Returns:
            True if saving was successful, False otherwise
            
        Raises:
            DataProcessingError: If saving fails
        """
        pass
    
    @abstractmethod
    def get_data_info(self, source: str) -> Dict[str, Any]:
        """
        Get information about data at a source.
        
        Args:
            source: Data source identifier
            
        Returns:
            Dictionary with data information
            
        Raises:
            DataProcessingError: If information retrieval fails
        """
        pass
    
    @abstractmethod
    def validate_source(self, source: str) -> bool:
        """
        Validate if a data source is accessible.
        
        Args:
            source: Data source identifier
            
        Returns:
            True if source is valid and accessible, False otherwise
        """
        pass


class FileDataRepository(DataRepository):
    """
    File-based implementation of DataRepository.
    
    This repository handles data access operations for file-based
    meteorological data sources (CSV, Excel, etc.).
    """
    
    def __init__(self, base_path: Union[str, Path] = "data"):
        """
        Initialize the file data repository.
        
        Args:
            base_path: Base path for data files
        """
        super().__init__()
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized FileDataRepository with base path: {self.base_path}")
    
    def load_data(self, source: str, **kwargs) -> MeteorologicalData:
        """
        Load meteorological data from a file.
        
        Args:
            source: File path or name
            **kwargs: Additional loading parameters
            
        Returns:
            MeteorologicalData instance
            
        Raises:
            DataProcessingError: If loading fails
        """
        try:
            file_path = self._resolve_file_path(source)
            
            if not file_path.exists():
                raise DataProcessingError(
                    f"Data file not found: {file_path}",
                    operation="load_data",
                    data_info={"source": source, "file_path": str(file_path)}
                )
            
            # Determine file type and load accordingly
            if file_path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path, **kwargs)
            else:
                raise DataProcessingError(
                    f"Unsupported file type: {file_path.suffix}",
                    operation="load_data",
                    data_info={"source": source, "file_path": str(file_path)}
                )
            
            # Extract variable type from file name or kwargs
            variable_type = kwargs.get('variable_type', self._extract_variable_type(file_path.name))
            unit = kwargs.get('unit', self._get_unit_for_variable(variable_type))
            
            self.logger.info(f"Successfully loaded data from {file_path}: {data.shape}")
            
            return MeteorologicalData.from_dataframe(
                data=data,
                variable_type=variable_type,
                unit=unit,
                source_file=str(file_path),
                config=kwargs.get('config', {})
            )
            
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(
                f"Error loading data from {source}: {str(e)}",
                operation="load_data",
                data_info={"source": source, "error": str(e)}
            ) from e
    
    def save_data(self, data: MeteorologicalData, destination: str, **kwargs) -> bool:
        """
        Save meteorological data to a file.
        
        Args:
            data: MeteorologicalData to save
            destination: File path or name
            **kwargs: Additional saving parameters
            
        Returns:
            True if saving was successful, False otherwise
            
        Raises:
            DataProcessingError: If saving fails
        """
        try:
            file_path = self._resolve_file_path(destination)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine file type and save accordingly
            if file_path.suffix.lower() == '.csv':
                data.data.to_csv(file_path, index=False, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                data.data.to_excel(file_path, index=False, **kwargs)
            else:
                raise DataProcessingError(
                    f"Unsupported file type: {file_path.suffix}",
                    operation="save_data",
                    data_info={"destination": destination, "file_path": str(file_path)}
                )
            
            self.logger.info(f"Successfully saved data to {file_path}: {data.data.shape}")
            return True
            
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(
                f"Error saving data to {destination}: {str(e)}",
                operation="save_data",
                data_info={"destination": destination, "error": str(e)}
            ) from e
    
    def get_data_info(self, source: str) -> Dict[str, Any]:
        """
        Get information about data at a source.
        
        Args:
            source: File path or name
            
        Returns:
            Dictionary with data information
            
        Raises:
            DataProcessingError: If information retrieval fails
        """
        try:
            file_path = self._resolve_file_path(source)
            
            if not file_path.exists():
                raise DataProcessingError(
                    f"Data file not found: {file_path}",
                    operation="get_data_info",
                    data_info={"source": source, "file_path": str(file_path)}
                )
            
            # Load a sample to get information
            sample_data = self.load_data(source, nrows=1000)
            
            info = {
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'last_modified': file_path.stat().st_mtime,
                'variable_type': sample_data.variable_type,
                'unit': sample_data.unit,
                'columns': list(sample_data.data.columns),
                'sample_shape': sample_data.data.shape,
                'date_range': sample_data.date_range,
                'stations': sample_data.stations[:10],  # First 10 stations
                'statistics': sample_data.statistics
            }
            
            return info
            
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(
                f"Error getting data info for {source}: {str(e)}",
                operation="get_data_info",
                data_info={"source": source, "error": str(e)}
            ) from e
    
    def validate_source(self, source: str) -> bool:
        """
        Validate if a data source is accessible.
        
        Args:
            source: File path or name
            
        Returns:
            True if source is valid and accessible, False otherwise
        """
        try:
            file_path = self._resolve_file_path(source)
            return file_path.exists() and file_path.is_file()
        except Exception:
            return False
    
    def _resolve_file_path(self, source: str) -> Path:
        """
        Resolve a source identifier to a file path.
        
        Args:
            source: Source identifier (file name or path)
            
        Returns:
            Resolved file path
        """
        source_path = Path(source)
        
        # If it's already an absolute path, return as is
        if source_path.is_absolute():
            return source_path
        
        # If it's a relative path, resolve against base path
        return self.base_path / source_path
    
    def _extract_variable_type(self, filename: str) -> str:
        """
        Extract variable type from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Extracted variable type
        """
        filename_lower = filename.lower()
        
        if 'temp_min' in filename_lower or 'minima' in filename_lower:
            return 'temp_min'
        elif 'temp_max' in filename_lower or 'maxima' in filename_lower:
            return 'temp_max'
        elif 'precip' in filename_lower or 'lluvia' in filename_lower:
            return 'precipitation'
        elif 'humidity' in filename_lower or 'humedad' in filename_lower:
            return 'humidity'
        else:
            return 'temperature'  # Default
    
    def _get_unit_for_variable(self, variable_type: str) -> str:
        """
        Get the unit for a variable type.
        
        Args:
            variable_type: Type of meteorological variable
            
        Returns:
            Unit of measurement
        """
        units = {
            'temp_min': '°C',
            'temp_max': '°C',
            'temperature': '°C',
            'precipitation': 'mm',
            'humidity': '%'
        }
        
        return units.get(variable_type, '')
    
    def list_available_sources(self, pattern: str = "*") -> List[str]:
        """
        List available data sources matching a pattern.
        
        Args:
            pattern: File pattern to match (default: all files)
            
        Returns:
            List of available source identifiers
        """
        try:
            files = list(self.base_path.glob(pattern))
            return [str(f.relative_to(self.base_path)) for f in files if f.is_file()]
        except Exception as e:
            self.logger.error(f"Error listing available sources: {e}")
            return [] 