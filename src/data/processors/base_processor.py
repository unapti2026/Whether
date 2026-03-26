"""
Base Data Processor Module

This module provides the abstract base class for all data processors
in the weather prediction system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import pandas as pd
import logging
from pathlib import Path


class BaseDataProcessor(ABC):
    """
    Abstract base class for data processors.
    
    This class defines the interface that all data processors must implement.
    It provides common functionality for data loading, validation, and processing.
    
    Attributes:
        file_path (Path): Path to the input data file
        config (Dict[str, Any]): Configuration parameters for processing
        logger (logging.Logger): Logger instance for debugging and monitoring
        data (Dict[str, pd.DataFrame]): Dictionary containing different stages of processed data
    """
    
    def __init__(self, file_path: Union[str, Path], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base data processor.
        
        Args:
            file_path: Path to the input data file
            config: Optional configuration dictionary with processing parameters
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the file path is invalid
        """
        self.file_path = Path(file_path)
        self.config = config or {}
        self.logger = self._setup_logger()
        self.data: Dict[str, pd.DataFrame] = {
            'raw': pd.DataFrame(),
            'cleaned': pd.DataFrame(),
            'processed': pd.DataFrame()
        }
        
        self._validate_file_path()
        self._load_data()
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger for the processor instance.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _validate_file_path(self) -> None:
        """
        Validate that the input file exists and is accessible.
        
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file path is invalid
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.file_path}")
        
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")
    
    def _load_data(self) -> None:
        """
        Load raw data from the input file.
        
        This method should be implemented by subclasses to handle
        specific file formats and loading requirements.
        """
        try:
            self.logger.info(f"Loading data from: {self.file_path}")
            self.data['raw'] = self._load_raw_data()
            self.logger.info(f"Successfully loaded {len(self.data['raw'])} rows")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    @abstractmethod
    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from file. Must be implemented by subclasses.
        
        Returns:
            Raw data as pandas DataFrame
        """
        pass
    
    @abstractmethod
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the raw data. Must be implemented by subclasses.
        
        Returns:
            Cleaned data as pandas DataFrame
        """
        pass
    
    @abstractmethod
    def process_data(self) -> pd.DataFrame:
        """
        Process the cleaned data. Must be implemented by subclasses.
        
        Returns:
            Processed data as pandas DataFrame
        """
        pass
    
    def get_data_stage(self, stage: str) -> pd.DataFrame:
        """
        Get data from a specific processing stage.
        
        Args:
            stage: Stage name ('raw', 'cleaned', 'processed')
            
        Returns:
            DataFrame from the specified stage
            
        Raises:
            KeyError: If the stage doesn't exist
        """
        if stage not in self.data:
            raise KeyError(f"Unknown data stage: {stage}. Available stages: {list(self.data.keys())}")
        return self.data[stage].copy()
    
    def save_data(self, output_path: Union[str, Path], stage: str = 'processed') -> None:
        """
        Save data from a specific stage to file.
        
        Args:
            output_path: Path where to save the data
            stage: Stage name to save ('raw', 'cleaned', 'processed')
            
        Raises:
            KeyError: If the stage doesn't exist
            IOError: If there's an error saving the file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data_to_save = self.get_data_stage(stage)
            data_to_save.to_csv(output_path, index=False)
            self.logger.info(f"Data saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the processing results.
        
        Returns:
            Dictionary with processing statistics and metadata
        """
        summary = {
            'input_file': str(self.file_path),
            'stages_processed': list(self.data.keys()),
            'raw_data_shape': self.data['raw'].shape if not self.data['raw'].empty else (0, 0),
            'processed_data_shape': self.data['processed'].shape if not self.data['processed'].empty else (0, 0)
        }
        
        if not self.data['processed'].empty:
            summary['processing_stats'] = {
                'total_rows': len(self.data['processed']),
                'columns': list(self.data['processed'].columns),
                'missing_values': self.data['processed'].isnull().sum().to_dict()
            }
        
        return summary 