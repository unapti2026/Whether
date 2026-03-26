"""
Preprocessing Service Module

This module provides services for preprocessing meteorological data,
including data cleaning, validation, and preparation for analysis.
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from src.core.exceptions import DataProcessingError
from src.config.settings import get_config_for_variable, get_file_paths_for_variable
from src.data.processors.meteorological_processor import MeteorologicalDataProcessor


class ProcessingStage(Enum):
    """Enumeration for data processing stages."""
    RAW = "raw"
    CLEANED = "cleaned"
    PROCESSED = "processed"


class ProcessingStatus(Enum):
    """Enumeration for processing status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingConfig:
    """Configuration class for preprocessing parameters."""
    variable_type: str
    input_path: Path
    output_path: Optional[Path] = None
    custom_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        if self.output_path is None:
            self.output_path = self._generate_output_path()
    
    def _generate_output_path(self) -> Path:
        """Generate output path based on input file name and variable type."""
        clean_dir = get_file_paths_for_variable(self.variable_type)['output_clean']
        clean_dir.mkdir(parents=True, exist_ok=True)
        
        input_stem = self.input_path.stem
        output_filename = f"restructured_{input_stem}.csv"
        return clean_dir / output_filename


@dataclass
class ProcessingSummary:
    """Data class for processing summary information."""
    status: ProcessingStatus
    input_file: str
    output_file: str
    variable_type: str
    processing_steps: List[str]
    data_statistics: Dict[str, Any]
    date_range: Dict[str, str]
    stations: Dict[str, Any]
    value_statistics: Dict[str, Any]
    processing_time: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'status': self.status.value,
            'input_file': self.input_file,
            'output_file': self.output_file,
            'variable_type': self.variable_type,
            'processing_steps': self.processing_steps,
            'data_statistics': self.data_statistics,
            'date_range': self.date_range,
            'stations': self.stations,
            'value_statistics': self.value_statistics,
            'processing_time': self.processing_time,
            'error_message': self.error_message
        }


@dataclass
class DataStatistics:
    """Data class for data statistics."""
    raw_rows: int
    cleaned_rows: int
    processed_rows: int
    final_columns: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'raw_rows': self.raw_rows,
            'cleaned_rows': self.cleaned_rows,
            'processed_rows': self.processed_rows,
            'final_columns': self.final_columns
        }


@dataclass
class ValueStatistics:
    """Data class for value statistics."""
    min_value: float
    max_value: float
    mean_value: float
    missing_values: int
    total_values: int
    
    @property
    def completeness_rate(self) -> float:
        """Calculate data completeness rate."""
        if self.total_values == 0:
            return 0.0
        return ((self.total_values - self.missing_values) / self.total_values) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'min': self.min_value,
            'max': self.max_value,
            'mean': self.mean_value,
            'missing_values': self.missing_values,
            'total_values': self.total_values,
            'completeness_rate': round(self.completeness_rate, 2)
        }


class PreprocessingService:
    """
    Professional service for coordinating meteorological data preprocessing pipeline.
    
    This service orchestrates the complete preprocessing workflow with robust
    error handling, comprehensive logging, and detailed reporting capabilities.
    
    Attributes:
        config (ProcessingConfig): Configuration for the preprocessing process
        processor (MeteorologicalDataProcessor): The data processor instance
        logger (logging.Logger): Logger instance for monitoring
        processing_summary (Optional[ProcessingSummary]): Summary of processing results
    """
    
    def __init__(self, 
                 input_path: Union[str, Path],
                 variable_type: str = 'temperature',
                 output_path: Optional[Union[str, Path]] = None,
                 custom_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessing service.
        
        Args:
            input_path: Path to the CSV file containing meteorological data
            variable_type: Type of meteorological variable ('temperature', 'precipitation', etc.)
            output_path: Path where to save processed data (auto-generated if None)
            custom_config: Custom configuration dictionary (uses default if None)
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the input path is invalid
        """
        # Initialize configuration
        self.config = ProcessingConfig(
            variable_type=variable_type,
            input_path=Path(input_path),
            output_path=Path(output_path) if output_path else None,
            custom_config=custom_config
        )
        
        # Set up logger
        self.logger = self._setup_logger()
        
        # Get processing configuration
        self.processing_config = self._get_processing_config()
        
        # Initialize processor
        self.processor = self._initialize_processor()
        
        # Processing state
        self.processing_summary: Optional[ProcessingSummary] = None
        
        self._log_initialization()
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger for the service instance.
        
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
    
    def _get_processing_config(self) -> Dict[str, Any]:
        """
        Get processing configuration for the variable type.
        
        Returns:
            Configuration dictionary for processing
        """
        if self.config.custom_config:
            return self.config.custom_config
        return get_config_for_variable(self.config.variable_type)
    
    def _initialize_processor(self) -> MeteorologicalDataProcessor:
        """
        Initialize the meteorological data processor.
        
        Returns:
            Configured MeteorologicalDataProcessor instance
        """
        return MeteorologicalDataProcessor(
            str(self.config.input_path), 
            self.processing_config
        )
    
    def _log_initialization(self) -> None:
        """Log initialization information."""
        self.logger.info(f"PreprocessingService initialized for {self.config.variable_type} data")
        self.logger.info(f"Input: {self.config.input_path}")
        self.logger.info(f"Output: {self.config.output_path}")
        self.logger.info(f"Configuration: {len(self.processing_config)} parameters")
    
    def process_data(self) -> Dict[str, Any]:
        """
        Execute the complete preprocessing pipeline.
        
        This method orchestrates the complete preprocessing workflow:
        1. Data loading and validation
        2. Data cleaning
        3. Data restructuring (monthly to time series)
        4. Data saving and reporting
        
        Returns:
            Dictionary containing processing results and statistics
            
        Raises:
            DataProcessingError: If any step in the pipeline fails
        """
        start_time = datetime.now()
        self.logger.info("Starting preprocessing pipeline")
        
        try:
            # Execute processing steps
            cleaned_data = self._execute_cleaning_step()
            processed_data = self._execute_processing_step()
            self._execute_saving_step()
            
            # Generate comprehensive summary
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_summary = self._generate_processing_summary(
                cleaned_data, processed_data, processing_time
            )
            
            self.logger.info("Preprocessing pipeline completed successfully")
            return self.processing_summary.to_dict()
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in preprocessing pipeline: {e}")
            
            # Create failure summary
            self.processing_summary = self._create_failure_summary(processing_time, str(e))
            
            raise DataProcessingError(f"Preprocessing pipeline failed: {e}") from e
    
    def _execute_cleaning_step(self) -> pd.DataFrame:
        """
        Execute the data cleaning step.
        
        Returns:
            Cleaned DataFrame
            
        Raises:
            DataProcessingError: If cleaning fails
        """
        try:
            self.logger.info("Step 1: Cleaning data")
            cleaned_data = self.processor.clean_data()
            self.logger.info(f"Data cleaning completed: {len(cleaned_data)} rows")
            return cleaned_data
        except Exception as e:
            self.logger.error(f"Error in data cleaning step: {e}")
            raise DataProcessingError(f"Data cleaning failed: {e}") from e
    
    def _execute_processing_step(self) -> pd.DataFrame:
        """
        Execute the data processing/restructuring step.
        
        Returns:
            Processed DataFrame
            
        Raises:
            DataProcessingError: If processing fails
        """
        try:
            self.logger.info("Step 2: Restructuring data")
            processed_data = self.processor.process_data()
            self.logger.info(f"Data restructuring completed: {len(processed_data)} rows")
            return processed_data
        except Exception as e:
            self.logger.error(f"Error in data processing step: {e}")
            raise DataProcessingError(f"Data processing failed: {e}") from e
    
    def _execute_saving_step(self) -> None:
        """
        Execute the data saving step.
        
        Raises:
            DataProcessingError: If saving fails
        """
        try:
            self.logger.info("Step 3: Saving processed data")
            if self.config.output_path:
                self.processor.save_data(str(self.config.output_path))
                self.logger.info(f"Data saved to: {self.config.output_path}")
        except Exception as e:
            self.logger.error(f"Error in data saving step: {e}")
            raise DataProcessingError(f"Data saving failed: {e}") from e
    
    def _generate_processing_summary(self, 
                                   cleaned_data: pd.DataFrame, 
                                   processed_data: pd.DataFrame,
                                   processing_time: float) -> ProcessingSummary:
        """
        Generate a comprehensive summary of the processing results.
        
        Args:
            cleaned_data: DataFrame after cleaning step
            processed_data: DataFrame after processing step
            processing_time: Total processing time in seconds
            
        Returns:
            ProcessingSummary object with comprehensive statistics
        """
        # Calculate data statistics
        data_stats = self._calculate_data_statistics(cleaned_data, processed_data)
        
        # Calculate value statistics
        value_stats = self._calculate_value_statistics(processed_data)
        
        # Calculate date range
        date_range = self._calculate_date_range(processed_data)
        
        # Calculate station information
        stations = self._calculate_station_information(processed_data)
        
        return ProcessingSummary(
            status=ProcessingStatus.COMPLETED,
            input_file=str(self.config.input_path),
            output_file=str(self.config.output_path),
            variable_type=self.config.variable_type,
            processing_steps=['load', 'clean', 'restructure', 'save'],
            data_statistics=data_stats.to_dict(),
            date_range=date_range,
            stations=stations,
            value_statistics=value_stats.to_dict(),
            processing_time=processing_time
        )
    
    def _create_failure_summary(self, processing_time: float, error_message: str) -> ProcessingSummary:
        """
        Create a summary for failed processing.
        
        Args:
            processing_time: Processing time before failure
            error_message: Error message describing the failure
            
        Returns:
            ProcessingSummary object for failed processing
        """
        return ProcessingSummary(
            status=ProcessingStatus.FAILED,
            input_file=str(self.config.input_path),
            output_file=str(self.config.output_path),
            variable_type=self.config.variable_type,
            processing_steps=[],
            data_statistics={},
            date_range={},
            stations={},
            value_statistics={},
            processing_time=processing_time,
            error_message=error_message
        )
    
    def _calculate_data_statistics(self, cleaned_data: pd.DataFrame, 
                                 processed_data: pd.DataFrame) -> DataStatistics:
        """
        Calculate comprehensive data statistics.
        
        Args:
            cleaned_data: DataFrame after cleaning
            processed_data: DataFrame after processing
            
        Returns:
            DataStatistics object with calculated statistics
        """
        raw_data = self.processor.data.get('raw')
        raw_rows = len(raw_data) if raw_data is not None else 0
        
        return DataStatistics(
            raw_rows=raw_rows,
            cleaned_rows=len(cleaned_data),
            processed_rows=len(processed_data),
            final_columns=list(processed_data.columns)
        )
    
    def _calculate_value_statistics(self, processed_data: pd.DataFrame) -> ValueStatistics:
        """
        Calculate value statistics for the processed data.
        
        Args:
            processed_data: DataFrame after processing
            
        Returns:
            ValueStatistics object with calculated statistics
        """
        value_column = self.processing_config['value_column']
        value_series = processed_data[value_column]
        
        return ValueStatistics(
            min_value=float(value_series.min()),
            max_value=float(value_series.max()),
            mean_value=float(value_series.mean()),
            missing_values=int(value_series.isnull().sum()),
            total_values=len(value_series)
        )
    
    def _calculate_date_range(self, processed_data: pd.DataFrame) -> Dict[str, str]:
        """
        Calculate the date range of the processed data.
        
        Args:
            processed_data: DataFrame after processing
            
        Returns:
            Dictionary with start and end dates
        """
        date_column = self.processing_config['date_column']
        date_series = processed_data[date_column]
        
        return {
            'start': date_series.min().strftime('%Y-%m-%d'),
            'end': date_series.max().strftime('%Y-%m-%d')
        }
    
    def _calculate_station_information(self, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate station information for the processed data.
        
        Args:
            processed_data: DataFrame after processing
            
        Returns:
            Dictionary with station information
        """
        station_column = self.processing_config['station_column']
        station_series = processed_data[station_column]
        
        return {
            'count': int(station_series.nunique()),
            'names': station_series.unique().tolist()
        }
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the processed data from the processor.
        
        Returns:
            Processed DataFrame
            
        Raises:
            ValueError: If data hasn't been processed yet
        """
        if self.processor.data['processed'].empty:
            raise ValueError("Data hasn't been processed yet. Run process_data() first.")
        
        return self.processor.get_data_stage('processed')
    
    def get_data_by_station(self, station_name: str) -> pd.DataFrame:
        """
        Get processed data filtered by station name.
        
        Args:
            station_name: Name of the station to filter by
            
        Returns:
            Filtered DataFrame containing only data from the specified station
            
        Raises:
            ValueError: If data hasn't been processed yet
        """
        if self.processor.data['processed'].empty:
            raise ValueError("Data hasn't been processed yet. Run process_data() first.")
        
        return self.processor.get_data_by_station(station_name)
    
    def get_data_by_code(self, code: int) -> pd.DataFrame:
        """
        Get processed data filtered by station code.
        
        Args:
            code: Station code to filter by
            
        Returns:
            Filtered DataFrame containing only data from the specified station code
            
        Raises:
            ValueError: If data hasn't been processed yet
        """
        if self.processor.data['processed'].empty:
            raise ValueError("Data hasn't been processed yet. Run process_data() first.")
        
        return self.processor.get_data_by_code(code)
    
    def prepare_data_for_plotting(self, 
                                 data: Optional[pd.DataFrame] = None,
                                 date_column: Optional[str] = None,
                                 value_column: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare data for plotting by ensuring proper data types and sorting.
        
        Args:
            data: DataFrame to prepare (uses processed data if None)
            date_column: Name of the date column (uses configured date_column if None)
            value_column: Name of the value column (uses configured value_column if None)
            
        Returns:
            Prepared DataFrame ready for plotting
            
        Raises:
            ValueError: If data hasn't been processed yet
        """
        if data is None and self.processor.data['processed'].empty:
            raise ValueError("Data hasn't been processed yet. Run process_data() first.")
        
        return self.processor.prepare_data_for_plotting(data, date_column, value_column)
    
    def get_processing_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get the processing summary if available.
        
        Returns:
            Processing summary dictionary or None if not available
        """
        if self.processing_summary:
            return self.processing_summary.to_dict()
        return None
    
    def get_processing_status(self) -> ProcessingStatus:
        """
        Get the current processing status.
        
        Returns:
            Current processing status
        """
        if self.processing_summary:
            return self.processing_summary.status
        return ProcessingStatus.PENDING
    
    def reset(self) -> None:
        """Reset the service state."""
        self.processing_summary = None
        self.logger.info("Service state reset") 