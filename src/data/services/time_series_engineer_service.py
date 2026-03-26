"""
Time Series Engineer Service

This service handles the conversion of Excel files to CSV and coordinates
the data processing pipeline for time series meteorological data.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.data.processors.meteorological_processor import MeteorologicalDataProcessor
from src.config.settings import get_config_for_variable, get_file_paths_for_variable, get_supported_variables

logger = logging.getLogger(__name__)

class TimeSeriesEngineerService:
    """
    Service for converting Excel files to CSV and processing meteorological time series data.
    
    This service maintains compatibility with the original workflow while using
    the new modular architecture and supporting multiple meteorological variables.
    """
    
    def __init__(self, variable_type: str, data_path: str = "data"):
        """
        Initialize the TimeSeriesEngineerService.
        
        Args:
            variable_type: Type of meteorological variable ('temp_min', 'temp_max', 'precipitation', 'humidity')
            data_path: Directory path containing the data files (default: "data")
        """
        # Validate variable type
        supported_vars = get_supported_variables()
        if variable_type not in supported_vars:
            raise ValueError(f"Unsupported variable type: {variable_type}. Supported: {supported_vars}")
        
        self.variable_type = variable_type
        self.data_path = Path(data_path)
        
        # Get configuration for this variable
        self.config = get_config_for_variable(variable_type)
        self.file_paths = get_file_paths_for_variable(variable_type)
        
        # Setup logging
        logger.info(f"Initialized TimeSeriesEngineerService for {variable_type}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Configuration: {len(self.config)} parameters")
        
        # Convert Excel to CSV if needed
        self.csv_file_name = self._convert_excel_to_csv()
        
        # Initialize the data processor with the new architecture
        csv_path = str(self.file_paths['input_csv'])
        self.data_processor = MeteorologicalDataProcessor(csv_path, self.config)
        
    def _convert_excel_to_csv(self) -> str:
        """
        Convert Excel file to CSV format.
        
        Returns:
            Name of the created CSV file
        """
        excel_path = self.file_paths['input_excel']
        csv_path = self.file_paths['input_csv']
        
        # Check if Excel file exists
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
        # Convert Excel to CSV if CSV doesn't exist or Excel is newer
        if not csv_path.exists() or excel_path.stat().st_mtime > csv_path.stat().st_mtime:
            logger.info(f"Converting {excel_path.name} to CSV format...")
            
            try:
                # Read Excel file
                data = pd.read_excel(excel_path)
                
                # Save as CSV
                data.to_csv(csv_path, index=False, encoding='utf-8')
                
                logger.info(f"Successfully converted to {csv_path.name}")
                print(f"Archivo convertido y guardado como {csv_path}")
                
            except Exception as e:
                logger.error(f"Error converting Excel to CSV: {e}")
                raise
        else:
            logger.info(f"CSV file already exists and is up to date: {csv_path.name}")
        
        return csv_path.name
    
    def process_data(self) -> Dict[str, Any]:
        """
        Process the data through the complete pipeline.
        
        Returns:
            Dictionary with processing results and statistics
        """
        logger.info(f"Starting data processing pipeline for {self.variable_type}...")
        
        try:
            # Clean raw data
            logger.info("Step 1: Cleaning raw data...")
            self.data_processor.clean_data()
            
            # Process data (convert to time series)
            logger.info("Step 2: Processing data...")
            self.data_processor.process_data()
            
            # Save processed data
            output_file = str(self.file_paths['output_processed'])
            logger.info(f"Step 3: Saving processed data to {output_file}")
            self.data_processor.save_data(output_file)
            
            # Generate processing summary
            summary = self._generate_processing_summary(output_file)
            
            logger.info("Data processing pipeline completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error during data processing: {e}")
            raise
    
    def _generate_processing_summary(self, output_file: str) -> Dict[str, Any]:
        """
        Generate a summary of the processing results.
        
        Args:
            output_file: Path to the output file
            
        Returns:
            Dictionary with processing summary
        """
        try:
            # Read the processed data for statistics
            processed_data = pd.read_csv(output_file)
            
            summary = {
                'variable_type': self.variable_type,
                'display_name': self.config['display_name'],
                'unit': self.config['unit'],
                'input_file': str(self.file_paths['input_excel']),
                'output_file': output_file,
                'csv_file': str(self.file_paths['input_csv']),
                'processing_status': 'completed',
                'data_statistics': {
                    'total_rows': len(processed_data),
                    'columns': list(processed_data.columns),
                    'date_range': {
                        'start': processed_data[self.config['date_column']].min(),
                        'end': processed_data[self.config['date_column']].max()
                    },
                    'stations': {
                        'count': processed_data[self.config['station_column']].nunique(),
                        'names': processed_data[self.config['station_column']].unique().tolist()
                    },
                    'value_statistics': {
                        'min': processed_data[self.config['value_column']].min(),
                        'max': processed_data[self.config['value_column']].max(),
                        'mean': processed_data[self.config['value_column']].mean(),
                        'missing_values': processed_data[self.config['value_column']].isnull().sum()
                    }
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating processing summary: {e}")
            return {
                'variable_type': self.variable_type,
                'input_file': str(self.file_paths['input_excel']),
                'output_file': output_file,
                'processing_status': 'completed_with_warnings',
                'error': str(e)
            }
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the processed data as a DataFrame.
        
        Returns:
            Processed data DataFrame
        """
        output_file = str(self.file_paths['output_processed'])
        if Path(output_file).exists():
            return pd.read_csv(output_file)
        else:
            raise FileNotFoundError(f"Processed data file not found: {output_file}")
    
    def get_data_by_station(self, station_name: str) -> pd.DataFrame:
        """
        Get data filtered by station name.
        
        Args:
            station_name: Name of the station to filter by
            
        Returns:
            Filtered DataFrame
        """
        data = self.get_processed_data()
        filtered_data = data.query(f"{self.config['station_column']} == '{station_name}'").copy()
        return filtered_data
    
    def prepare_data_for_plotting(self) -> pd.DataFrame:
        """
        Prepare data for plotting by cleaning and sorting.
        
        Returns:
            DataFrame ready for plotting
        """
        data = self.get_processed_data()
        
        # Convert date column to datetime
        data[self.config['date_column']] = pd.to_datetime(data[self.config['date_column']])
        
        # Convert value column to numeric
        data[self.config['value_column']] = pd.to_numeric(data[self.config['value_column']], errors='coerce')
        
        # Sort by date
        data = data.sort_values(self.config['date_column'])
        
        # Remove rows with missing values
        data = data.dropna(subset=[self.config['value_column']])
        
        return data
    
    @classmethod
    def get_supported_variables(cls) -> List[str]:
        """
        Get list of supported meteorological variables.
        
        Returns:
            List of supported variable types
        """
        return get_supported_variables()
    
    @classmethod
    def create_for_variable(cls, variable_type: str, data_path: str = "data") -> 'TimeSeriesEngineerService':
        """
        Factory method to create a TimeSeriesEngineerService for a specific variable.
        
        Args:
            variable_type: Type of meteorological variable
            data_path: Directory path containing the data files
            
        Returns:
            Configured TimeSeriesEngineerService instance
        """
        return cls(variable_type, data_path) 