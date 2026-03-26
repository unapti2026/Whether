
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.services.time_series_engineer_service import TimeSeriesEngineerService
from src.config.settings import get_supported_variables, get_file_paths_for_variable, get_config_for_variable
from src.config.preprocessing_config import VariableType, PreprocessingConfig as BasePreprocessingConfig
from src.config.path_manager import get_path_manager


@dataclass
class PreprocessingConfig:
    variable_type: VariableType
    data_path: Optional[str] = None
    output_dir: Optional[str] = None
    enable_validation: bool = True
    enable_logging: bool = True
    save_intermediate: bool = False
    force_reprocess: bool = False
    
    def __post_init__(self):
        if not isinstance(self.variable_type, VariableType):
            self.variable_type = VariableType(self.variable_type)
        
        path_manager = get_path_manager()
        if self.data_path is None:
            self.data_path = str(path_manager.input_dir)
        
        if not Path(self.data_path).exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        supported_vars = get_supported_variables()
        if self.variable_type.value not in supported_vars:
            raise ValueError(
                f'Unsupported variable: {self.variable_type.value}. '
                f'Supported: {supported_vars}'
            )


@dataclass
class ProcessingResult:
    """Enhanced data class representing preprocessing result."""
    success: bool
    summary: Dict[str, Any]
    processed_data: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    validation_passed: bool = False
    warnings: List[str] = field(default_factory=list)
    
    @property
    def display_name(self) -> str:
        """Get display name from summary."""
        return self.summary.get('display_name', 'Unknown')
    
    @property
    def unit(self) -> str:
        """Get unit from summary."""
        return self.summary.get('unit', 'Unknown')
    
    @property
    def total_rows(self) -> int:
        """Get total rows from summary."""
        return self.summary.get('data_statistics', {}).get('total_rows', 0)
    
    @property
    def station_count(self) -> int:
        """Get station count from summary."""
        return self.summary.get('data_statistics', {}).get('stations', {}).get('count', 0)
    
    @property
    def missing_values_count(self) -> int:
        """Get missing values count from summary."""
        return self.summary.get('data_statistics', {}).get('value_statistics', {}).get('missing_values', 0)
    
    @property
    def missing_values_percentage(self) -> float:
        """Get missing values percentage."""
        total = self.total_rows
        missing = self.missing_values_count
        return (missing / total * 100) if total > 0 else 0.0


class DataValidator:
    """Enhanced data validation class."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_input_files(self) -> bool:
        """Validate that input files exist and are accessible."""
        try:
            file_paths = get_file_paths_for_variable(self.config.variable_type.value)
            excel_path = file_paths['input_excel']
            
            if not excel_path.exists():
                raise FileNotFoundError(f"Input Excel file not found: {excel_path}")
            
            self.logger.info(f"✓ Input file validation passed: {excel_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Input file validation failed: {e}")
            return False
    
    def validate_processed_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate processed data quality."""
        validation_results = {
            'passed': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check if data is empty
            if data.empty:
                validation_results['passed'] = False
                validation_results['errors'].append("Processed data is empty")
                return validation_results
            
            # Check required columns
            required_columns = ['Estación', 'Código', 'Fecha']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                validation_results['passed'] = False
                validation_results['errors'].append(f"Missing required columns: {missing_columns}")
            
            # Check for stations
            if 'Estación' in data.columns:
                station_count = data['Estación'].nunique()
                if station_count == 0:
                    validation_results['passed'] = False
                    validation_results['errors'].append("No stations found in data")
                elif station_count < 2:
                    validation_results['warnings'].append(f"Only {station_count} station found")
            
            # Check for missing values in value column
            value_column = self._get_value_column()
            if value_column in data.columns:
                missing_pct = data[value_column].isnull().sum() / len(data) * 100
                if missing_pct > 50:
                    validation_results['warnings'].append(f"High missing values: {missing_pct:.1f}%")
                elif missing_pct > 80:
                    validation_results['passed'] = False
                    validation_results['errors'].append(f"Too many missing values: {missing_pct:.1f}%")
            
            # Check date range
            if 'Fecha' in data.columns:
                try:
                    dates = pd.to_datetime(data['Fecha'])
                    date_range = dates.max() - dates.min()
                    if date_range.days < 30:
                        validation_results['warnings'].append("Short date range detected")
                except Exception:
                    validation_results['warnings'].append("Date column format issues detected")
            
            self.logger.info(f"✓ Data validation completed: {len(validation_results['warnings'])} warnings, {len(validation_results['errors'])} errors")
            
        except Exception as e:
            validation_results['passed'] = False
            validation_results['errors'].append(f"Validation error: {e}")
            self.logger.error(f"❌ Data validation failed: {e}")
        
        return validation_results
    
    def _get_value_column(self) -> str:
        """Get the value column name based on variable type."""
        column_mapping = {
            VariableType.TEMP_MAX: 'Temperatura',
            VariableType.TEMP_MIN: 'Temperatura',
            VariableType.PRECIPITATION: 'Precipitación',
            VariableType.HUMIDITY: 'Humedad'
        }
        return column_mapping.get(self.config.variable_type, 'Temperatura')


class MeteorologicalPreprocessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.service = None
        self.validator = DataValidator(config)
        self.processing_result: Optional[ProcessingResult] = None
        self.logger = self._setup_logger()
        
        self._validate_configuration()
        self._initialize_service()
        self._log_initialization()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the preprocessor."""
        logger = logging.getLogger(f"preprocessor_{self.config.variable_type.value}")
        if self.config.enable_logging:
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        return logger
    
    def _validate_configuration(self) -> None:
        """Validate the preprocessing configuration."""
        try:
            # Validate variable type
            supported_vars = get_supported_variables()
            if self.config.variable_type.value not in supported_vars:
                raise ValueError(
                    f'Unsupported variable: {self.config.variable_type.value}. '
                    f'Supported: {supported_vars}'
                )
            
            # Validate input files
            if not self.validator.validate_input_files():
                raise ValueError("Input file validation failed")
            
            self.logger.info("✓ Configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    def _initialize_service(self) -> None:
        """Initialize the time series engineer service."""
        try:
            self.service = TimeSeriesEngineerService(
                variable_type=self.config.variable_type.value,
                data_path=self.config.data_path
            )
            self.logger.info("✓ TimeSeriesEngineerService initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize TimeSeriesEngineerService: {e}")
            raise
    
    def _log_initialization(self) -> None:
        """Log initialization information."""
        self.logger.info(f"✓ Preprocessing service initialized for {self.config.variable_type.value}")
        self.logger.info(f"  Data path: {self.config.data_path}")
        if self.service:
            self.logger.info(f"  CSV file: {self.service.csv_file_name}")
            self.logger.info(f"  Configuration: {len(self.service.config)} parameters")
    
    def process_data(self) -> ProcessingResult:
        """
        Execute the complete preprocessing pipeline.
        
        Returns:
            ProcessingResult object with processing results and statistics
        """
        start_time = datetime.now()
        self.logger.info(f"🚀 Starting preprocessing for {self.config.variable_type.value}...")
        
        try:
            if not self.service:
                raise ValueError("Service not initialized")
            
            # Execute preprocessing
            summary = self.service.process_data()
            
            # Get processed data
            processed_data = self.service.get_processed_data()
            
            # Validate processed data
            validation_results = self.validator.validate_processed_data(processed_data)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            self.processing_result = ProcessingResult(
                success=validation_results['passed'],
                summary=summary,
                processed_data=processed_data,
                processing_time=processing_time,
                validation_passed=validation_results['passed'],
                warnings=validation_results['warnings']
            )
            
            if not validation_results['passed']:
                self.processing_result.error_message = "; ".join(validation_results['errors'])
            
            self._log_processing_result()
            return self.processing_result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ Error during preprocessing: {e}")
            
            self.processing_result = ProcessingResult(
                success=False,
                summary={},
                processing_time=processing_time,
                error_message=str(e)
            )
            
            return self.processing_result
    
    def _log_processing_result(self) -> None:
        """Log processing results."""
        if not self.processing_result:
            return
        
        if self.processing_result.success:
            self.logger.info("✅ Preprocessing completed successfully")
            self.logger.info(f"📊 Summary for {self.processing_result.display_name}:")
            self.logger.info(f"  Unit: {self.processing_result.unit}")
            self.logger.info(f"  Total rows: {self.processing_result.total_rows:,}")
            self.logger.info(f"  Stations: {self.processing_result.station_count}")
            self.logger.info(f"  Processing time: {self.processing_result.processing_time:.2f}s")
            self.logger.info(f"  Missing values: {self.processing_result.missing_values_percentage:.1f}%")
            
            if self.processing_result.warnings:
                self.logger.warning(f"⚠️ Warnings: {len(self.processing_result.warnings)}")
                for warning in self.processing_result.warnings:
                    self.logger.warning(f"  - {warning}")
            
            self._display_detailed_statistics(self.processing_result.summary)
        else:
            self.logger.error(f"❌ Preprocessing failed: {self.processing_result.error_message}")
    
    def _display_detailed_statistics(self, summary: Dict[str, Any]) -> None:
        """Display detailed processing statistics."""
        if not self.processing_result:
            return
            
        # Value statistics
        stats = summary.get('data_statistics', {}).get('value_statistics', {})
        if stats:
            self.logger.info(f"📈 Value Statistics:")
            self.logger.info(f"  Minimum: {stats.get('min', 0):.2f} {self.processing_result.unit}")
            self.logger.info(f"  Maximum: {stats.get('max', 0):.2f} {self.processing_result.unit}")
            self.logger.info(f"  Average: {stats.get('mean', 0):.2f} {self.processing_result.unit}")
            self.logger.info(f"  Missing values: {stats.get('missing_values', 0):,}")
        
        # Date range
        date_range = summary.get('data_statistics', {}).get('date_range', {})
        if date_range:
            self.logger.info(f"📅 Date Range:")
            self.logger.info(f"  From: {date_range.get('start', 'Unknown')}")
            self.logger.info(f"  To: {date_range.get('end', 'Unknown')}")
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the processed data.
        
        Returns:
            Processed DataFrame
            
        Raises:
            ValueError: If data hasn't been processed yet
        """
        if not self.processing_result or not self.processing_result.success:
            raise ValueError("Data hasn't been processed successfully yet. Run process_data() first.")
        
        if self.processing_result.processed_data is None:
            raise ValueError("No processed data available.")
        
        return self.processing_result.processed_data
    
    def get_stations(self) -> List[str]:
        """
        Get list of available stations.
        
        Returns:
            List of station names
            
        Raises:
            ValueError: If data hasn't been processed yet
        """
        if not self.service:
            raise ValueError("Service not initialized.")
        
        processed_data = self.get_processed_data()
        station_column = self.service.config.get('station_column', 'Estación')
        return processed_data[station_column].unique().tolist()
    
    def get_station_data(self, station_name: str) -> pd.DataFrame:
        """
        Get data for a specific station.
        
        Args:
            station_name: Name of the station
            
        Returns:
            DataFrame with data for the specified station
            
        Raises:
            ValueError: If data hasn't been processed yet or station not found
        """
        if not self.service:
            raise ValueError("Service not initialized.")
        
        return self.service.get_data_by_station(station_name)
    
    def prepare_data_for_plotting(self) -> pd.DataFrame:
        """
        Prepare data for plotting.
        
        Returns:
            DataFrame ready for plotting
            
        Raises:
            ValueError: If data hasn't been processed yet
        """
        if not self.service:
            raise ValueError("Service not initialized.")
        
        return self.service.prepare_data_for_plotting()
    
    def get_processing_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get the processing summary.
        
        Returns:
            Processing summary dictionary or None if not available
        """
        if self.processing_result:
            return self.processing_result.summary
        return None
    
    def display_station_information(self, max_display: int = 5) -> None:
        """
        Display information about available stations.
        
        Args:
            max_display: Maximum number of stations to display
        """
        try:
            stations = self.get_stations()
            self.logger.info(f"📋 Available Stations ({len(stations)}):")
            
            for i, station in enumerate(stations[:max_display], 1):
                station_data = self.get_station_data(station)
                self.logger.info(f"  {i}. {station} ({station_data.shape[0]} rows)")
            
            if len(stations) > max_display:
                self.logger.info(f"  ... and {len(stations) - max_display} more")
                
        except Exception as e:
            self.logger.error(f"⚠️ Error displaying station information: {e}")
    
    def test_data_access(self) -> None:
        """Test data access methods."""
        try:
            self.logger.info("🧪 Testing data access methods...")
            
            # Test getting processed data
            processed_data = self.get_processed_data()
            self.logger.info(f"  ✓ Processed data shape: {processed_data.shape}")
            self.logger.info(f"  ✓ Columns: {list(processed_data.columns)}")
            
            # Test getting stations
            stations = self.get_stations()
            self.logger.info(f"  ✓ Available stations: {len(stations)}")
            
            # Test getting station data
            if stations:
                test_station = stations[0]
                station_data = self.get_station_data(test_station)
                self.logger.info(f"  ✓ Station '{test_station}' data: {station_data.shape[0]} rows")
            
            # Test plotting data preparation
            plot_data = self.prepare_data_for_plotting()
            self.logger.info(f"  ✓ Plot-ready data: {plot_data.shape[0]} rows")
            self.logger.info(f"  ✓ Data completeness: {plot_data.shape[0]/processed_data.shape[0]*100:.1f}%")
            
            self.logger.info("  ✅ All data access tests passed")
            
        except Exception as e:
            self.logger.error(f"  ❌ Data access test failed: {e}")
    
    def reset(self) -> None:
        """Reset the preprocessor state."""
        self.processing_result = None
        self.logger.info("✓ Preprocessor state reset")


def load_processed_data(variable_type: VariableType, data_path: str) -> pd.DataFrame:
    """
    Load processed data for the specified variable type.
    
    Args:
        variable_type: Type of variable to load
        data_path: Path to the data directory
        
    Returns:
        DataFrame containing processed data
    """
    print('Loading processed data...')
    
    # Get file paths
    file_paths = get_file_paths_for_variable(variable_type.value)
    processed_file = file_paths['output_processed']
    
    if processed_file.exists():
        processed_data = pd.read_csv(processed_file)
        
        print(f'Data loaded: {len(processed_data):,} rows')
        print(f'  Columns: {list(processed_data.columns)}')
        print(f'  Stations: {processed_data["Estación"].nunique()}')
        
        print('Missing values summary:')
        print(processed_data.isnull().sum())
        
        return processed_data
    else:
        print('No processed data found. Processing data first...')
        
        # Process data if not available
        config = PreprocessingConfig(variable_type=variable_type, data_path=data_path)
        preprocessor = MeteorologicalPreprocessor(config)
        result = preprocessor.process_data()
        
        if result.success and result.processed_data is not None:
            print(f'Data processed and loaded: {len(result.processed_data):,} rows')
            return result.processed_data
        else:
            raise ValueError(f"Failed to process data: {result.error_message}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Meteorological Data Preprocessing Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    python preprocess_meteorological.py                    # Process temp_max (default)
    python preprocess_meteorological.py temp_min          # Process temp_min
    python preprocess_meteorological.py precipitation     # Process precipitation
    python preprocess_meteorological.py humidity          # Process humidity
    python preprocess_meteorological.py --help            # Show this help
    
PIPELINE INTEGRATION:
    # Called from pipeline with max_stations parameter
    python preprocess_meteorological.py precipitation --max-stations 5
        """
    )
    
    parser.add_argument(
        'variable',
        nargs='?',
        default='temp_max',
        choices=['temp_max', 'temp_min', 'precipitation', 'humidity'],
        help='Meteorological variable to process (default: temp_max)'
    )
    
    parser.add_argument(
        '--data-path',
        default=None,
        help='Path to data directory (default: uses PathManager)'
    )
    
    parser.add_argument(
        '--max-stations',
        type=int,
        default=None,
        help='Maximum number of stations to process (default: all)'
    )
    
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable data validation'
    )
    
    parser.add_argument(
        '--no-logging',
        action='store_true',
        help='Disable logging'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Convert variable string to VariableType enum
    variable_mapping = {
        'temp_max': VariableType.TEMP_MAX,
        'temp_min': VariableType.TEMP_MIN,
        'precipitation': VariableType.PRECIPITATION,
        'humidity': VariableType.HUMIDITY
    }
    
    variable_type = variable_mapping[args.variable]
    
    path_manager = get_path_manager()
    data_path = args.data_path if args.data_path else str(path_manager.input_dir)
    
    config = PreprocessingConfig(
        variable_type=variable_type,
        data_path=data_path,
        enable_validation=not args.no_validation,
        enable_logging=not args.no_logging
    )
    
    # Display configuration
    print(f'Preprocessing configuration:')
    print(f'  Variable: {config.variable_type.value}')
    print(f'  Data path: {config.data_path}')
    print(f'  Max stations: {args.max_stations if args.max_stations else "All"}')
    print(f'  Validation enabled: {config.enable_validation}')
    print(f'  Logging enabled: {config.enable_logging}')
    print(f'  Plots enabled: {not args.no_plots}')
    
    # Get file paths
    file_paths = get_file_paths_for_variable(config.variable_type.value)
    print(f'  Excel file: {file_paths["input_excel"]}')
    print(f'  CSV file: {file_paths["input_csv"]}')
    
    # Load and process data
    processed_data = load_processed_data(variable_type, args.data_path)
    
    # Apply station limiting if specified
    if args.max_stations and args.max_stations > 0:
        unique_stations = processed_data['Estación'].unique()
        if len(unique_stations) > args.max_stations:
            print(f'Limiting to {args.max_stations} stations (from {len(unique_stations)} total)')
            selected_stations = unique_stations[:args.max_stations]
            processed_data = processed_data[processed_data['Estación'].isin(selected_stations)]
            print(f'Data limited to {len(selected_stations)} stations')
    
    # Display final data summary
    print(f'Final data summary:')
    print(f'  Total rows: {len(processed_data):,}')
    print(f'  Stations: {processed_data["Estación"].nunique()}')
    print(f'  Date range: {processed_data["Fecha"].min()} to {processed_data["Fecha"].max()}')
    print(f'  Missing values: {processed_data.isnull().sum().sum()}')
    
    print(f'Preprocessing completed successfully!')
    print(f'Output: {file_paths["input_csv"]}')
    
    return processed_data


if __name__ == "__main__":
    main()
