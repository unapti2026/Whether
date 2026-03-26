"""
Prediction Processor Service

This service orchestrates the complete prediction workflow for meteorological data,
including data loading, EEMD decomposition, model training, and prediction generation.
Implements IVariableAgnosticProcessor for complete variable independence.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import re

from src.core.interfaces.prediction_strategy import (
    PredictionConfig, EEMDResult, ModelTrainingResult, PredictionResult
)
from src.core.interfaces.variable_agnostic_interfaces import (
    IVariableAgnosticProcessor, ProcessingConfig, ProcessingResult
)
from src.core.config.unified_configuration_system import (
    UnifiedConfigurationFactory, UnifiedConfigurationValidator,
    UnifiedConfigurationLogger, UnifiedConfigurationMemoryManager
)
from src.config.configuration_service import ConfigurationService
from src.core.services.validation_service import ValidationService
from src.core.services.logging_service import LoggingService
from src.core.exceptions.processing_exceptions import ProcessingError
from .eemd_service import EEMDService
from .hybrid_model_service import HybridModelService
from .prediction_service import PredictionService
from .model_persistence_service import ModelPersistenceService
from src.data.visualization.services.visualization_service import VisualizationService
from .threshold_loader import ThresholdLoader
from .alert_detector import AlertDetector, TemperatureAlert

logger = logging.getLogger(__name__)


class PredictionProcessor(IVariableAgnosticProcessor):
    """
    Main processor for the prediction workflow.
    
    This service implements the Single Responsibility Principle by orchestrating
    the prediction workflow and delegating specific tasks to specialized services.
    """
    
    def __init__(self, 
                 config: PredictionConfig,
                 eemd_service: Optional[EEMDService] = None,
                 hybrid_model_service: Optional[HybridModelService] = None,
                 prediction_service: Optional[PredictionService] = None,
                 model_persistence_service: Optional[ModelPersistenceService] = None,
                 visualization_service: Optional[VisualizationService] = None,
                 configuration_service: Optional[ConfigurationService] = None,
                 validation_service: Optional[ValidationService] = None,
                 logging_service: Optional[LoggingService] = None):
        """
        Initialize the prediction processor with dependency injection.
        
        Args:
            config: Prediction configuration
            eemd_service: EEMD service (optional, will be created if not provided)
            hybrid_model_service: Hybrid model service (optional, will be created if not provided)
            prediction_service: Prediction service (optional, will be created if not provided)
            model_persistence_service: Model persistence service (optional, will be created if not provided)
            visualization_service: Visualization service (optional, will be created if not provided)
            configuration_service: Configuration service (optional, will be created if not provided)
            validation_service: Validation service (optional, will be created if not provided)
            logging_service: Logging service (optional, will be created if not provided)
        """
        self.config = config
        
        # Initialize core services with dependency injection
        self.configuration_service = configuration_service or ConfigurationService()
        self.validation_service = validation_service or ValidationService()
        self.logging_service = logging_service or LoggingService()
        
        # Get logger from logging service
        self.logger = self.logging_service.get_logger("prediction_processor")
        
        # Initialize specialized services with dependency injection
        self.eemd_service = eemd_service or EEMDService(config)
        self.hybrid_model_service = hybrid_model_service or HybridModelService(config.variable_type)
        self.prediction_service = prediction_service or PredictionService(config.variable_type)
        self.model_persistence_service = model_persistence_service or ModelPersistenceService(config.variable_type)
        self.visualization_service = visualization_service or VisualizationService()
        
        # FASE 3: Initialize alert system (only for temp_max and temp_min)
        self.threshold_loader = None
        self.alert_detector = None
        if config.variable_type in ['temp_max', 'temp_min']:
            try:
                self.threshold_loader = ThresholdLoader()
                self.alert_detector = AlertDetector(self.threshold_loader)
                self.logger.info("[OK] Alert system initialized (Fase 3)")
            except Exception as e:
                self.logger.warning(f"[WARNING] Alert system initialization failed: {e}. Continuing without alerts.")
        
        # Setup output directories
        self._setup_output_directories()
        
        # Add models directory to output directories
        from src.config.path_manager import get_path_manager
        path_manager = get_path_manager()
        base_prediction_dir = path_manager.get_output_subdir("prediction", config.variable_type)
        self.output_dirs['models'] = base_prediction_dir / "models"
        self.output_dirs['prediction_plots'] = base_prediction_dir / "plots"
        
        # Log service initialization
        self.logging_service.log_service_initialization("PredictionProcessor", config.variable_type)
        
        # Initialize unified system components
        self.unified_factory = UnifiedConfigurationFactory()
        self.unified_validator = UnifiedConfigurationValidator()
        self.unified_logger = UnifiedConfigurationLogger()
        self.unified_memory_manager = UnifiedConfigurationMemoryManager()
        
        # Current processing state
        self._current_unified_config: Optional[ProcessingConfig] = None
        self._current_unified_data: Optional[pd.DataFrame] = None
        self._processing_start_time: Optional[float] = None
    
    def _setup_output_directories(self) -> None:
        """Create necessary output directories."""
        from src.config.path_manager import get_path_manager
        path_manager = get_path_manager()
        base_output_dir = path_manager.get_output_subdir("prediction", self.config.variable_type)
        
        # Create main directories
        self.output_dirs = {
            'base': base_output_dir,
            'csv_files': base_output_dir / "csv_files",
            'eemd_plots': base_output_dir / "eemd_plots",
            'prediction_plots': base_output_dir / "prediction_plots",
            'models': base_output_dir / "models",
            'reports': base_output_dir / "reports",
            'eemd_results': base_output_dir / "eemd_results",
            'alerts': base_output_dir / "alerts"  # FASE 3: Alert directory
        }
        
        # Automatically create all directories if they don't exist
        for dir_path in self.output_dirs.values():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logging_service.log_error(
                    "PredictionProcessor", "Directory Creation", str(e)
                )
                raise ProcessingError(f"Failed to create directory {dir_path}: {e}")
        
        self.logging_service.log_validation_result(
            "PredictionProcessor", "Directory Setup", True, 
            f"Created {len(self.output_dirs)} directories"
        )
    
    def _get_station_output_dirs(self, station_name: str) -> Dict[str, Path]:
        """
        Get station-specific output directories.
        
        Args:
            station_name: Name of the station
            
        Returns:
            Dictionary with station-specific output directories
        """
        # Sanitize station name for use in file paths
        safe_station_name = self._sanitize_station_name(station_name)
        
        # Create station-specific directories
        station_dirs = {
            'csv_files': self.output_dirs['csv_files'] / safe_station_name,
            'eemd_plots': self.output_dirs['eemd_plots'] / safe_station_name,
            'prediction_plots': self.output_dirs['prediction_plots'] / safe_station_name,
            'models': self.output_dirs['models'] / safe_station_name,
            'reports': self.output_dirs['reports'] / safe_station_name,
            'eemd_results': self.output_dirs['eemd_results'] / safe_station_name,
            'alerts': self.output_dirs.get('alerts', self.output_dirs['base'] / 'alerts') / safe_station_name  # FASE 3
        }
        
        # Create all station directories
        for dir_path in station_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return station_dirs
    
    def _sanitize_station_name(self, station_name: str) -> str:
        """
        Sanitize station name for use in file paths.
        
        Args:
            station_name: Original station name
            
        Returns:
            Sanitized station name safe for file paths
        """
        # Replace problematic characters with underscores
        sanitized = station_name.replace(' ', '_').replace(',', '').replace('.', '').replace('/', '_').replace('\\', '_')
        # Remove any other potentially problematic characters
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c in '_-')
        # Limit length to avoid path issues
        sanitized = sanitized[:50]
        
        # Validate sanitized name using validation service
        try:
            # Create a proper object with the name attribute
            class StationNameConfig:
                def __init__(self, name):
                    self.name = name
            
            config_obj = StationNameConfig(sanitized)
            self.validation_service.validate_configuration(
                config_obj, ['name']
            )
        except Exception as e:
            self.logging_service.log_warning(
                "PredictionProcessor", "Name Sanitization", 
                f"Station name sanitization issue: {e}"
            )
        
        return sanitized
    
    def load_imputed_data(self, variable_type: str) -> pd.DataFrame:
        """
        Load imputed data for the specified variable type.
        
        Args:
            variable_type: Type of meteorological variable
            
        Returns:
            DataFrame containing imputed data
            
        Raises:
            ProcessingError: If data loading fails
        """
        try:
            # Look for imputed data in the imputation output directory
            from src.config.path_manager import get_path_manager
            path_manager = get_path_manager()
            
            # Try primary location (new structure)
            imputation_dir = path_manager.get_output_subdir("imputation", variable_type) / "csv_files"
            
            # Check if primary location has files (optimize: check existence first)
            primary_has_files = False
            if imputation_dir.exists():
                csv_files_list = list(imputation_dir.glob("*.csv"))
                primary_has_files = len(csv_files_list) > 0
            
            # Fallback to old location if primary doesn't exist or is empty
            if not primary_has_files:
                old_location = path_manager.project_root / "static" / "output" / "imputation" / variable_type / "csv_files"
                old_has_files = False
                if old_location.exists():
                    old_csv_files = list(old_location.glob("*.csv"))
                    old_has_files = len(old_csv_files) > 0
                
                if old_has_files:
                    self.logger.info(f"Using legacy imputation directory: {old_location}")
                    imputation_dir = old_location
                elif not imputation_dir.exists():
                    # Create directory if it doesn't exist (automatic creation)
                    imputation_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.warning(f"Created empty imputation directory: {imputation_dir}")
            
            # Log which directory we're using
            self.logger.info(f"Loading imputed data from: {imputation_dir}")
            
            # Find all CSV files in the directory
            self.logger.info(f"Scanning for CSV files in {imputation_dir}...")
            all_csv_files = list(imputation_dir.glob("*.csv"))
            self.logger.info(f"Found {len(all_csv_files)} CSV files")
            
            if not all_csv_files:
                raise ProcessingError(
                    f"No CSV files found in: {imputation_dir}. "
                    f"Please ensure imputation process completed successfully."
                )
            
            # Filter out consolidated files and include only individual station files
            station_csv_files = []
            excluded_files = []
            
            for csv_file in all_csv_files:
                filename = csv_file.name
                
                # Exclude consolidated files (files that start with "all_stations")
                if filename.startswith("all_stations"):
                    excluded_files.append(filename)
                    continue
                
                # Include files that follow the pattern: {code}_{station_name}_Imputed.csv
                if "_Imputed.csv" in filename and not filename.startswith("all_stations"):
                    station_csv_files.append(csv_file)
                else:
                    excluded_files.append(filename)
            
            # CRITICAL FIX: Sort CSV files by station code for consistency
            # Extract code from filename pattern: {code}_{station_name}_Imputed.csv
            def extract_station_code(filename: str) -> int:
                """Extract station code from filename for sorting."""
                try:
                    # Filename format: {code}_{station_name}_Imputed.csv
                    # Extract the first numeric part (the code)
                    code_match = re.search(r'^(\d+)_', filename)
                    if code_match:
                        return int(code_match.group(1))
                    # Fallback: try to find any number at the start
                    code_match = re.search(r'(\d+)', filename)
                    return int(code_match.group(1)) if code_match else 999999
                except (ValueError, AttributeError):
                    return 999999  # Sort files without codes last
            
            # Sort files by station code (ascending)
            station_csv_files.sort(key=lambda f: extract_station_code(f.name))
            self.logger.info(f"Sorted {len(station_csv_files)} station files by code for consistency")
            
            self.logging_service.log_validation_result(
                "PredictionProcessor", "File Discovery", True,
                f"Found {len(station_csv_files)} station files, {len(excluded_files)} excluded"
            )
            
            if not station_csv_files:
                raise ProcessingError("No individual station files found")
            
            # Load and combine all individual station CSV files
            self.logger.info(f"Loading {len(station_csv_files)} station CSV files...")
            dataframes = []
            for idx, csv_file in enumerate(station_csv_files, 1):
                try:
                    self.logger.info(f"  [{idx}/{len(station_csv_files)}] Loading {csv_file.name}...")
                    # Load CSV with robust date column detection
                    df = self._load_csv_with_date_index(csv_file)
                    
                    # Validate the loaded DataFrame using validation service
                    self._validate_loaded_dataframe(df, csv_file)
                    
                    station_name = csv_file.stem  # Get filename without extension
                    df['Estación'] = station_name
                    
                    # Log detailed information
                    self._log_dataframe_info(df, station_name)
                    
                    dataframes.append(df)
                    self.logger.info(f"  [OK] Loaded {station_name} ({len(df)} records)")
                    self.logging_service.log_validation_result(
                        "PredictionProcessor", "Data Loading", True,
                        f"Loaded {station_name} ({len(df)} records)"
                    )
                except Exception as e:
                    self.logger.error(f"  [ERROR] Failed to load {csv_file.name}: {e}")
                    self.logging_service.log_error(
                        "PredictionProcessor", "Data Loading", str(e),
                        f"Failed to load {csv_file.name}"
                    )
            
            if not dataframes:
                raise ProcessingError("No data could be loaded")
            
            # Combine all dataframes
            combined_df = pd.concat(dataframes, ignore_index=False)
            combined_df = combined_df.sort_index()
            
            self.logger.info(f"[STATS] Combined dataset: {len(combined_df)} total records")
            self.logger.info(f"   - Date range: {combined_df.index.min()} to {combined_df.index.max()}")
            self.logger.info(f"   - Stations: {combined_df['Estación'].nunique()}")
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Failed to load imputed data: {e}")
            raise ProcessingError(f"Failed to load imputed data: {e}")
    
    def _load_csv_with_date_index(self, csv_file: Path) -> pd.DataFrame:
        """
        Load CSV file with robust date column detection.
        
        Args:
            csv_file: Path to the CSV file
            
        Returns:
            DataFrame with date column as index
            
        Raises:
            ProcessingError: If no date column is found or loading fails
        """
        try:
            # First, read the CSV to inspect columns
            df_temp = pd.read_csv(csv_file, low_memory=False)
            
            # Look for date-related column names (case-insensitive)
            date_column_candidates = []
            for col in df_temp.columns:
                col_lower = col.lower()
                if any(date_keyword in col_lower for date_keyword in ['fecha', 'date', 'time', 'timestamp', 'datetime']):
                    date_column_candidates.append(col)
            
            # If no obvious date column found, try to infer from data
            if not date_column_candidates:
                for col in df_temp.columns:
                    try:
                        # Skip columns that are clearly not dates
                        if col.lower() in ['código', 'codigo', 'estación', 'estacion', 'station', 'code']:
                            continue
                        
                        # Try to parse first few non-null values as dates
                        sample_values = df_temp[col].dropna().head(10)
                        if len(sample_values) == 0:
                            continue
                            
                        # Try different date parsing strategies
                        try:
                            pd.to_datetime(sample_values)
                            date_column_candidates.append(col)
                            break
                        except (ValueError, TypeError):
                            # Try with different date formats
                            try:
                                pd.to_datetime(sample_values, format='%Y-%m-%d')
                                date_column_candidates.append(col)
                                break
                            except (ValueError, TypeError):
                                try:
                                    pd.to_datetime(sample_values, format='%d/%m/%Y')
                                    date_column_candidates.append(col)
                                    break
                                except (ValueError, TypeError):
                                    continue
                    except Exception:
                        continue
            
            if not date_column_candidates:
                raise ProcessingError(
                    f"No date column found in {csv_file.name}. "
                    f"Available columns: {list(df_temp.columns)}. "
                    f"Please ensure there's a column with date values."
                )
            
            # Use the first found date column
            date_column = date_column_candidates[0]
            self.logger.debug(f"Using '{date_column}' as date column for {csv_file.name}")
            
            # Try different parsing strategies for the full dataset
            df = None
            parsing_errors = []
            
            # Strategy 1: Standard pandas parsing
            try:
                df = pd.read_csv(csv_file, index_col=date_column, parse_dates=True, low_memory=False)
                if isinstance(df.index, pd.DatetimeIndex):
                    return df
            except Exception as e:
                parsing_errors.append(f"Standard parsing failed: {e}")
            
            # Strategy 2: Manual date parsing
            try:
                df_temp = pd.read_csv(csv_file, low_memory=False)
                df_temp[date_column] = pd.to_datetime(df_temp[date_column])
                df = df_temp.set_index(date_column)
                if isinstance(df.index, pd.DatetimeIndex):
                    return df
            except Exception as e:
                parsing_errors.append(f"Manual parsing failed: {e}")
            
            # Strategy 3: Try with specific format
            try:
                df_temp = pd.read_csv(csv_file, low_memory=False)
                df_temp[date_column] = pd.to_datetime(df_temp[date_column], format='%Y-%m-%d')
                df = df_temp.set_index(date_column)
                if isinstance(df.index, pd.DatetimeIndex):
                    return df
            except Exception as e:
                parsing_errors.append(f"Format parsing failed: {e}")
            
            # If all strategies failed
            raise ProcessingError(
                f"Could not parse date column '{date_column}' in {csv_file.name}. "
                f"Errors: {'; '.join(parsing_errors)}"
            )
            
        except Exception as e:
            if isinstance(e, ProcessingError):
                raise
            raise ProcessingError(f"Failed to load {csv_file.name} with date index: {e}")
    
    def _validate_loaded_dataframe(self, df: pd.DataFrame, csv_file: Path) -> None:
        """
        Validate that the loaded DataFrame has the correct structure using validation service.
        
        Args:
            df: DataFrame to validate
            csv_file: Path to the original CSV file
            
        Raises:
            ProcessingError: If validation fails
        """
        try:
            # Check if index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ProcessingError(f"Index is not datetime in {csv_file.name}")
            
            # Check if index is sorted
            if not df.index.is_monotonic_increasing:
                self.logging_service.log_warning(
                    "PredictionProcessor", "Data Validation", 
                    f"Date index is not sorted in {csv_file.name}, sorting..."
                )
                df.sort_index(inplace=True)
            
            # Check for required columns using validation service
            # Get target column name dynamically based on variable type
            target_column = self._get_target_column_name()
            required_columns = ['Estación', target_column]
            self.validation_service.validate_dataframe(df, required_columns, min_rows=1)
            
            # Check for reasonable date range
            date_range = df.index.max() - df.index.min()
            if date_range.days < 30:  # Less than a month
                self.logging_service.log_warning(
                    "PredictionProcessor", "Data Validation", 
                    f"Very short date range in {csv_file.name}: {date_range.days} days"
                )
            
            # Check for duplicate dates
            if df.index.duplicated().any():
                self.logging_service.log_warning(
                    "PredictionProcessor", "Data Validation", 
                    f"Duplicate dates found in {csv_file.name}, keeping first occurrence"
                )
                df = df[~df.index.duplicated(keep='first')]
            
            # Log successful validation
            self.logging_service.log_validation_result(
                "PredictionProcessor", "DataFrame Validation", True,
                f"Validated {csv_file.name} ({len(df)} rows)"
            )
            
        except Exception as e:
            self.logging_service.log_validation_result(
                "PredictionProcessor", "DataFrame Validation", False,
                f"Validation failed for {csv_file.name}: {str(e)}"
            )
            raise ProcessingError(f"DataFrame validation failed for {csv_file.name}: {e}")
    
    def _log_dataframe_info(self, df: pd.DataFrame, station_name: str) -> None:
        """
        Log detailed information about a loaded DataFrame using logging service.
        
        Args:
            df: DataFrame to analyze
            station_name: Name of the station
        """
        try:
            # Get logger for detailed logging
            logger_instance = self.logging_service.get_logger("prediction_processor")
            
            logger_instance.debug(f"[INFO] DataFrame info for {station_name}:")
            logger_instance.debug(f"   - Shape: {df.shape}")
            logger_instance.debug(f"   - Index type: {type(df.index)}")
            logger_instance.debug(f"   - Index range: {df.index.min()} to {df.index.max()}")
            logger_instance.debug(f"   - Columns: {list(df.columns)}")
            logger_instance.debug(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                logger_instance.debug(f"   - Missing values: {missing_values.to_dict()}")
            else:
                logger_instance.debug(f"   - No missing values")
                
        except Exception as e:
            self.logging_service.log_warning(
                "PredictionProcessor", "Data Logging", 
                f"Could not log DataFrame info for {station_name}: {e}"
            )
    
    def process_stations(self, imputed_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process all stations for prediction.
        
        Args:
            imputed_data: DataFrame containing imputed data for all stations
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()  # Track processing time
        
        if imputed_data.empty:
            raise ProcessingError("No data to process")
        
        # Get unique stations
        stations = imputed_data['Estación'].unique()
        total_stations = len(stations)
        
        # Log processing start using logging service
        self.logging_service.log_service_initialization(
            "PredictionProcessor", f"Processing {total_stations} stations"
        )
        
        # Get logger for detailed logging
        logger_instance = self.logging_service.get_logger("prediction_processor")
        logger_instance.info(f"   - Variable: {self.config.variable_type}")
        logger_instance.info(f"   - Max stations to process: {self.config.max_stations}")
        
        # Limit number of stations if specified
        if self.config.max_stations and total_stations > self.config.max_stations:
            stations = stations[:self.config.max_stations]
            self.logging_service.log_warning(
                "PredictionProcessor", "Station Limiting", 
                f"Limited to first {len(stations)} stations (max: {self.config.max_stations})"
            )
        
        logger_instance.info(f"📋 Stations to process:")
        for i, station in enumerate(stations, 1):
            station_data = imputed_data[imputed_data['Estación'] == station]
            logger_instance.info(f"   {i:2d}. {station} ({len(station_data)} records)")
        
        # Process each station
        successful_stations = 0
        failed_stations = 0
        results = []
        
        for i, station in enumerate(stations, 1):
            logger_instance.info(f"\n{'='*60}")
            logger_instance.info(f"🔧 Processing station {i}/{len(stations)}: {station}")
            logger_instance.info(f"{'='*60}")
            
            try:
                result = self._process_single_station(imputed_data, station, i, len(stations))
                results.append(result)
                successful_stations += 1
                self.logging_service.log_validation_result(
                    "PredictionProcessor", "Station Processing", True,
                    f"Station {station} processed successfully"
                )
            except Exception as e:
                failed_stations += 1
                self.logging_service.log_error(
                    "PredictionProcessor", "Station Processing", str(e),
                    f"Failed to process station {station}"
                )
                continue
        
        # Generate summary
        summary = {
            'total_stations': len(stations),
            'successful_stations': successful_stations,
            'failed_stations': failed_stations,
            'success_rate': (successful_stations / len(stations)) * 100 if len(stations) > 0 else 0,
            'results': results
        }
        
        # Log summary using logging service
        logger_instance.info(f"\n{'='*60}")
        logger_instance.info(f"[SUMMARY] PROCESSING SUMMARY")
        logger_instance.info(f"{'='*60}")
        logger_instance.info(f"   [OK] Successful: {successful_stations}/{len(stations)} stations")
        logger_instance.info(f"   [ERROR] Failed: {failed_stations}/{len(stations)} stations")
        logger_instance.info(f"   [STATS] Success rate: {summary['success_rate']:.1f}%")
        
        # Log completion
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        self.logging_service.log_service_completion("PredictionProcessor", processing_time)
        
        return summary
    
    def _process_single_station(self, 
                               imputed_data: pd.DataFrame, 
                               station_name: str,
                               station_index: int, 
                               total_stations: int) -> Dict[str, Any]:
        """
        Process a single station for prediction using centralized services.
        
        Args:
            imputed_data: DataFrame containing imputed data
            station_name: Name of the station to process
            station_index: Index of the station in the processing queue
            total_stations: Total number of stations to process
            
        Returns:
            Dictionary with processing results for the station
        """
        start_time = time.time()
        
        # Extract station data
        station_data = imputed_data[imputed_data['Estación'] == station_name].copy()
        
        # Validate station data using validation service
        try:
            # Get target column name dynamically based on variable type
            target_column = self._get_target_column_name()
            self.validation_service.validate_dataframe(station_data, ['Estación', target_column], min_rows=1)
        except Exception as e:
            raise ProcessingError(f"No valid data found for station {station_name}: {e}")
        
        # Get target column name based on variable type
        target_column = self._get_target_column_name()
        
        if target_column not in station_data.columns:
            raise ProcessingError(f"Target column '{target_column}' not found in station data")
        
        self.logger.info(f"[INFO] Station data summary:")
        self.logger.info(f"   - Records: {len(station_data)}")
        self.logger.info(f"   - Date range: {station_data.index.min()} to {station_data.index.max()}")
        self.logger.info(f"   - Target column: {target_column}")
        self.logger.info(f"   - Missing values: {station_data[target_column].isnull().sum()}")
        
        # Step 1: EEMD Decomposition with improved architecture
        self.logger.info(f"\n🔍 Step 1: EEMD Decomposition")
        time_series = station_data[target_column]
        
        # Create adaptive configuration for memory optimization
        from src.config.adaptive_config import create_adaptive_config
        adaptive_config = create_adaptive_config(len(time_series), self.config.variable_type)
        
        # Use the improved EEMD service directly with memory optimization
        eemd_result = self.eemd_service.decompose_time_series(time_series, adaptive_config)
        
        # Get station-specific output directories
        station_dirs = self._get_station_output_dirs(station_name)
        
        # Save EEMD results
        eemd_output_path = station_dirs['eemd_results']
        self.eemd_service.save_eemd_results(eemd_result, eemd_output_path, station_name)
        
        # Generate EEMD plots using unified visualization service (if enabled)
        if getattr(self.config, 'enable_plots', True):
            eemd_plots_path = station_dirs['eemd_plots']
            try:
                generated_plots = self.visualization_service.generate_eemd_plots(
                    eemd_result, station_name, eemd_plots_path, time_series
                )
                self.logging_service.log_validation_result(
                    "VisualizationService", "EEMD Plots", True,
                    f"Generated {len(generated_plots)} EEMD plots"
                )
                
                # Generate summary report
                summary_report = self.visualization_service.generate_summary_report(
                    station_name, eemd_result, None, eemd_plots_path
                )
                self.logger.info(f"   📋 EEMD summary report: {summary_report}")
                
            except Exception as e:
                self.logging_service.log_error(
                    "VisualizationService", "EEMD Plots", str(e),
                    f"Failed to generate EEMD plots for {station_name}"
                )
        else:
            self.logger.info(f"   [INFO] EEMD plots generation disabled (--no-plots flag)")
        
        # Display EEMD results
        self._log_eemd_results(eemd_result)
        
        # Step 2: Model Training
        self.logger.info(f"\n🤖 Step 2: Model Training")
        try:
            model_result = self.hybrid_model_service.train_models_legacy(eemd_result, time_series, self.config, adaptive_config)
            
            if not model_result.success:
                self.logger.error(f"   [ERROR] Model training failed: {model_result.error_message}")
                return {
                    'station_name': station_name,
                    'station_index': station_index,
                    'total_records': len(station_data),
                    'eemd_result': eemd_result,
                    'processing_time': time.time() - start_time,
                    'success': False,
                    'error_message': model_result.error_message
                }
                
            self.logger.info(f"   [OK] Model training completed successfully")
            self.logger.info(f"   - Training time: {model_result.training_time:.2f}s")
            self.logger.info(f"   - SVR models: {len(model_result.svr_models)}")
            self.logger.info(f"   - SARIMAX models: {len(model_result.sarimax_model) if model_result.sarimax_model else 0}")
            
            # Save trained models
            try:
                models_output_dir = station_dirs['models']
                saved_model_files = self.model_persistence_service.save_models(
                    model_result, station_name, models_output_dir, format='joblib'
                )
                self.logger.info(f"   💾 Trained models saved successfully")
                self.logger.info(f"   - Saved files: {len(saved_model_files)}")
                for file_type, file_path in saved_model_files.items():
                    self.logger.info(f"     - {file_type}: {file_path}")
            except Exception as e:
                self.logger.warning(f"   [WARNING] Failed to save trained models: {e}")
                # Continue processing even if model saving fails
            
        except Exception as e:
            self.logger.error(f"   [ERROR] Model training failed: {e}")
            return {
                'station_name': station_name,
                'station_index': station_index,
                'total_records': len(station_data),
                'eemd_result': eemd_result,
                'processing_time': time.time() - start_time,
                'success': False,
                'error_message': str(e)
            }
        
        # Step 3: Prediction Generation
        self.logger.info(f"\n🔮 Step 3: Prediction Generation")
        try:
            prediction_result = self.prediction_service.generate_predictions(
                eemd_result, model_result, time_series, self.config
            )
            
            if not prediction_result.success:
                self.logger.error(f"   [ERROR] Prediction generation failed: {prediction_result.error_message}")
                return {
                    'station_name': station_name,
                    'station_index': station_index,
                    'total_records': len(station_data),
                    'eemd_result': eemd_result,
                    'model_result': model_result,
                    'processing_time': time.time() - start_time,
                    'success': False,
                    'error_message': prediction_result.error_message
                }
                
            self.logger.info(f"   [OK] Prediction generation completed successfully")
            self.logger.info(f"   - Prediction steps: {prediction_result.prediction_length}")
            self.logger.info(f"   - Processing time: {prediction_result.processing_time:.2f}s")
            
            # Log prediction quality metrics
            if prediction_result.prediction_quality_metrics:
                metrics = prediction_result.prediction_quality_metrics
                self.logger.info(f"   [INFO] Prediction Quality Metrics:")
                self.logger.info(f"     - Mean consistency: {metrics.get('mean_consistency', 0):.4f}")
                self.logger.info(f"     - Trend consistency: {metrics.get('trend_consistency', 0):.4f}")
                self.logger.info(f"     - Diversity score: {metrics.get('diversity_score', 0):.4f}")
                self.logger.info(f"     - IMFs used: {metrics.get('num_imfs_used', 0)}")
            
            # FASE 3: Step 4: Detect Alerts (only for temp_max and temp_min)
            alerts = []
            if self.alert_detector and self.config.variable_type in ['temp_max', 'temp_min']:
                self.logger.info(f"\n🚨 Step 4: Alert Detection (Fase 3)")
                try:
                    # Extract station code from station name
                    station_code = self._extract_station_code(station_name)
                    self.logger.info(f"   🔍 Extracted station code: {station_code} from '{station_name}'")
                    
                    if station_code:
                        # Check if thresholds exist for this station
                        has_thresholds = self.threshold_loader.has_thresholds(station_code)
                        self.logger.info(f"   📋 Thresholds available for station {station_code}: {has_thresholds}")
                        
                        if not has_thresholds:
                            self.logger.warning(f"   [WARNING] No thresholds found for station {station_code}. Available stations: {list(self.threshold_loader.get_all_station_codes())[:5]}...")
                        
                        # Detect alerts
                        if prediction_result.future_dates is not None and prediction_result.final_prediction is not None:
                            self.logger.info(f"   [INFO] Prediction data: {len(prediction_result.final_prediction)} values, {len(prediction_result.future_dates)} dates")
                            
                            alerts = self.alert_detector.detect_alerts(
                                predictions=pd.Series(prediction_result.final_prediction),
                                future_dates=prediction_result.future_dates,
                                station_code=station_code,
                                variable_type=self.config.variable_type
                            )
                            
                            if alerts:
                                self.logger.info(f"   [ALERT] Detected {len(alerts)} alerts")
                                # Log critical alerts
                                critical_alerts = [a for a in alerts if a.severity.value == 'critical']
                                if critical_alerts:
                                    self.logger.warning(f"   🔴 CRITICAL: {len(critical_alerts)} extreme alerts detected!")
                                    for alert in critical_alerts[:3]:  # Show first 3
                                        self.logger.warning(f"      - {alert.message}")
                                
                                # Save alerts
                                self._save_alerts(alerts, station_name, station_dirs)
                            else:
                                self.logger.info(f"   [OK] No alerts detected (predictions within normal ranges)")
                                # Create empty directory marker to indicate system ran
                                alerts_dir = station_dirs.get('alerts', self.output_dirs.get('alerts'))
                                if alerts_dir:
                                    alerts_dir.mkdir(parents=True, exist_ok=True)
                                    # Create a marker file to show the system ran
                                    marker_file = alerts_dir / ".alert_system_ran.txt"
                                    with open(marker_file, 'w') as f:
                                        f.write(f"Alert system executed for {station_name} on {pd.Timestamp.now()}\n")
                                        f.write(f"No alerts detected - predictions within normal temperature ranges\n")
                        else:
                            self.logger.warning(f"   [WARNING] Cannot detect alerts: missing prediction data")
                            self.logger.warning(f"      future_dates: {prediction_result.future_dates is not None}")
                            self.logger.warning(f"      final_prediction: {prediction_result.final_prediction is not None}")
                    else:
                        self.logger.warning(f"   [WARNING] Cannot extract station code from '{station_name}', skipping alerts")
                except Exception as e:
                    self.logger.error(f"   [ERROR] Alert detection failed: {e}", exc_info=True)
                    # Continue processing even if alert detection fails
            
            # Step 5: Save predictions to CSV
            self.logger.info(f"\n[SAVE] Step 5: Saving Predictions to CSV")
            try:
                self.save_prediction_csv(prediction_result, station_name, station_dirs)
                self.logger.info(f"   [OK] CSV files saved successfully")
            except Exception as e:
                self.logger.error(f"   [ERROR] Failed to save CSV files: {e}")
                # Continue processing even if CSV saving fails
            
            # Generate prediction plots using unified visualization service (if enabled)
            if getattr(self.config, 'enable_plots', True):
                try:
                    plots_output_dir = station_dirs['prediction_plots']
                    self.logger.info(f"   [PLOT] Generating prediction plots for {station_name}...")
                    self.logger.info(f"   [INFO] Output directory: {plots_output_dir}")
                    
                    # Force garbage collection before plot generation
                    import gc
                    gc.collect()
                    
                    # FASE 3: Pass alerts to visualization
                    generated_plots = self.visualization_service.generate_prediction_plots(
                        prediction_result, station_name, plots_output_dir, alerts=alerts
                    )
                    
                    if generated_plots and len(generated_plots) > 0:
                        self.logging_service.log_validation_result(
                            "VisualizationService", "Prediction Plots", True,
                            f"Generated {len(generated_plots)} prediction plots"
                        )
                        self.logger.info(f"   [OK] Generated {len(generated_plots)} prediction plots")
                        for plot_type, plot_path in generated_plots.items():
                            self.logger.info(f"     - {plot_type}: {plot_path}")
                    else:
                        self.logger.warning(f"   [WARNING] No plots generated for {station_name}")
                except Exception as e:
                    self.logger.error(f"   [ERROR] Failed to generate prediction plots for {station_name}: {e}")
                    self.logging_service.log_error(
                        "VisualizationService", "Prediction Plots", str(e),
                        f"Failed to generate prediction plots for {station_name}"
                    )
                    # Continue processing even if plot generation fails
            else:
                self.logger.info(f"   [INFO] Prediction plots generation disabled (--no-plots flag)")
            
        except Exception as e:
            self.logger.error(f"   [ERROR] Prediction generation failed: {e}")
            return {
                'station_name': station_name,
                'station_index': station_index,
                'total_records': len(station_data),
                'eemd_result': eemd_result,
                'model_result': model_result,
                'processing_time': time.time() - start_time,
                'success': False,
                'error_message': str(e)
            }
        
        processing_time = time.time() - start_time
        
        # Memory cleanup
        if adaptive_config and adaptive_config.enable_garbage_collection:
            import gc
            gc.collect()
            self.logger.info(f"   🧹 Memory cleanup completed")
        
        self.logger.info(f"\n[OK] Station {station_name} processing completed in {processing_time:.2f}s")
        
        return {
            'station_name': station_name,
            'station_index': station_index,
            'total_records': len(station_data),
            'eemd_result': eemd_result,
            'model_result': model_result,
            'prediction_result': prediction_result,
            'alerts': alerts,  # FASE 3: Include alerts in results
            'processing_time': processing_time,
            'success': True
        }
    
    def _extract_station_code(self, station_name: str) -> Optional[str]:
        """
        Extract station code from station name.
        
        Station names can be in formats:
        - "{code}_{station_name}_Imputed" (from CSV files)
        - "{code} {station_name}" (from original data)
        - Just the code
        
        Args:
            station_name: Station name or identifier
            
        Returns:
            Station code as string, or None if not found
        """
        try:
            # Try to extract code from beginning of name
            code_match = re.search(r'^(\d+)', str(station_name))
            if code_match:
                return code_match.group(1)
            
            # Try to extract from pattern like "86011_Adrián_Jara_Imputed"
            code_match = re.search(r'^(\d+)_', str(station_name))
            if code_match:
                return code_match.group(1)
            
            # If station_name is already a code, return it
            if str(station_name).isdigit():
                return str(station_name)
            
            return None
        except Exception as e:
            self.logger.warning(f"Failed to extract station code from '{station_name}': {e}")
            return None
    
    def _save_alerts(self, alerts: List[TemperatureAlert], station_name: str, station_dirs: Dict[str, Path]) -> None:
        """
        Save alerts to JSON and CSV files.
        
        Args:
            alerts: List of TemperatureAlert objects
            station_name: Station name
            station_dirs: Dictionary with output directories
        """
        if not alerts:
            return
        
        try:
            alerts_dir = station_dirs.get('alerts', self.output_dirs.get('alerts'))
            if alerts_dir:
                alerts_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            import json
            alerts_json = [alert.to_dict() for alert in alerts]
            json_file = alerts_dir / f"{self._sanitize_station_name(station_name)}_alerts.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(alerts_json, f, indent=2, ensure_ascii=False)
            self.logger.info(f"   💾 Saved {len(alerts)} alerts to JSON: {json_file}")
            
            # Save as CSV
            alerts_df = pd.DataFrame(alerts_json)
            csv_file = alerts_dir / f"{self._sanitize_station_name(station_name)}_alerts.csv"
            alerts_df.to_csv(csv_file, index=False, encoding='utf-8')
            self.logger.info(f"   💾 Saved {len(alerts)} alerts to CSV: {csv_file}")
            
            # Save summary
            summary = self.alert_detector.summarize_alerts(alerts)
            summary_file = alerts_dir / f"{self._sanitize_station_name(station_name)}_alerts_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"   [ERROR] Failed to save alerts: {e}")
    
    def _get_target_column_name(self) -> str:
        """
        Get the target column name based on variable type using configuration service.
        
        Returns:
            Target column name
        """
        try:
            # Import settings to get variable configuration
            from src.config.settings import METEOROLOGICAL_VARIABLES
            
            # Debug logging
            self.logger.info(f"🔍 Getting target column for variable type: {self.config.variable_type}")
            self.logger.info(f"🔍 Available variables: {list(METEOROLOGICAL_VARIABLES.keys())}")
            
            if self.config.variable_type in METEOROLOGICAL_VARIABLES:
                target_column = METEOROLOGICAL_VARIABLES[self.config.variable_type]['value_column']
                self.logger.info(f"[OK] Found target column: {target_column}")
                return target_column
            else:
                self.logging_service.log_warning(
                    "PredictionProcessor", "Configuration", 
                    f"Unknown variable type: {self.config.variable_type}, using default"
                )
                self.logger.warning(f"[WARNING] Unknown variable type: {self.config.variable_type}, using default 'Temperatura'")
                return 'Temperatura'
        except Exception as e:
            self.logging_service.log_warning(
                "PredictionProcessor", "Configuration", 
                f"Failed to get target column name: {e}, using default"
            )
            self.logger.error(f"[ERROR] Error getting target column name: {e}, using default 'Temperatura'")
            return 'Temperatura'
    
    def _log_eemd_results(self, eemd_result: EEMDResult) -> None:
        """
        Log EEMD results in a structured format.
        
        Args:
            eemd_result: EEMD decomposition result
        """
        self.logger.info(f"\n[STATS] EEMD Results Summary:")
        self.logger.info(f"   - Number of IMFs: {eemd_result.num_imfs}")
        self.logger.info(f"   - Best sd_thresh: {eemd_result.best_sd_thresh:.3f}")
        self.logger.info(f"   - Orthogonality score: {eemd_result.orthogonality_score:.4f}")
        self.logger.info(f"   - Top 3 variance explained: {eemd_result.decomposition_quality['top3_variance']:.4f}")
        self.logger.info(f"   - Variance concentration: {eemd_result.decomposition_quality['variance_concentration']:.4f}")
        self.logger.info(f"   - Max correlation: {eemd_result.decomposition_quality['max_correlation']:.4f}")
        self.logger.info(f"   - Mean correlation: {eemd_result.decomposition_quality['mean_correlation']:.4f}")
        self.logger.info(f"   - Reconstruction error: {eemd_result.decomposition_quality['reconstruction_error']:.4f}")
        self.logger.info(f"   - Reconstruction correlation: {eemd_result.decomposition_quality['reconstruction_correlation']:.4f}")
        self.logger.info(f"   - Mean IMF quality: {eemd_result.decomposition_quality['mean_imf_quality']:.4f}")
        self.logger.info(f"   - Seasonality strength: {eemd_result.decomposition_quality['seasonality_strength']:.4f}")
        
        # Display IMF correlations
        self.logger.info(f"\n🔗 IMF Correlations with Original Series:")
        for i, corr in enumerate(eemd_result.correlations):
            self.logger.info(f"   - IMF {i+1}: {corr:.4f}")
        
        # Display variance explained
        self.logger.info(f"\n[INFO] Variance Explained by IMFs:")
        for _, row in eemd_result.variance_explained.iterrows():
            self.logger.info(f"   - IMF {row['imf_index']}: {row['explained_ratio']:.4f} ({row['explained_ratio']*100:.1f}%)")
        
        # Display meteorological patterns
        if 'meteorological_patterns' in eemd_result.decomposition_quality:
            patterns = eemd_result.decomposition_quality['meteorological_patterns']
            self.logger.info(f"\n🌤️ Enhanced Meteorological Pattern Analysis:")
            
            # Basic patterns
            if patterns.get('strongest_annual_imf', -1) >= 0:
                self.logger.info(f"   - Strongest annual seasonality: IMF {patterns['strongest_annual_imf'] + 1}")
            if patterns.get('strongest_monthly_imf', -1) >= 0:
                self.logger.info(f"   - Strongest monthly pattern: IMF {patterns['strongest_monthly_imf'] + 1}")
            if patterns.get('strongest_weekly_imf', -1) >= 0:
                self.logger.info(f"   - Strongest weekly pattern: IMF {patterns['strongest_weekly_imf'] + 1}")
            if patterns.get('strongest_trend_imf', -1) >= 0:
                self.logger.info(f"   - Strongest trend component: IMF {patterns['strongest_trend_imf'] + 1}")
            
            # Extreme events
            if patterns.get('strongest_heat_wave_imf', -1) >= 0:
                self.logger.info(f"   - Strongest heat wave pattern: IMF {patterns['strongest_heat_wave_imf'] + 1}")
            if patterns.get('strongest_cold_spell_imf', -1) >= 0:
                self.logger.info(f"   - Strongest cold spell pattern: IMF {patterns['strongest_cold_spell_imf'] + 1}")
            
            # Climate trends
            if patterns.get('strongest_long_term_trend_imf', -1) >= 0:
                self.logger.info(f"   - Strongest long-term climate trend: IMF {patterns['strongest_long_term_trend_imf'] + 1}")
            if patterns.get('strongest_decadal_trend_imf', -1) >= 0:
                self.logger.info(f"   - Strongest decadal trend: IMF {patterns['strongest_decadal_trend_imf'] + 1}")
            
            if patterns.get('noisiest_imf', -1) >= 0:
                self.logger.info(f"   - Noisiest component: IMF {patterns['noisiest_imf'] + 1}")
        
        # Display IMF classifications for modeling
        if 'imf_classifications' in eemd_result.decomposition_quality:
            classifications = eemd_result.decomposition_quality['imf_classifications']
            self.logger.info(f"\n🎯 IMF Classifications for Modeling:")
            self.logger.info(f"   - High frequency (SVR): IMFs {[i+1 for i in classifications.get('high_frequency_imfs', [])]}")
            self.logger.info(f"   - Low frequency (SARIMAX): IMFs {[i+1 for i in classifications.get('low_frequency_imfs', [])]}")
            self.logger.info(f"   - Seasonal patterns: IMFs {[i+1 for i in classifications.get('seasonal_imfs', [])]}")
            self.logger.info(f"   - Trend components: IMFs {[i+1 for i in classifications.get('trend_imfs', [])]}")
            self.logger.info(f"   - Extreme events: IMFs {[i+1 for i in classifications.get('extreme_event_imfs', [])]}")
            self.logger.info(f"   - Noise components: IMFs {[i+1 for i in classifications.get('noise_imfs', [])]}")
        
        # Check quality acceptability with enhanced criteria
        orthogonality_ok = eemd_result.orthogonality_score <= 0.1
        variance_ok = eemd_result.decomposition_quality['top3_variance'] >= 0.3
        reconstruction_ok = eemd_result.decomposition_quality['reconstruction_error'] <= 1.0
        reconstruction_corr_ok = eemd_result.decomposition_quality['reconstruction_correlation'] >= 0.8
        imf_quality_ok = eemd_result.decomposition_quality['mean_imf_quality'] >= 0.5
        seasonality_ok = eemd_result.decomposition_quality['seasonality_strength'] >= 0.1
        
        # Enhanced quality assessment
        quality_score = 0
        quality_score += 1 if orthogonality_ok else 0
        quality_score += 1 if variance_ok else 0
        quality_score += 1 if reconstruction_ok else 0
        quality_score += 1 if reconstruction_corr_ok else 0
        quality_score += 1 if imf_quality_ok else 0
        quality_score += 1 if seasonality_ok else 0
        
        is_acceptable = quality_score >= 4  # At least 4 out of 6 criteria met
        
        self.logger.info(f"\n[OK] Enhanced Quality Assessment:")
        self.logger.info(f"   - Orthogonality: {'OK' if orthogonality_ok else 'POOR'} ({eemd_result.orthogonality_score:.4f})")
        self.logger.info(f"   - Variance concentration: {'OK' if variance_ok else 'POOR'} ({eemd_result.decomposition_quality['top3_variance']:.4f})")
        self.logger.info(f"   - Reconstruction error: {'OK' if reconstruction_ok else 'POOR'} ({eemd_result.decomposition_quality['reconstruction_error']:.4f})")
        self.logger.info(f"   - Reconstruction correlation: {'OK' if reconstruction_corr_ok else 'POOR'} ({eemd_result.decomposition_quality['reconstruction_correlation']:.4f})")
        self.logger.info(f"   - Mean IMF quality: {'OK' if imf_quality_ok else 'POOR'} ({eemd_result.decomposition_quality['mean_imf_quality']:.4f})")
        self.logger.info(f"   - Seasonality strength: {'OK' if seasonality_ok else 'POOR'} ({eemd_result.decomposition_quality['seasonality_strength']:.4f})")
        self.logger.info(f"   - Quality score: {quality_score}/6")
        self.logger.info(f"   - Overall quality: {'ACCEPTABLE' if is_acceptable else 'POOR'}")
        
        if not is_acceptable:
            self.logger.warning(f"   - [WARNING] Consider adjusting EEMD parameters for better quality")
            if not orthogonality_ok:
                self.logger.warning(f"   - 💡 Try increasing ensemble size or adjusting noise factor")
            if not reconstruction_ok:
                self.logger.warning(f"   - 💡 Try different sd_thresh values or noise factor")
            if not seasonality_ok:
                self.logger.warning(f"   - 💡 Series may not have strong seasonal patterns")
    
    def save_processing_summary(self, summary: Dict[str, Any]) -> None:
        """
        Save processing summary to files.
        
        Args:
            summary: Processing summary dictionary
        """
        try:
            # Save summary as CSV
            summary_df = pd.DataFrame([{
                'total_stations': summary['total_stations'],
                'successful_stations': summary['successful_stations'],
                'failed_stations': summary['failed_stations'],
                'success_rate': summary['success_rate'],
                'variable_type': self.config.variable_type,
                'timestamp': pd.Timestamp.now()
            }])
            
            summary_file = self.output_dirs['reports'] / f"prediction_summary_{self.config.variable_type}.csv"
            summary_df.to_csv(summary_file, index=False)
            
            # Save detailed results
            if summary['results']:
                results_df = pd.DataFrame(summary['results'])
                results_file = self.output_dirs['reports'] / f"prediction_results_{self.config.variable_type}.csv"
                results_df.to_csv(results_file, index=False)
            
            self.logger.info(f"Processing summary saved to {self.output_dirs['reports']}")
            
        except Exception as e:
            self.logger.error(f"Failed to save processing summary: {e}")
            raise ProcessingError(f"Failed to save processing summary: {e}")
    
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """
        List all saved models for the current variable type.
        
        Returns:
            List of saved model information
        """
        try:
            models_dir = self.output_dirs['models']
            return self.model_persistence_service.list_saved_models(models_dir)
        except Exception as e:
            self.logger.error(f"Failed to list saved models: {e}")
            return []
    
    def load_saved_models(self, station_name: str) -> Tuple[ModelTrainingResult, Dict[str, Any]]:
        """
        Load saved models for a specific station.
        
        Args:
            station_name: Name of the station
            
        Returns:
            Tuple with (ModelTrainingResult, metadata)
        """
        try:
            models_dir = self.output_dirs['models']
            return self.model_persistence_service.load_models(station_name, models_dir)
        except Exception as e:
            self.logger.error(f"Failed to load models for station {station_name}: {e}")
            raise
    
    def save_prediction_csv(self, prediction_result: PredictionResult, station_name: str, station_dirs: Dict[str, Path]) -> None:
        """
        Save prediction results to CSV file for a specific station.
        
        Args:
            prediction_result: Prediction result containing all prediction data
            station_name: Name of the station
            station_dirs: Station-specific output directories
        """
        try:
            self.logger.info(f"💾 Saving prediction CSV for station: {station_name}")
            
            # Create comprehensive prediction DataFrame
            prediction_data = []
            
            # Get target column name for proper labeling
            target_column = self._get_target_column_name()
            
            # Add original data (last 30 days for context)
            original_values = prediction_result.original_data.iloc[:, 0]  # First column
            context_days = min(30, len(original_values))
            
            # CRITICAL FIX: Use real dates from original data index if available
            if isinstance(original_values.index, pd.DatetimeIndex):
                # Use actual dates from the original series
                historical_dates = original_values.index[-context_days:]
                self.logger.debug(f"Using real dates from original data for CSV: {historical_dates[0]} to {historical_dates[-1]}")
            elif len(prediction_result.future_dates) > 0:
                # Fallback: generate dates backwards from first prediction date
                last_future_date = prediction_result.future_dates[0]
                historical_dates = pd.date_range(
                    end=last_future_date - pd.Timedelta(days=1),
                    periods=context_days,
                    freq='D'
                )
                self.logger.debug(f"Generated historical dates backwards from first prediction for CSV")
            else:
                # Last resort: use current date
                today = datetime.now().date()
                historical_dates = pd.date_range(
                    end=today,
                    periods=context_days,
                    freq='D'
                )
                self.logger.warning("Using current date as fallback for historical dates in CSV")
            
            # Add last 30 days of original data for context with real dates
            for i, date in enumerate(historical_dates):
                value_idx = len(original_values) - context_days + i
                if value_idx >= 0 and value_idx < len(original_values):
                    prediction_data.append({
                        'date': date,
                        'type': 'historical',
                        'value': original_values.iloc[value_idx],
                    'imf_1': None,
                    'imf_2': None,
                    'imf_3': None,
                    'imf_4': None,
                    'imf_5': None,
                    'imf_6': None,
                    'imf_7': None,
                    'imf_8': None,
                    'lower_ci': None,
                    'upper_ci': None,
                    'confidence_level': None
                })
            
            # Add prediction data
            for i, (date, pred_value) in enumerate(zip(prediction_result.future_dates, prediction_result.final_prediction)):
                # Get IMF predictions for this step
                imf_values = {}
                for imf_idx, imf_pred in prediction_result.imf_predictions.items():
                    if i < len(imf_pred):
                        imf_values[f'imf_{imf_idx + 1}'] = imf_pred[i]
                    else:
                        imf_values[f'imf_{imf_idx + 1}'] = None
                
                # Get confidence intervals
                lower_ci = None
                upper_ci = None
                confidence_level = None
                if prediction_result.confidence_intervals:
                    lower_ci = prediction_result.confidence_intervals[0][i] if i < len(prediction_result.confidence_intervals[0]) else None
                    upper_ci = prediction_result.confidence_intervals[1][i] if i < len(prediction_result.confidence_intervals[1]) else None
                    confidence_level = 0.95  # 95% confidence interval
                
                prediction_data.append({
                    'date': date,
                    'type': 'prediction',
                    'value': pred_value,
                    'imf_1': imf_values.get('imf_1'),
                    'imf_2': imf_values.get('imf_2'),
                    'imf_3': imf_values.get('imf_3'),
                    'imf_4': imf_values.get('imf_4'),
                    'imf_5': imf_values.get('imf_5'),
                    'imf_6': imf_values.get('imf_6'),
                    'imf_7': imf_values.get('imf_7'),
                    'imf_8': imf_values.get('imf_8'),
                    'lower_ci': lower_ci,
                    'upper_ci': upper_ci,
                    'confidence_level': confidence_level
                })
            
            # Create DataFrame
            prediction_df = pd.DataFrame(prediction_data)
            
            # Add metadata columns
            prediction_df['station_name'] = station_name
            prediction_df['variable_type'] = self.config.variable_type
            prediction_df['prediction_date'] = pd.Timestamp.now()
            prediction_df['prediction_steps'] = len(prediction_result.final_prediction)
            prediction_df['processing_time'] = prediction_result.processing_time
            
            # Add quality metrics
            if prediction_result.prediction_quality_metrics:
                metrics = prediction_result.prediction_quality_metrics
                prediction_df['mean_consistency'] = metrics.get('mean_consistency', None)
                prediction_df['trend_consistency'] = metrics.get('trend_consistency', None)
                prediction_df['diversity_score'] = metrics.get('diversity_score', None)
                prediction_df['num_imfs_used'] = metrics.get('num_imfs_used', None)
            
            # Reorder columns for better readability
            column_order = [
                'date', 'type', 'value', 'station_name', 'variable_type',
                'imf_1', 'imf_2', 'imf_3', 'imf_4', 'imf_5', 'imf_6', 'imf_7', 'imf_8',
                'lower_ci', 'upper_ci', 'confidence_level',
                'mean_consistency', 'trend_consistency', 'diversity_score', 'num_imfs_used',
                'prediction_date', 'prediction_steps', 'processing_time'
            ]
            
            # Only include columns that exist
            existing_columns = [col for col in column_order if col in prediction_df.columns]
            prediction_df = prediction_df[existing_columns]
            
            # Save to CSV
            csv_filename = f"{station_name}_predictions_{self.config.variable_type}.csv"
            csv_filepath = station_dirs['csv_files'] / csv_filename
            prediction_df.to_csv(csv_filepath, index=False)
            
            self.logger.info(f"   [OK] Prediction CSV saved: {csv_filepath}")
            self.logger.info(f"   [INFO] Records: {len(prediction_df)} (Historical: {context_days}, Predictions: {len(prediction_result.final_prediction)})")
            self.logger.info(f"   [STATS] Date range: {prediction_df['date'].min()} to {prediction_df['date'].max()}")
            
            # Save summary statistics
            self._save_prediction_summary_stats(prediction_df, station_name, station_dirs)
            
        except Exception as e:
            self.logger.error(f"Failed to save prediction CSV for {station_name}: {e}")
            raise ProcessingError(f"Failed to save prediction CSV for {station_name}: {e}")
    
    def _save_prediction_summary_stats(self, prediction_df: pd.DataFrame, station_name: str, station_dirs: Dict[str, Path]) -> None:
        """
        Save summary statistics for predictions.
        
        Args:
            prediction_df: DataFrame with prediction data
            station_name: Name of the station
            station_dirs: Station-specific output directories
        """
        try:
            # Filter only prediction data (not historical)
            pred_only = prediction_df[prediction_df['type'] == 'prediction'].copy()
            
            if pred_only.empty:
                self.logger.warning(f"No prediction data found for {station_name}")
                return
            
            # Ensure date column is properly converted to datetime
            pred_only['date'] = pd.to_datetime(pred_only['date'])
            
            # Calculate summary statistics
            summary_stats = {
                'station_name': station_name,
                'variable_type': self.config.variable_type,
                'prediction_date': pd.Timestamp.now(),
                'total_predictions': len(pred_only),
                'prediction_start_date': pred_only['date'].min(),
                'prediction_end_date': pred_only['date'].max(),
                'prediction_mean': pred_only['value'].mean(),
                'prediction_std': pred_only['value'].std(),
                'prediction_min': pred_only['value'].min(),
                'prediction_max': pred_only['value'].max(),
                'prediction_median': pred_only['value'].median(),
                'mean_consistency': pred_only['mean_consistency'].iloc[0] if 'mean_consistency' in pred_only.columns else None,
                'trend_consistency': pred_only['trend_consistency'].iloc[0] if 'trend_consistency' in pred_only.columns else None,
                'diversity_score': pred_only['diversity_score'].iloc[0] if 'diversity_score' in pred_only.columns else None,
                'num_imfs_used': pred_only['num_imfs_used'].iloc[0] if 'num_imfs_used' in pred_only.columns else None,
                'processing_time': pred_only['processing_time'].iloc[0] if 'processing_time' in pred_only.columns else None
            }
            
            # Save summary statistics
            summary_filename = f"{station_name}_prediction_summary_{self.config.variable_type}.csv"
            summary_filepath = station_dirs['csv_files'] / summary_filename
            
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_csv(summary_filepath, index=False)
            
            self.logger.info(f"   📋 Summary stats saved: {summary_filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save prediction summary stats for {station_name}: {e}")
            # Don't raise exception here as it's not critical
    
    # ============================================================================
    # IVariableAgnosticProcessor Interface Implementation
    # ============================================================================
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validar datos de entrada genéricos.
        
        Args:
            data: DataFrame con datos temporales
            
        Returns:
            True si los datos son válidos
        """
        try:
            if self._current_unified_config is None:
                # Crear configuración por defecto si no hay una configurada
                self._current_unified_config = self.unified_factory.create_default_config()
            
            # Validar estructura y calidad de datos
            structure_valid = self.unified_validator.validate_data_structure(data, self._current_unified_config)
            quality_valid = self.unified_validator.validate_data_quality(data, self._current_unified_config)
            
            if structure_valid and quality_valid:
                self.logger.info("Data validation passed")
                return True
            else:
                self.logger.error("Data validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return False
    
    def preprocess_data(self, data: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
        """
        Preprocesar datos genéricos.
        
        Args:
            data: DataFrame con datos originales
            config: Configuración de procesamiento
            
        Returns:
            DataFrame preprocesado
        """
        try:
            self.logger.info("Starting data preprocessing")
            
            # Crear copia de los datos
            processed_data = data.copy()
            
            # Asegurar que la columna de fecha sea datetime
            if config.date_column in processed_data.columns:
                processed_data[config.date_column] = pd.to_datetime(processed_data[config.date_column])
            
            # Ordenar por fecha
            if config.date_column in processed_data.columns:
                processed_data = processed_data.sort_values(config.date_column).reset_index(drop=True)
            
            # Manejar valores faltantes en la columna objetivo
            if config.target_column in processed_data.columns:
                target_series = processed_data[config.target_column]
                missing_count = target_series.isnull().sum()
                
                if missing_count > 0:
                    self.logger.info(f"Found {missing_count} missing values in target column")
                    # Interpolación lineal para valores faltantes
                    processed_data[config.target_column] = target_series.interpolate(method='linear')
            
            # Aplicar downsampling si está habilitado
            if config.enable_downsampling and len(processed_data) > config.downsampling_threshold:
                original_size = len(processed_data)
                step = len(processed_data) // config.downsampling_threshold
                processed_data = processed_data.iloc[::step].reset_index(drop=True)
                self.logger.info(f"Applied downsampling: {original_size} -> {len(processed_data)} points")
            
            self.logger.info(f"Data preprocessing completed: {len(processed_data)} points")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Data preprocessing error: {e}")
            raise
    
    def decompose_series(self, series: pd.Series, config: ProcessingConfig) -> Any:
        """
        Descomponer serie temporal genérica.
        
        Args:
            series: Serie temporal a descomponer
            config: Configuración de procesamiento
            
        Returns:
            Resultado de descomposición
        """
        try:
            self.logger.info("Starting series decomposition")
            
            # Configurar parámetros EEMD
            eemd_params = {
                'ensembles': config.eemd_ensembles,
                'noise_factor': config.eemd_noise_factor,
                'sd_thresh_range': config.eemd_sd_thresh_range,
                'max_imfs': config.eemd_max_imfs,
                'quality_threshold': config.eemd_quality_threshold
            }
            
            # Realizar descomposición usando el servicio EEMD existente
            decomposition_result = self.eemd_service.decompose_series(series, **eemd_params)
            
            self.logger.info("Series decomposition completed")
            return decomposition_result
            
        except Exception as e:
            self.logger.error(f"Series decomposition error: {e}")
            raise
    
    def classify_components(self, decomposition_result: Any, config: ProcessingConfig) -> Dict[str, List[int]]:
        """
        Clasificar componentes de la descomposición.
        
        Args:
            decomposition_result: Resultado de descomposición
            config: Configuración de procesamiento
            
        Returns:
            Diccionario con clasificación de componentes
        """
        try:
            self.logger.info("Starting component classification")
            
            # Usar el método de clasificación existente del servicio EEMD
            if hasattr(decomposition_result, 'classify_imfs'):
                classifications = decomposition_result.classify_imfs()
            else:
                # Clasificación por defecto si no hay método específico
                num_imfs = decomposition_result.imfs.shape[1] if hasattr(decomposition_result, 'imfs') else 0
                classifications = {
                    'sarimax_imfs': [num_imfs // 2] if num_imfs > 0 else [],
                    'svr_imfs': list(range(1, min(4, num_imfs))) if num_imfs > 1 else [],
                    'extrapolation_imfs': list(range(max(4, num_imfs - 2), num_imfs)) if num_imfs > 4 else [],
                    'noise_imfs': [0] if num_imfs > 0 else []
                }
            
            self.logger.info(f"Component classification completed: {classifications}")
            return classifications
            
        except Exception as e:
            self.logger.error(f"Component classification error: {e}")
            raise
    
    def train_models(self, 
                    decomposition_result: Any, 
                    classifications: Dict[str, List[int]], 
                    config: ProcessingConfig) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Entrenar modelos genéricos.
        
        Args:
            decomposition_result: Resultado de descomposición
            classifications: Clasificación de componentes
            config: Configuración de procesamiento
            
        Returns:
            Tupla con modelos entrenados y métricas
        """
        try:
            self.logger.info("Starting model training")
            
            # Configurar parámetros de entrenamiento
            training_params = {
                'svr_lags': config.svr_lags,
                'svr_test_size': config.svr_test_size,
                'sarimax_max_iter': config.sarimax_max_iter,
                'sarimax_data_limit_years': config.sarimax_data_limit_years
            }
            
            # Entrenar modelos usando el servicio híbrido existente
            trained_models, model_metrics = self.hybrid_model_service.train_models(
                decomposition_result, classifications, config
            )
            
            self.logger.info("Model training completed")
            return trained_models, model_metrics
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            raise
    
    def generate_predictions(self, 
                           models: Dict[str, Any], 
                           decomposition_result: Any,
                           config: ProcessingConfig) -> Tuple[pd.Series, Optional[Tuple[pd.Series, pd.Series]]]:
        """
        Generar predicciones genéricas.
        
        Args:
            models: Modelos entrenados
            decomposition_result: Resultado de descomposición
            config: Configuración de procesamiento
            
        Returns:
            Tupla con predicciones e intervalos de confianza
        """
        try:
            self.logger.info("Starting prediction generation")
            
            # Calcular pasos de predicción
            prediction_steps = int(len(self._current_unified_data) * config.prediction_steps_ratio)
            
            # Configurar parámetros de predicción
            prediction_params = {
                'prediction_steps': prediction_steps,
                'confidence_level': config.confidence_level
            }
            
            # Generar predicciones usando el servicio de predicción existente
            predictions, confidence_intervals = self.prediction_service.generate_predictions(
                models, decomposition_result, **prediction_params
            )
            
            self.logger.info(f"Prediction generation completed: {len(predictions)} steps")
            return predictions, confidence_intervals
            
        except Exception as e:
            self.logger.error(f"Prediction generation error: {e}")
            raise
    
    def evaluate_quality(self, 
                        input_data: pd.DataFrame, 
                        predictions: pd.Series, 
                        config: ProcessingConfig) -> Dict[str, float]:
        """
        Evaluar calidad de las predicciones.
        
        Args:
            input_data: Datos de entrada originales
            predictions: Predicciones generadas
            config: Configuración de procesamiento
            
        Returns:
            Diccionario con métricas de calidad
        """
        try:
            self.logger.info("Starting quality evaluation")
            
            # Calcular métricas básicas
            target_series = input_data[config.target_column]
            
            # Métricas de consistencia
            mean_consistency = abs(predictions.mean() - target_series.mean()) / target_series.std()
            
            # Métricas de tendencia
            target_trend = target_series.diff().mean()
            prediction_trend = predictions.diff().mean()
            trend_consistency = abs(prediction_trend - target_trend) / abs(target_trend) if target_trend != 0 else 0
            
            # Métricas de diversidad
            diversity_score = predictions.std() / target_series.std()
            
            # Calcular score de calidad general
            quality_score = max(0, 1 - (mean_consistency + trend_consistency) / 2)
            
            quality_metrics = {
                'mean_consistency': mean_consistency,
                'trend_consistency': trend_consistency,
                'diversity_score': diversity_score,
                'quality_score': quality_score,
                'prediction_length': len(predictions),
                'target_mean': target_series.mean(),
                'prediction_mean': predictions.mean(),
                'target_std': target_series.std(),
                'prediction_std': predictions.std()
            }
            
            self.logger.info(f"Quality evaluation completed: score = {quality_score:.4f}")
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Quality evaluation error: {e}")
            raise
    
    def save_results(self, 
                    result: ProcessingResult, 
                    output_dir: Path, 
                    config: ProcessingConfig) -> Dict[str, str]:
        """
        Guardar resultados genéricos.
        
        Args:
            result: Resultado de procesamiento
            output_dir: Directorio de salida
            config: Configuración de procesamiento
            
        Returns:
            Diccionario con rutas de archivos guardados
        """
        try:
            self.logger.info("Starting results saving")
            
            # Crear directorio de salida
            output_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = {}
            
            # Guardar predicciones como CSV
            if result.predictions is not None:
                predictions_file = output_dir / "predictions.csv"
                result.predictions.to_csv(predictions_file)
                saved_files['predictions'] = str(predictions_file)
            
            # Guardar configuración
            config_file = output_dir / "config.json"
            import json
            with open(config_file, 'w') as f:
                json.dump(config.__dict__, f, indent=2, default=str)
            saved_files['config'] = str(config_file)
            
            # Guardar métricas de calidad
            if result.prediction_metrics:
                metrics_file = output_dir / "quality_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(result.prediction_metrics, f, indent=2)
                saved_files['metrics'] = str(metrics_file)
            
            # Guardar metadatos del procesamiento
            metadata = {
                'processing_time': result.processing_time,
                'memory_usage_mb': result.memory_usage_mb,
                'quality_score': result.quality_score,
                'success': result.success,
                'error_message': result.error_message
            }
            
            metadata_file = output_dir / "processing_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            saved_files['metadata'] = str(metadata_file)
            
            self.logger.info(f"Results saving completed: {len(saved_files)} files")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Results saving error: {e}")
            raise
    
    def process_data(self, 
                    data: pd.DataFrame, 
                    config: Optional[ProcessingConfig] = None,
                    output_dir: Optional[Path] = None) -> ProcessingResult:
        """
        Procesar datos completos usando el sistema unificado.
        
        Args:
            data: DataFrame con datos de entrada
            config: Configuración de procesamiento (opcional)
            output_dir: Directorio de salida (opcional)
            
        Returns:
            Resultado del procesamiento
        """
        try:
            # Inicializar tiempo de procesamiento
            self._processing_start_time = time.time()
            
            # Configurar configuración
            if config is None:
                config = self.unified_factory.create_adaptive_config(data)
            
            self._current_unified_config = config
            self._current_unified_data = data
            
            # Registrar inicio
            self.unified_logger.log_processing_start(config)
            
            # Crear resultado
            result = ProcessingResult(
                input_data=data,
                config=config
            )
            
            try:
                # Paso 1: Validar datos
                if not self.validate_data(data):
                    raise ValueError("Data validation failed")
                
                # Paso 2: Preprocesar datos
                processed_data = self.preprocess_data(data, config)
                self.unified_logger.log_processing_step("Data Preprocessing", {
                    'original_size': len(data),
                    'processed_size': len(processed_data)
                })
                
                # Paso 3: Descomponer serie
                target_series = processed_data[config.target_column]
                decomposition_result = self.decompose_series(target_series, config)
                result.eemd_result = decomposition_result
                
                self.unified_logger.log_processing_step("Series Decomposition", {
                    'imfs_count': decomposition_result.imfs.shape[1] if hasattr(decomposition_result, 'imfs') else 0
                })
                
                # Paso 4: Clasificar componentes
                classifications = self.classify_components(decomposition_result, config)
                result.imf_classifications = classifications
                
                self.unified_logger.log_processing_step("Component Classification", {
                    'sarimax_count': len(classifications.get('sarimax_imfs', [])),
                    'svr_count': len(classifications.get('svr_imfs', [])),
                    'extrapolation_count': len(classifications.get('extrapolation_imfs', []))
                })
                
                # Paso 5: Entrenar modelos
                trained_models, model_metrics = self.train_models(decomposition_result, classifications, config)
                result.trained_models = trained_models
                result.model_metrics = model_metrics
                
                self.unified_logger.log_processing_step("Model Training", {
                    'models_count': len(trained_models),
                    'training_time': model_metrics.get('training_time', 0)
                })
                
                # Paso 6: Generar predicciones
                predictions, confidence_intervals = self.generate_predictions(trained_models, decomposition_result, config)
                result.predictions = predictions
                result.confidence_intervals = confidence_intervals
                
                self.unified_logger.log_processing_step("Prediction Generation", {
                    'prediction_steps': len(predictions)
                })
                
                # Paso 7: Evaluar calidad
                quality_metrics = self.evaluate_quality(data, predictions, config)
                result.prediction_metrics = quality_metrics
                result.quality_score = quality_metrics.get('quality_score', 0.0)
                
                # Paso 8: Guardar resultados
                if output_dir:
                    saved_files = self.save_results(result, output_dir, config)
                    result.output_files = saved_files
                
                # Marcar como exitoso
                result.success = True
                
            except Exception as e:
                result.success = False
                result.error_message = str(e)
                self.unified_logger.log_error(e, {'config': config.__dict__})
                raise
            
            finally:
                # Calcular tiempo de procesamiento
                if self._processing_start_time:
                    result.processing_time = time.time() - self._processing_start_time
                
                # Calcular uso de memoria
                memory_info = self.unified_memory_manager.check_memory_usage()
                result.memory_usage_mb = memory_info.get('used_mb', 0.0)
                
                # Limpiar memoria
                self.unified_memory_manager.cleanup_memory()
                
                # Registrar completación
                self.unified_logger.log_processing_complete(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            raise
    
    def get_available_presets(self) -> Dict[str, str]:
        """
        Obtener presets disponibles.
        
        Returns:
            Diccionario con presets disponibles
        """
        return self.unified_factory.get_available_presets()
    
    def create_config_from_preset(self, preset_name: str) -> ProcessingConfig:
        """
        Crear configuración desde preset.
        
        Args:
            preset_name: Nombre del preset
            
        Returns:
            Configuración del preset
        """
        return self.unified_factory.create_config_from_preset(preset_name)
    
    def estimate_memory_requirements(self, data: pd.DataFrame, config: ProcessingConfig) -> float:
        """
        Estimar requerimientos de memoria.
        
        Args:
            data: DataFrame con datos
            config: Configuración de procesamiento
            
        Returns:
            Estimación de memoria en MB
        """
        return self.unified_memory_manager.estimate_memory_requirements(data, config) 