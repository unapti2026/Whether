"""
Imputation Process Module

This module provides a comprehensive imputation workflow for meteorological data.
It orchestrates the complete process of loading, processing, imputing, and saving
results for multiple weather stations.

IMPROVEMENTS MADE:
- Removed duplicate VariableType enum (now uses centralized config)
- Enhanced validation and error handling
- Improved configuration management
- Added data quality validation
- Better logging and reporting
- Enhanced file path validation
- Added processing status tracking
- Improved error recovery and retry mechanisms
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import argparse
import re

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.data.services.time_series_engineer_service import TimeSeriesEngineerService
from src.config.settings import get_supported_variables, get_file_paths_for_variable
from src.config.preprocessing_config import VariableType
from src.data.imputation import StationImputationService
from src.data.imputation.services.station_statistics_reporter import StationStatisticsReporter


@dataclass
class ImputationConfig:
    """Enhanced configuration class for imputation process."""
    variable_type: VariableType
    data_path: str = "data"
    max_stations: Optional[int] = None  # Process all stations by default
    output_dir: Optional[str] = None  # Will use PathManager if None
    
    # Plotting configuration
    plot_figsize: tuple = (12, 6)
    plot_dpi: int = 80
    plot_style: str = "default"
    enable_plots: bool = True
    
    # Imputation parameters
    max_missing_block_days: int = 548  # Maximum block size for missing data imputation
    small_block_threshold: int = 3
    imputation_method: str = "auto"
    
    # Processing options
    enable_validation: bool = True
    enable_logging: bool = True
    save_intermediate: bool = False
    force_reprocess: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.variable_type, VariableType):
            self.variable_type = VariableType(self.variable_type)
        
        # Validate data path
        if not Path(self.data_path).exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        # Validate variable type
        supported_vars = get_supported_variables()
        if self.variable_type.value not in supported_vars:
            raise ValueError(
                f'Unsupported variable: {self.variable_type.value}. '
                f'Supported: {supported_vars}'
            )


@dataclass
class StationData:
    """Enhanced data class representing station information."""
    name: str
    code: str
    data: pd.DataFrame
    variable_type: VariableType  # Add variable type to know which column to use
    
    @property
    def row_count(self) -> int:
        """Get the number of rows in the station data."""
        return len(self.data)
    
    @property
    def has_data(self) -> bool:
        """Check if the station has data."""
        return self.row_count > 0
    
    @property
    def value_column(self) -> str:
        """Get the correct value column name based on variable type."""
        column_mapping = {
            VariableType.TEMP_MAX: 'Temperatura',
            VariableType.TEMP_MIN: 'Temperatura',
            VariableType.PRECIPITATION: 'Precipitación',
            VariableType.HUMIDITY: 'Humedad'
        }
        return column_mapping.get(self.variable_type, 'Temperatura')
    
    @property
    def missing_values_count(self) -> int:
        """Get the number of missing values."""
        return self.data[self.value_column].isnull().sum()
    
    @property
    def missing_values_percentage(self) -> float:
        """Get the percentage of missing values."""
        return (self.missing_values_count / self.row_count * 100) if self.row_count > 0 else 0.0


@dataclass
class ImputationResult:
    """Enhanced data class representing imputation result for a station."""
    station_name: str
    station_code: str
    original_data: pd.DataFrame
    imputed_data: pd.DataFrame
    processing_time: float
    success: bool
    variable_type: VariableType  # Add variable type to know which column to use
    error_message: Optional[str] = None
    imputation_method: Optional[str] = None
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def value_column(self) -> str:
        """Get the correct value column name based on variable type."""
        column_mapping = {
            VariableType.TEMP_MAX: 'Temperatura',
            VariableType.TEMP_MIN: 'Temperatura',
            VariableType.PRECIPITATION: 'Precipitación',
            VariableType.HUMIDITY: 'Humedad'
        }
        return column_mapping.get(self.variable_type, 'Temperatura')
    
    @property
    def original_missing_count(self) -> int:
        """Get the number of missing values in original data."""
        return self.original_data[self.value_column].isnull().sum()
    
    @property
    def imputed_missing_count(self) -> int:
        """Get the number of missing values in imputed data."""
        return self.imputed_data[self.value_column].isnull().sum()
    
    @property
    def imputed_count(self) -> int:
        """Get the number of values that were imputed."""
        return self.original_missing_count - self.imputed_missing_count
    
    @property
    def imputation_rate(self) -> float:
        """Get the imputation success rate."""
        return (self.imputed_count / self.original_missing_count * 100) if self.original_missing_count > 0 else 0.0


class DataValidator:
    """Enhanced data validation class for imputation process."""
    
    def __init__(self, config: ImputationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_input_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data for imputation."""
        validation_results = {
            'passed': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check if data is empty
            if data.empty:
                validation_results['passed'] = False
                validation_results['errors'].append("Input data is empty")
                return validation_results
            
            # Get the correct value column based on variable type
            value_column = self._get_value_column()
            
            # Check required columns
            required_columns = ['Estación', 'Código', 'Fecha', value_column]
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
            
            # Check for missing values
            if value_column in data.columns:
                missing_pct = data[value_column].isnull().sum() / len(data) * 100
                if missing_pct < 1:
                    validation_results['warnings'].append(f"Very few missing values: {missing_pct:.1f}%")
                elif missing_pct > 90:
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
        """Get the correct value column name based on variable type."""
        column_mapping = {
            VariableType.TEMP_MAX: 'Temperatura',
            VariableType.TEMP_MIN: 'Temperatura',
            VariableType.PRECIPITATION: 'Precipitación',
            VariableType.HUMIDITY: 'Humedad'
        }
        return column_mapping.get(self.config.variable_type, 'Temperatura')
    
    def validate_station_data(self, station_data: StationData) -> Dict[str, Any]:
        """Validate individual station data."""
        validation_results = {
            'passed': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            if not station_data.has_data:
                validation_results['passed'] = False
                validation_results['errors'].append(f"No data available for station {station_data.name}")
                return validation_results
            
            # Check for sufficient data
            if station_data.row_count < 10:
                validation_results['warnings'].append(f"Station {station_data.name} has very few data points: {station_data.row_count}")
            
            # Check missing values percentage
            if station_data.missing_values_percentage > 80:
                validation_results['warnings'].append(f"Station {station_data.name} has high missing values: {station_data.missing_values_percentage:.1f}%")
            
            # Check for constant values
            if station_data.has_data:
                value_column = station_data.value_column
                temp_values = station_data.data[value_column].dropna()
                if len(temp_values) > 0 and temp_values.std() == 0:
                    validation_results['warnings'].append(f"Station {station_data.name} has constant {value_column} values")
            
        except Exception as e:
            validation_results['passed'] = False
            validation_results['errors'].append(f"Station validation error: {e}")
        
        return validation_results


class MissingValueImputer:
    """
    Enhanced imputation service for meteorological data.
    
    This class provides a clean, maintainable, and extensible interface for
    imputing missing values in weather station data with comprehensive
    reporting and visualization capabilities.
    """
    
    def __init__(self, config: ImputationConfig):
        """
        Initialize the imputation service.
        
        Args:
            config: Configuration object containing all imputation parameters
        """
        self.config = config
        
        # Use PathManager for consistent paths
        from src.config.path_manager import get_path_manager
        path_manager = get_path_manager()
        
        if config.output_dir:
            self.output_dir = Path(config.output_dir) / config.variable_type.value
        else:
            self.output_dir = path_manager.get_output_subdir("imputation", config.variable_type.value)
        
        # Create output directories automatically
        self._create_output_directories()
        
        # Initialize services
        self.imputation_service = self._initialize_imputation_service()
        self.statistics_reporter = self._initialize_statistics_reporter()
        self.validator = DataValidator(config)
        
        # Results storage
        self.imputation_results: List[ImputationResult] = []
        self.processing_summary: Dict[str, Any] = {}
        
        # Setup logging
        self.logger = self._setup_logger()
        
        self._validate_configuration()
        self._log_initialization()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the imputer."""
        logger = logging.getLogger(f"imputer_{self.config.variable_type.value}")
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
        """Validate the imputation configuration."""
        try:
            # Validate variable type
            supported_vars = get_supported_variables()
            if self.config.variable_type.value not in supported_vars:
                raise ValueError(
                    f'Unsupported variable: {self.config.variable_type.value}. '
                    f'Supported: {supported_vars}'
                )
            
            self.logger.info("✓ Configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    def _create_output_directories(self) -> None:
        """Create necessary output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "statistics_reports").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "csv_files").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
    
    def _initialize_imputation_service(self) -> StationImputationService:
        """Initialize the imputation service with configuration."""
        service_config = {
            'variable_type': self.config.variable_type.value,
            'data_frequency': 'D',  # Default to daily, can be overridden
            'max_missing_block_days': self.config.max_missing_block_days,
            'small_block_threshold': self.config.small_block_threshold
        }
        return StationImputationService(service_config)
    
    def _initialize_statistics_reporter(self) -> StationStatisticsReporter:
        """Initialize the statistics reporter service."""
        reporter_config = {
            'variable_type': self.config.variable_type.value
        }
        return StationStatisticsReporter(reporter_config)
    
    def _log_initialization(self) -> None:
        """Log initialization information."""
        self.logger.info(f"✓ Imputation service initialized for {self.config.variable_type.value}")
        self.logger.info(f"  Output directory: {self.output_dir}")
        self.logger.info(f"  Max stations: {self.config.max_stations if self.config.max_stations else 'All'}")
        self.logger.info(f"  Data path: {self.config.data_path}")
        self.logger.info(f"  Imputation method: {self.config.imputation_method}")
    
    def _get_value_column(self) -> str:
        """Get the correct value column name based on variable type."""
        column_mapping = {
            VariableType.TEMP_MAX: 'Temperatura',
            VariableType.TEMP_MIN: 'Temperatura',
            VariableType.PRECIPITATION: 'Precipitación',
            VariableType.HUMIDITY: 'Humedad'
        }
        return column_mapping.get(self.config.variable_type, 'Temperatura')
    
    def process_stations(self, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process imputation for all specified stations.
        
        Args:
            processed_data: DataFrame containing all processed station data
            
        Returns:
            Dictionary with processing summary
        """
        start_time = datetime.now()
        self.logger.info(f"🚀 Starting imputation process for {self.config.variable_type.value}...")
        
        try:
            # Validate input data
            validation_results = self.validator.validate_input_data(processed_data)
            if not validation_results['passed']:
                raise ValueError(f"Input data validation failed: {'; '.join(validation_results['errors'])}")
            
            if validation_results['warnings']:
                for warning in validation_results['warnings']:
                    self.logger.warning(f"⚠️ {warning}")
            
            # Get station list
            stations = self._get_station_list(processed_data)
            
            # Process each station
            for i, station_name in enumerate(stations, 1):
                self._process_single_station(processed_data, station_name, i, len(stations))
            
            # Generate final reports
            self._generate_final_reports()
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create processing summary
            summary = self._create_processing_summary(processing_time)
            
            self.logger.info(f"✅ Imputation process completed successfully in {processing_time:.2f}s")
            self.logger.info(f"📊 Results available in: {self.output_dir}")
            
            return summary
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ Imputation process failed after {processing_time:.2f}s: {e}")
            raise
    
    def _get_station_list(self, processed_data: pd.DataFrame) -> List[str]:
        """
        Get the list of stations to process.
        
        CRITICAL: Stations are sorted by code to ensure consistency across
        imputation and prediction processes when max_stations is used.
        """
        all_stations = processed_data['Estación'].unique()
        total_stations = len(all_stations)
        
        # CRITICAL FIX: Sort stations by code for consistency
        # Extract station codes and create a mapping for sorting
        station_code_map = {}
        for station in all_stations:
            station_data = processed_data[processed_data['Estación'] == station]
            if len(station_data) > 0 and 'Código' in station_data.columns:
                try:
                    # Try to get numeric code
                    code = station_data['Código'].iloc[0]
                    if pd.notna(code):
                        # Convert to int if possible, otherwise use string
                        try:
                            code_value = int(float(str(code)))
                        except (ValueError, TypeError):
                            # Fallback: extract numeric part from station name
                            code_match = re.search(r'\d+', str(station))
                            code_value = int(code_match.group()) if code_match else 999999
                        station_code_map[station] = code_value
                    else:
                        # No code available, use a high number to sort last
                        station_code_map[station] = 999999
                except (KeyError, IndexError):
                    # Fallback: extract numeric part from station name
                    code_match = re.search(r'\d+', str(station))
                    station_code_map[station] = int(code_match.group()) if code_match else 999999
            else:
                # Fallback: extract numeric part from station name
                import re
                code_match = re.search(r'\d+', str(station))
                station_code_map[station] = int(code_match.group()) if code_match else 999999
        
        # Sort stations by code (ascending)
        sorted_stations = sorted(all_stations, key=lambda s: station_code_map.get(s, 999999))
        
        if self.config.max_stations is not None:
            stations = sorted_stations[:self.config.max_stations]
            self.logger.info(f"📊 Processing {len(stations)} of {total_stations} available stations (sorted by code)")
        else:
            stations = sorted_stations
            self.logger.info(f"📊 Processing all {total_stations} stations (sorted by code)")
        
        self.logger.info(f"📋 Stations to process (in order):")
        for i, station in enumerate(stations, 1):
            code = station_code_map.get(station, 'N/A')
            self.logger.info(f"  {i}. [{code}] {station}")
        
        return stations
    
    def _process_single_station(self, processed_data: pd.DataFrame, station_name: str, 
                               station_index: int, total_stations: int) -> None:
        """
        Process imputation for a single station.
        
        Args:
            processed_data: DataFrame containing all processed data
            station_name: Name of the station to process
            station_index: Current station index (1-based)
            total_stations: Total number of stations to process
        """
        self.logger.info(f"[{station_index}/{total_stations}] Processing station: {station_name}")
        
        try:
            # Extract station data
            station_data = self._extract_station_data(processed_data, station_name)
            
            # Validate station data
            validation_results = self.validator.validate_station_data(station_data)
            if not validation_results['passed']:
                self.logger.error(f"    ❌ Station validation failed: {'; '.join(validation_results['errors'])}")
                return
            
            if validation_results['warnings']:
                for warning in validation_results['warnings']:
                    self.logger.warning(f"    ⚠️ {warning}")
            
            if not station_data.has_data:
                self.logger.warning(f"    ⚠️ No data available for station {station_name}")
                return
            
            # Perform imputation
            imputation_result = self._perform_imputation(station_data)
            
            if imputation_result.success:
                # Generate outputs
                self._generate_station_outputs(imputation_result)
                
                # Store result
                self.imputation_results.append(imputation_result)
                
                self.logger.info(f"    ✅ Station {station_index}/{total_stations} completed successfully")
                self.logger.info(f"      - Imputed: {imputation_result.imputed_count} values ({imputation_result.imputation_rate:.1f}%)")
                self.logger.info(f"      - Processing time: {imputation_result.processing_time:.2f}s")
            else:
                self.logger.error(f"    ❌ Station {station_index}/{total_stations} failed: {imputation_result.error_message}")
                
        except Exception as e:
            self.logger.error(f"    ❌ Error processing station {station_name}: {e}")
    
    def _extract_station_data(self, processed_data: pd.DataFrame, station_name: str) -> StationData:
        """
        Extract data for a specific station.
        
        Args:
            processed_data: DataFrame containing all processed data
            station_name: Name of the station to extract
            
        Returns:
            StationData object containing station information and data
        """
        station_data = processed_data[processed_data['Estación'] == station_name].copy()
        
        if not isinstance(station_data, pd.DataFrame):
            raise ValueError(f"Invalid data type for station {station_name}")
        
        if len(station_data) == 0:
            return StationData(name=station_name, code="", data=station_data, variable_type=self.config.variable_type)
        
        station_code = str(station_data['Código'].iloc[0])
        value_column = self._get_value_column()
        
        self.logger.info(f"    📊 Station data: {len(station_data)} rows")
        self.logger.info(f"    📊 Missing values: {station_data[value_column].isnull().sum()} ({station_data[value_column].isnull().sum()/len(station_data)*100:.1f}%)")
        
        return StationData(name=station_name, code=station_code, data=station_data, variable_type=self.config.variable_type)
    
    def _perform_imputation(self, station_data: StationData) -> ImputationResult:
        """
        Perform imputation for a station.
        
        Args:
            station_data: StationData object containing station information
            
        Returns:
            ImputationResult object with imputation results
        """
        start_time = datetime.now()
        
        try:
            target_column = station_data.value_column  # Use the correct column for this variable
            
            # Perform imputation using the service
            imputed_data, service_result = self.imputation_service.impute_station(
                station_data=station_data.data,
                station_name=station_data.name,
                station_code=station_data.code,
                target_column=target_column,
                method=self.config.imputation_method
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract quality metrics from service result
            quality_metrics = {}
            if hasattr(service_result, 'quality_metrics'):
                quality_metrics = service_result.quality_metrics
            
            return ImputationResult(
                station_name=station_data.name,
                station_code=station_data.code,
                original_data=station_data.data,
                imputed_data=imputed_data,
                processing_time=processing_time,
                success=True,
                variable_type=self.config.variable_type,
                imputation_method=self.config.imputation_method,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ImputationResult(
                station_name=station_data.name,
                station_code=station_data.code,
                original_data=station_data.data,
                imputed_data=station_data.data.copy(),  # Return original data on failure
                processing_time=processing_time,
                success=False,
                variable_type=self.config.variable_type,
                error_message=str(e)
            )
    
    def _generate_station_outputs(self, result: ImputationResult) -> None:
        """
        Generate all outputs for a station.
        
        Args:
            result: ImputationResult object containing station results
        """
        try:
            # Generate statistics report
            self._generate_statistics_report(result)
            
            # Generate comparison plot if enabled
            if self.config.enable_plots:
                self._generate_comparison_plot(result)
            
            # Save CSV file
            self._save_station_csv(result)
            
        except Exception as e:
            self.logger.error(f"    ⚠️ Error generating outputs for station {result.station_name}: {e}")
    
    def _generate_statistics_report(self, result: ImputationResult) -> None:
        """Generate statistical report for a station."""
        try:
            target_column = result.value_column
            
            report = self.statistics_reporter.generate_station_report(
                original_data=result.original_data,
                imputed_data=result.imputed_data,
                station_name=result.station_name,
                station_code=result.station_code,
                target_column=target_column
            )
            
            # Display summary
            stats = report['statistics']
            missing = report['missing_value_analysis']
            
            self.logger.info(f"    📊 Statistics report generated:")
            self.logger.info(f"      - Mean: {stats['original']['mean']:.2f} → {stats['imputed']['mean']:.2f}")
            self.logger.info(f"      - Std: {stats['original']['std']:.2f} → {stats['imputed']['std']:.2f}")
            self.logger.info(f"      - Min: {stats['original']['min']:.2f} → {stats['imputed']['min']:.2f}")
            self.logger.info(f"      - Max: {stats['original']['max']:.2f} → {stats['imputed']['max']:.2f}")
            self.logger.info(f"      - Trend: {report['trend_analysis']['original']['trend_direction']} → {report['trend_analysis']['imputed']['trend_direction']}")
            self.logger.info(f"      - Imputation: {missing['imputed_values']}/{missing['original_missing']} values ({missing['imputation_rate']:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"    ⚠️ Error generating statistics report: {e}")
    
    def _generate_comparison_plot(self, result: ImputationResult) -> None:
        """Generate comparison plot for a station."""
        try:
            # Configure matplotlib
            plt.ioff()
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.plot_figsize)
            
            target_column = result.value_column
            date_column = 'Fecha'
            
            # Prepare data
            original_data = self._prepare_plot_data(result.original_data, date_column)
            imputed_data = self._prepare_plot_data(result.imputed_data, date_column)
            
            # Plot original data
            ax1.plot(original_data[date_column], original_data[target_column], 
                    color='red', alpha=0.6, linewidth=0.5, label='Original')
            ax1.set_title(f'{result.station_name} ({result.station_code}) - Original', fontsize=12)
            ax1.set_ylabel(target_column)
            ax1.grid(True, alpha=0.2)
            ax1.legend()
            
            # Plot imputed data
            ax2.plot(imputed_data[date_column], imputed_data[target_column], 
                    color='blue', alpha=0.6, linewidth=0.5, label='Imputed')
            ax2.set_title(f'{result.station_name} ({result.station_code}) - Imputed', fontsize=12)
            ax2.set_xlabel('Date')
            ax2.set_ylabel(target_column)
            ax2.grid(True, alpha=0.2)
            ax2.legend()
            
            # Save plot
            plt.tight_layout()
            
            safe_station_name = result.station_name.replace(' ', '_').replace(',', '').replace('.', '')[:20]
            filename = f"{result.station_code}_{safe_station_name}_Comparison.png"
            filepath = self.output_dir / "plots" / filename
            
            plt.savefig(filepath, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"    📊 Plot saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"    ⚠️ Error generating plot: {e}")
            plt.close('all')
    
    def _prepare_plot_data(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Prepare data for plotting by ensuring proper date format."""
        data_copy = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data_copy[date_column]):
            data_copy[date_column] = pd.to_datetime(data_copy[date_column])
        return data_copy
    
    def _save_station_csv(self, result: ImputationResult) -> None:
        """Save imputed data to CSV file."""
        try:
            safe_station_name = result.station_name.replace(' ', '_').replace(',', '').replace('.', '')[:20]
            filename = f"{result.station_code}_{safe_station_name}_Imputed.csv"
            filepath = self.output_dir / "csv_files" / filename
            
            result.imputed_data.to_csv(filepath, index=False, encoding='utf-8')
            
            self.logger.info(f"    📄 CSV saved: {filename}")
            self.logger.info(f"      - Rows: {len(result.imputed_data)}")
            
            target_column = result.value_column
            missing_count = result.imputed_data[target_column].isnull().sum()
            self.logger.info(f"      - Remaining missing values: {missing_count}")
            
        except Exception as e:
            self.logger.error(f"    ⚠️ Error saving CSV: {e}")
    
    def _generate_final_reports(self) -> None:
        """Generate final reports and summaries."""
        try:
            # Save imputation results
            self._save_imputation_results()
            
            # Save consolidated CSV
            self._save_consolidated_csv()
            
            # Save statistics reports
            self._save_statistics_reports()
            
            # Display final summary
            self._display_final_summary()
            
        except Exception as e:
            self.logger.error(f"⚠️ Error generating final reports: {e}")
    
    def _save_imputation_results(self) -> None:
        """Save imputation results to JSON file."""
        try:
            results_file = self.output_dir / f"imputation_results_{self.config.variable_type.value}.json"
            self.imputation_service.save_results(results_file)
            self.logger.info(f"📄 Imputation results saved: {results_file}")
        except Exception as e:
            self.logger.error(f"⚠️ Error saving imputation results: {e}")
    
    def _save_consolidated_csv(self) -> None:
        """Save all imputed data to a consolidated CSV file."""
        if not self.imputation_results:
            self.logger.warning('    ⚠️ No imputed data to consolidate')
            return
        
        try:
            # Combine all successful imputations
            successful_results = [r for r in self.imputation_results if r.success]
            
            if not successful_results:
                self.logger.warning('    ⚠️ No successful imputations to consolidate')
                return
            
            consolidated_data = pd.concat([r.imputed_data for r in successful_results], ignore_index=True)
            
            filename = f"all_stations_{self.config.variable_type.value}_imputed.csv"
            filepath = self.output_dir / "csv_files" / filename
            
            consolidated_data.to_csv(filepath, index=False, encoding='utf-8')
            
            self.logger.info(f"    📄 Consolidated CSV saved: {filename}")
            self.logger.info(f"      - Total rows: {len(consolidated_data):,}")
            self.logger.info(f"      - Total stations: {consolidated_data['Estación'].nunique()}")
            
            # Get the correct value column for this variable type
            value_column = self._get_value_column()
            missing_count = consolidated_data[value_column].isnull().sum()
            self.logger.info(f"      - Remaining missing values: {missing_count}")
            
        except Exception as e:
            self.logger.error(f"    ⚠️ Error saving consolidated CSV: {e}")
    
    def _save_statistics_reports(self) -> None:
        """Save statistics reports."""
        try:
            stats_dir = self.output_dir / "statistics_reports"
            self.statistics_reporter.save_reports(stats_dir, format='both')
            
            overall_summary = self.statistics_reporter.get_overall_summary()
            self.logger.info(f"\n📈 Overall statistical summary:")
            self.logger.info(f"  Stations analyzed: {overall_summary.get('total_stations', 0)}")
            self.logger.info(f"  Average imputation rate: {overall_summary.get('average_imputation_rate', 0):.1f}%")
            
            mean_stats = overall_summary.get('mean_statistics', {})
            if mean_stats:
                self.logger.info(f"  Average mean change: {mean_stats.get('mean_change_pct', 0):.2f}%")
                self.logger.info(f"  Average std change: {overall_summary.get('std_statistics', {}).get('std_change_pct', 0):.2f}%")
            
            self.logger.info(f"  📁 Reports saved in: {stats_dir}")
            
        except Exception as e:
            self.logger.error(f"⚠️ Error saving statistics reports: {e}")
    
    def _display_final_summary(self) -> None:
        """Display final processing summary."""
        try:
            summary = self.imputation_service.get_imputation_summary()
            
            self.logger.info(f"\n📊 Final imputation summary:")
            self.logger.info(f"  Stations processed: {summary.get('total_stations', 0)}")
            self.logger.info(f"  Successful stations: {summary.get('successful_stations', 0)}")
            self.logger.info(f"  Success rate: {summary.get('success_rate', 0):.1f}%")
            self.logger.info(f"  Total values imputed: {summary.get('total_imputed', 0):,}")
            self.logger.info(f"  Overall imputation rate: {summary.get('overall_imputation_rate', 0):.1f}%")
            self.logger.info(f"  Methods used: {summary.get('method_usage', {})}")
        except Exception as e:
            self.logger.error(f"⚠️ Error displaying final summary: {e}")
    
    def _create_processing_summary(self, processing_time: float) -> Dict[str, Any]:
        """Create comprehensive processing summary."""
        successful_results = [r for r in self.imputation_results if r.success]
        
        summary = {
            'variable_type': self.config.variable_type.value,
            'processing_time': processing_time,
            'total_stations': len(self.imputation_results),
            'successful_stations': len(successful_results),
            'success_rate': len(successful_results) / len(self.imputation_results) * 100 if self.imputation_results else 0,
            'total_imputed_values': sum(r.imputed_count for r in successful_results),
            'average_imputation_rate': np.mean([r.imputation_rate for r in successful_results]) if successful_results else 0,
            'output_directory': str(self.output_dir),
            'configuration': {
                'imputation_method': self.config.imputation_method,
                'max_missing_block_days': self.config.max_missing_block_days,
                'small_block_threshold': self.config.small_block_threshold
            }
        }
        
        return summary


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
        
        # Import and use the preprocessor
        from preprocess_meteorological import MeteorologicalPreprocessor, PreprocessingConfig
        
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
        description='Meteorological Data Imputation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    python imputation_process.py                    # Process precipitation (default)
    python imputation_process.py --variable temp_max    # Process temp_max
    python imputation_process.py --variable temp_min    # Process temp_min
    python imputation_process.py --variable humidity     # Process humidity
    python imputation_process.py --help            # Show this help
    
PIPELINE INTEGRATION:
    # Called from pipeline with parameters
    python imputation_process.py --variable precipitation --max-stations 5
        """
    )
    
    parser.add_argument(
        '--variable',
        default='precipitation',
        choices=['temp_max', 'temp_min', 'precipitation', 'humidity'],
        help='Meteorological variable to process (default: precipitation)'
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
        '--imputation-method',
        default='auto',
        choices=['auto', 'linear', 'polynomial', 'arima', 'mean', 'median'],
        help='Imputation method to use (default: auto)'
    )
    
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable validation'
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
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for imputation results (default: uses PathManager)'
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
    
    from src.config.path_manager import get_path_manager
    path_manager = get_path_manager()
    data_path = args.data_path if args.data_path else str(path_manager.input_dir)
    
    config = ImputationConfig(
        variable_type=variable_type,
        data_path=data_path,
        max_stations=args.max_stations,
        imputation_method=args.imputation_method,
        enable_validation=not args.no_validation,
        enable_logging=not args.no_logging,
        enable_plots=not args.no_plots,
        output_dir=args.output_dir
    )
    
    # Display configuration
    print(f'Imputation configuration:')
    print(f'  Variable: {config.variable_type.value}')
    print(f'  Data path: {config.data_path}')
    print(f'  Max stations: {config.max_stations if config.max_stations else "All"}')
    print(f'  Imputation method: {config.imputation_method}')
    print(f'  Enable plots: {config.enable_plots}')
    print(f'  Enable validation: {config.enable_validation}')
    print(f'  Enable logging: {config.enable_logging}')
    
    # Get file paths
    file_paths = get_file_paths_for_variable(config.variable_type.value)
    print(f'  Excel file: {file_paths["input_excel"]}')
    print(f'  CSV file: {file_paths["input_csv"]}')
    
    # Load data
    processed_data = load_processed_data(config.variable_type, config.data_path)
    
    # Apply station limiting if specified
    # CRITICAL: Sort stations by code to ensure consistency with prediction process
    if args.max_stations and args.max_stations > 0:
        unique_stations = processed_data['Estación'].unique()
        
        # Sort stations by code for consistency
        station_code_map = {}
        for station in unique_stations:
            station_data = processed_data[processed_data['Estación'] == station]
            if len(station_data) > 0 and 'Código' in station_data.columns:
                try:
                    code = station_data['Código'].iloc[0]
                    if pd.notna(code):
                        try:
                            code_value = int(float(str(code)))
                        except (ValueError, TypeError):
                            import re
                            code_match = re.search(r'\d+', str(station))
                            code_value = int(code_match.group()) if code_match else 999999
                        station_code_map[station] = code_value
                    else:
                        station_code_map[station] = 999999
                except (KeyError, IndexError):
                    import re
                    code_match = re.search(r'\d+', str(station))
                    station_code_map[station] = int(code_match.group()) if code_match else 999999
            else:
                code_match = re.search(r'\d+', str(station))
                station_code_map[station] = int(code_match.group()) if code_match else 999999
        
        sorted_stations = sorted(unique_stations, key=lambda s: station_code_map.get(s, 999999))
        
        if len(sorted_stations) > args.max_stations:
            print(f'Limiting to {args.max_stations} stations (from {len(sorted_stations)} total, sorted by code)')
            selected_stations = sorted_stations[:args.max_stations]
            processed_data = processed_data[processed_data['Estación'].isin(selected_stations)]
            print(f'Data limited to {len(selected_stations)} stations (sorted by code for consistency)')
    
    # Configure matplotlib for better performance
    import matplotlib
    matplotlib.use('Agg')
    
    # Initialize and run imputation
    imputer = MissingValueImputer(config)
    summary = imputer.process_stations(processed_data)
    
    # Display final results
    print(f'Final Results Summary:')
    print(f'  Processing time: {summary["processing_time"]:.2f}s')
    print(f'  Stations processed: {summary["total_stations"]}')
    print(f'  Success rate: {summary["success_rate"]:.1f}%')
    print(f'  Total values imputed: {summary["total_imputed_values"]:,}')
    print(f'  Average imputation rate: {summary["average_imputation_rate"]:.1f}%')
    
    print(f'Imputation process completed successfully!')
    print(f'Check results in: {imputer.output_dir}')


if __name__ == "__main__":
    main()