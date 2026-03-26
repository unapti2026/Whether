"""
Meteorological Processor

This module defines the MeteorologicalDataProcessor class for processing
meteorological data in the weather prediction system.
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging
import re
from ...core.interfaces.data_processor import DataProcessorInterface
from ...core.exceptions import DataProcessingError, ValidationError
from ...config.frequency_config import FrequencyDetector, get_frequency_config, detect_and_validate_frequency, FrequencyConfig
from ...config.structure_detector import StructureDetector, detect_data_structure, validate_data_structure, DataStructureConfig
import os


class MeteorologicalDataProcessor(DataProcessorInterface):
    """
    Processor for cleaning and restructuring meteorological data.
    Implements DataProcessorInterface for consistency.
    """
    def __init__(self, input_path: str, config: Dict[str, Any]):
        self.input_path = input_path
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data: Dict[str, Optional[pd.DataFrame]] = {'raw': None, 'cleaned': None, 'processed': None}
        self.info: Dict[str, Any] = {}

    def process(self, data: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        """
        Run the full processing pipeline: clean and restructure data.
        Returns the processed DataFrame.
        """
        try:
            self.logger.info(f"Loading data from: {self.input_path}")
            self.data['raw'] = pd.read_csv(self.input_path)
            self.logger.info(f"Loaded raw data with shape: {self.data['raw'].shape}")
            self.clean_data()
            self.process_data()
            self.info['input'] = self.input_path
            self.info['output_shape'] = self.data['processed'].shape if self.data['processed'] is not None else None
            if self.data['processed'] is None:
                raise DataProcessingError("Processing failed - no output data generated")
            return self.data['processed']
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            raise DataProcessingError(f"Error processing data: {e}") from e

    def clean_data(self) -> pd.DataFrame:
        """Clean the raw data and store in self.data['cleaned']."""
        # Load data if not already loaded (for compatibility with TimeSeriesEngineerService)
        if self.data['raw'] is None:
            self.logger.info(f"Loading data from: {self.input_path}")
            self.data['raw'] = pd.read_csv(self.input_path)
            self.logger.info(f"Loaded raw data with shape: {self.data['raw'].shape}")
        
        df = self.data['raw'].copy()
        for val in self.config.get('clean_values', []):
            df.replace(val, pd.NA, inplace=True)
        self.data['cleaned'] = df
        self.logger.info(f"Data cleaning completed. Shape: {df.shape}")
        return df

    def process_data(self) -> pd.DataFrame:
        """Restructure cleaned data to time series format and store in self.data['processed']."""
        if self.data['cleaned'] is None:
            raise DataProcessingError("No cleaned data available.")
        df = self.data['cleaned']
        
        # Detect structure configuration
        structure_config = self._get_structure_config(df)
        
        # Update config with detected structure
        self.config['day_prefix'] = structure_config.day_prefix
        self.config['max_days'] = structure_config.max_days
        
        # Store structure information
        self.info['detected_structure'] = structure_config.to_dict()
        
        restructured = []
        for _, row in df.iterrows():
            year = row[self.config['year_column']]
            month = row[self.config['month_column']]
            station = row[self.config['station_column']]
            code = row[self.config['code_column']]
            
            # Convert year and month to integers for proper formatting
            try:
                year_int = int(year) if pd.notna(year) else None
                month_int = int(month) if pd.notna(month) else None
            except (ValueError, TypeError):
                self.logger.warning(f"Skipping row with invalid year/month: year={year}, month={month}")
                continue
            
            if year_int is None or month_int is None:
                self.logger.warning(f"Skipping row with missing year/month: year={year}, month={month}")
                continue
            
            # Use detected day columns if available, otherwise fallback to pattern
            if structure_config.has_day_columns:
                day_columns = structure_config.day_columns
            else:
                # Fallback to pattern-based detection
                day_columns = [f"{self.config['day_prefix']}{day:02d}" for day in range(1, self.config['max_days'] + 1)]
            
            for day_col in day_columns:
                if day_col in row and pd.notna(row[day_col]):
                    # Extract day number from column name
                    day_match = re.search(r'\d+', day_col)
                    if day_match:
                        day = int(day_match.group())
                        date_str = f"{year_int}-{month_int:02d}-{day:02d}"
                        restructured.append([
                            int(code),
                            station,
                            date_str,
                            row[day_col]  # Use the day column value directly
                        ])
        
        column_names = [self.config['code_column'], self.config['station_column'], self.config['date_column'], self.config['value_column']]
        processed_df = pd.DataFrame(restructured, columns=column_names)

        # Completar fechas faltantes para TODAS las estaciones
        if len(processed_df) > 0:
            self.logger.info("Starting date completion for all stations...")
            all_completed = []
            
            # Determine frequency configuration
            frequency_config = self._get_frequency_config(processed_df)
            
            for station, group in processed_df.groupby(self.config['station_column']):
                group = group.copy()
                
                # Convertir fechas a datetime
                group[self.config['date_column']] = pd.to_datetime(group[self.config['date_column']])
                
                # Obtener rango de fechas para esta estación
                min_date = group[self.config['date_column']].min()
                max_date = group[self.config['date_column']].max()
                
                # Crear rango completo de fechas usando la frecuencia detectada
                all_dates = pd.date_range(start=min_date, end=max_date, freq=frequency_config.pandas_freq)
                
                # Crear DataFrame con todas las fechas
                full_index = pd.DataFrame({self.config['date_column']: all_dates})
                
                # Obtener valores únicos de código y estación
                code = group[self.config['code_column']].iloc[0]
                station_name = group[self.config['station_column']].iloc[0]
                
                # Agregar código y estación al DataFrame completo
                full_index[self.config['code_column']] = code
                full_index[self.config['station_column']] = station_name
                
                # Merge para completar fechas faltantes
                merged = full_index.merge(
                    group[[self.config['date_column'], self.config['value_column']]],
                    on=self.config['date_column'],
                    how='left'
                )
                
                all_completed.append(merged)
                
                # Log del proceso
                original_count = len(group)
                completed_count = len(merged)
                nan_count = merged[self.config['value_column']].isna().sum()
                
                self.logger.info(f"Station: {station_name}")
                self.logger.info(f"  Original dates: {original_count}")
                self.logger.info(f"  Completed dates: {completed_count}")
                self.logger.info(f"  Added NaN values: {nan_count}")
                self.logger.info(f"  Date range: {min_date.date()} to {max_date.date()}")
                self.logger.info(f"  Frequency: {frequency_config.description} ({frequency_config.pandas_freq})")
            
            # Combine all results
            processed_df = pd.concat(all_completed, ignore_index=True)
            
            # Reordenar columnas
            processed_df = processed_df[[self.config['code_column'], self.config['station_column'], self.config['date_column'], self.config['value_column']]]
            
            # Convertir fechas de vuelta a string para consistencia
            processed_df[self.config['date_column']] = processed_df[self.config['date_column']].astype(str)
            
            # Store frequency information
            self.info['detected_frequency'] = frequency_config.frequency.value
            self.info['frequency_description'] = frequency_config.description
            self.info['pandas_freq'] = frequency_config.pandas_freq
            
            self.data['processed'] = processed_df
            self.logger.info(f"Data restructuring completed with date completion. Final shape: {processed_df.shape}")
            return processed_df
        else:
            self.data['processed'] = processed_df
            self.logger.info(f"Data restructuring completed (no data to process). Shape: {processed_df.shape}")
            return processed_df

    def validate_input(self, data: Optional[pd.DataFrame] = None) -> bool:
        """Validate the input data or file."""
        try:
            if data is not None:
                df = data
            else:
                df = pd.read_csv(self.input_path, nrows=10)
            required_cols = [self.config['year_column'], self.config['month_column'], self.config['station_column'], self.config['code_column']]
            for col in required_cols:
                if col not in df.columns:
                    raise ValidationError(f"Missing required column: {col}")
            return True
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise ValidationError(f"Validation failed: {e}") from e

    def get_processing_info(self) -> Dict[str, Any]:
        """Return information about the processing steps."""
        return self.info

    def reset(self) -> None:
        """Reset the processor state."""
        self.data = {'raw': None, 'cleaned': None, 'processed': None}
        self.info = {}

    def _get_frequency_config(self, data: pd.DataFrame) -> 'FrequencyConfig':
        """
        Get frequency configuration for the data.
        
        Args:
            data: DataFrame with date column
            
        Returns:
            FrequencyConfig object
        """
        # Check if auto-detection is enabled
        if self.config.get('auto_detect_frequency', True):
            # Detect frequency from data
            dates = pd.to_datetime(data[self.config['date_column']])
            try:
                frequency_config = detect_and_validate_frequency(
                    dates, 
                    expected_frequency=self.config.get('data_frequency')
                )
                self.logger.info(f"Auto-detected frequency: {frequency_config.description}")
                return frequency_config
            except Exception as e:
                self.logger.warning(f"Frequency auto-detection failed: {e}. Using default frequency.")
        
        # Use configured frequency or default
        frequency = self.config.get('data_frequency', 'D')
        frequency_config = get_frequency_config(frequency)
        self.logger.info(f"Using configured frequency: {frequency_config.description}")
        return frequency_config

    def _get_structure_config(self, data: pd.DataFrame) -> 'DataStructureConfig':
        """
        Get structure configuration for the data.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            DataStructureConfig object
        """
        # Check if auto-detection is enabled
        if self.config.get('auto_detect_structure', True):
            try:
                structure_config = detect_data_structure(data, sample_size=1000)
                
                # Validate structure if validation is enabled
                if self.config.get('structure_validation', True):
                    if not validate_data_structure(data, structure_config):
                        self.logger.warning("Structure validation failed. Using default structure.")
                        return self._get_default_structure_config()
                
                # Check confidence threshold
                confidence_threshold = self.config.get('structure_confidence_threshold', 0.7)
                if structure_config.confidence >= confidence_threshold:
                    self.logger.info(f"Auto-detected structure: {structure_config.structure_type} (confidence: {structure_config.confidence:.2f})")
                    return structure_config
                else:
                    self.logger.warning(f"Structure detection confidence too low: {structure_config.confidence:.2f}. Using default structure.")
                    return self._get_default_structure_config()
                    
            except Exception as e:
                self.logger.warning(f"Structure auto-detection failed: {e}. Using default structure.")
        
        return self._get_default_structure_config()

    def _get_default_structure_config(self) -> 'DataStructureConfig':
        """
        Get default structure configuration.
        
        Returns:
            Default DataStructureConfig
        """
        return DataStructureConfig(
            day_prefix=self.config.get('day_prefix', 'Día'),
            max_days=self.config.get('max_days', 31),
            day_columns=[],
            has_day_columns=False,
            structure_type='default',
            confidence=1.0,
            detected_patterns={}
        )

    def save_data(self, output_path: str) -> None:
        """
        Save the processed data to a file.
        
        Args:
            output_path: Path where to save the processed data
        """
        if self.data['processed'] is None:
            raise DataProcessingError("No processed data available to save")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        self.data['processed'].to_csv(output_path, index=False)
        self.logger.info(f"Processed data saved to: {output_path}")
        self.info['output_file'] = output_path 