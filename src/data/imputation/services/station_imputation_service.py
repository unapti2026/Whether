"""
Station Imputation Service

This module provides a service for imputing missing values in meteorological
data for individual stations. It centralizes the imputation logic and provides
a clean interface for station-specific imputation operations.
"""

from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

from ...models import ImputationResult
from ....core.exceptions import ImputationError, ValidationError
from ....core.validators import DataValidator
from ....config.frequency_config import get_frequency_config, FrequencyConfig


class ImputationMethod(Enum):
    """Enumeration for available imputation methods."""
    AUTO = "auto"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    ARIMA = "arima"
    MEAN = "mean"
    MEDIAN = "median"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"


class BlockSize(Enum):
    """Enumeration for missing block sizes."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@dataclass
class ImputationConfig:
    """Configuration class for imputation parameters."""
    max_missing_block_days: int = 548  # Maximum block size for missing data imputation
    small_block_threshold: int = 3
    medium_block_threshold: int = 30
    seasonal_period: int = 365
    min_data_points_for_arima: int = 100
    min_context_for_arima: int = 50
    arima_order: Tuple[int, int, int] = (1, 0, 1)
    polynomial_order: int = 2


@dataclass
class BlockInfo:
    """Data class representing a missing data block."""
    start: int
    end: int
    length: int
    size_category: BlockSize
    
    @classmethod
    def create(cls, start: int, end: int, small_threshold: int, medium_threshold: int) -> 'BlockInfo':
        """Create a BlockInfo instance with automatic size categorization."""
        length = end - start + 1
        if length <= small_threshold:
            size_category = BlockSize.SMALL
        elif length <= medium_threshold:
            size_category = BlockSize.MEDIUM
        else:
            size_category = BlockSize.LARGE
        return cls(start=start, end=end, length=length, size_category=size_category)


@dataclass
class ImputationStatistics:
    """Data class for imputation statistics."""
    original_missing_count: int
    cleaned_missing_count: int
    imputed_count: int
    total_count: int
    rows_removed: int
    strategy_used: str
    processing_time_seconds: float
    block_analysis: Dict[str, Any]
    cleaning_info: Dict[str, Any]


class StationImputationService:
    """
    Service for imputing missing values in meteorological data for individual stations.
    
    This service provides a centralized interface for applying imputation strategies
    to station-specific data, with support for different imputation methods and
    quality validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the station imputation service.
        
        Args:
            config: Configuration dictionary with service settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_validator = DataValidator()
        self.imputation_config = self._create_imputation_config()
        
        # Results storage
        self.imputation_results: Dict[str, ImputationResult] = {}
        
    def _create_imputation_config(self) -> ImputationConfig:
        """Create imputation configuration from service config."""
        arima_order_config = self.config.get('arima_order', (1, 0, 1))
        if isinstance(arima_order_config, (list, tuple)) and len(arima_order_config) == 3:
            arima_order = tuple(arima_order_config)
        else:
            arima_order = (1, 0, 1)
        
        # Get frequency configuration
        frequency = self.config.get('data_frequency', 'D')
        frequency_config = get_frequency_config(frequency)
        
        return ImputationConfig(
            max_missing_block_days=self.config.get('max_missing_block_days', frequency_config.max_gap_days),
            small_block_threshold=self.config.get('small_block_threshold', 3),
            medium_block_threshold=self.config.get('medium_block_threshold', 30),
            seasonal_period=self.config.get('seasonal_period', frequency_config.seasonal_period),
            min_data_points_for_arima=self.config.get('min_data_points_for_arima', frequency_config.min_data_points),
            min_context_for_arima=self.config.get('min_context_for_arima', 50),
            arima_order=arima_order,
            polynomial_order=self.config.get('polynomial_order', 2)
        )
        
    def impute_station(self, 
                      station_data: pd.DataFrame, 
                      station_name: str,
                      station_code: str,
                      target_column: str,
                      method: str = ImputationMethod.AUTO.value,
                      **kwargs) -> Tuple[pd.DataFrame, ImputationResult]:
        """
        Impute missing values for a specific station.
        
        Args:
            station_data: DataFrame with station data
            station_name: Name of the station
            station_code: Code of the station
            target_column: Name of the column to impute
            method: Imputation method (see ImputationMethod enum)
            **kwargs: Additional arguments for the imputation method
            
        Returns:
            Tuple of (imputed_data, imputation_result)
            
        Raises:
            ImputationError: If imputation fails
            ValidationError: If input validation fails
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting imputation for station: {station_name} ({station_code}) using method: {method}")
            
            # Validate input data
            self._validate_station_data(station_data, target_column)
            
            # Get original statistics
            original_missing_count = station_data[target_column].isnull().sum()
            total_count = len(station_data)
            
            if original_missing_count == 0:
                return self._handle_no_missing_values(station_data, target_column, start_time)
            
            # Clean data by removing large missing blocks
            cleaned_data, cleaning_info = self._remove_large_missing_blocks(
                station_data, target_column, self.imputation_config.max_missing_block_days
            )
            
            # Update statistics after cleaning
            cleaned_missing_count = cleaned_data[target_column].isnull().sum()
            cleaned_total_count = len(cleaned_data)
            
            self.logger.info(f"Data cleaning completed. "
                           f"Removed {total_count - cleaned_total_count} rows due to large missing blocks. "
                           f"Remaining missing values: {cleaned_missing_count}")
            
            if cleaned_missing_count == 0:
                return self._handle_cleaning_only_result(
                    station_data, cleaned_data, target_column, original_missing_count, 
                    total_count, cleaned_total_count, cleaning_info, start_time
                )
            
            # Select and apply imputation method
            selected_method = self._select_imputation_method(method, cleaned_data, target_column)
            imputed_data = self._apply_imputation_method(cleaned_data, target_column, selected_method, **kwargs)
            
            # Calculate final statistics
            final_missing_count = imputed_data[target_column].isnull().sum()
            imputed_count = cleaned_missing_count - final_missing_count
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create and store result
            result = self._create_imputation_result(
                target_column, original_missing_count, imputed_count, selected_method,
                processing_time, station_data.shape, imputed_data.shape
            )
            
            # Add station-specific information
            self._add_station_result(result, station_name, station_code, original_missing_count,
                                   cleaned_missing_count, imputed_count, cleaned_total_count,
                                   total_count, selected_method, processing_time, cleaning_info)
            
            # Store result
            self.imputation_results[station_name] = result
            
            self.logger.info(f"Imputation completed for station {station_name}. "
                           f"Removed {total_count - cleaned_total_count} rows, "
                           f"imputed {imputed_count}/{cleaned_missing_count} values "
                           f"({(imputed_count/cleaned_missing_count*100):.1f}%) "
                           f"using {selected_method}")
            
            return imputed_data, result
            
        except Exception as e:
            return self._handle_imputation_error(e, station_name, station_code, target_column, start_time)
    
    def _handle_no_missing_values(self, station_data: pd.DataFrame, target_column: str, 
                                 start_time: datetime) -> Tuple[pd.DataFrame, ImputationResult]:
        """Handle case when no missing values are found."""
        self.logger.info("No missing values found in station data")
        result = ImputationResult.create_success(
            variable_type=self.config.get('variable_type', 'unknown'),
            column_name=target_column,
            original_missing_count=0,
            imputed_count=0,
            imputation_method='none',
            processing_time=0.0
        )
        return station_data.copy(), result
    
    def _handle_cleaning_only_result(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame,
                                   target_column: str, original_missing_count: int, total_count: int,
                                   cleaned_total_count: int, cleaning_info: Dict[str, Any],
                                   start_time: datetime) -> Tuple[pd.DataFrame, ImputationResult]:
        """Handle case when cleaning removes all missing values."""
        self.logger.info("No missing values remaining after cleaning")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = ImputationResult.create_success(
            variable_type=self.config.get('variable_type', 'unknown'),
            column_name=target_column,
            original_missing_count=original_missing_count,
            imputed_count=0,
            imputation_method='cleaning_only',
            processing_time=processing_time,
            original_data_shape=original_data.shape,
            imputed_data_shape=cleaned_data.shape
        )
        
        # Add cleaning information
        result.add_station_result('station', {
            'original_missing_count': original_missing_count,
            'cleaned_missing_count': 0,
            'imputed_count': 0,
            'total_count': cleaned_total_count,
            'rows_removed': total_count - cleaned_total_count,
            'cleaning_info': cleaning_info,
            'strategy_used': 'cleaning_only',
            'processing_time_seconds': processing_time
        })
        
        return cleaned_data, result
    
    def _handle_imputation_error(self, error: Exception, station_name: str, station_code: str,
                               target_column: str, start_time: datetime) -> Tuple[pd.DataFrame, ImputationResult]:
        """Handle imputation errors."""
        processing_time = (datetime.now() - start_time).total_seconds()
        self.logger.error(f"Imputation failed for station {station_name}: {error}")
        
        # Create failure result
        result = ImputationResult.create_failure(
            variable_type=self.config.get('variable_type', 'unknown'),
            column_name=target_column,
            error_message=str(error)
        )
        
        # Add station-specific information
        result.add_station_result(station_name, {
            'station_code': station_code,
            'error': str(error),
            'processing_time_seconds': processing_time
        })
        
        # Store result
        self.imputation_results[station_name] = result
        
        raise ImputationError(f"Imputation failed for station {station_name}: {error}") from error
    
    def _select_imputation_method(self, method: str, data: pd.DataFrame, target_column: str) -> str:
        """Select the appropriate imputation method."""
        if method == ImputationMethod.AUTO.value:
            return self._select_imputation_method_by_block_size(data, target_column)
        return method
    
    def _apply_imputation_method(self, data: pd.DataFrame, target_column: str, 
                               method: str, **kwargs) -> pd.DataFrame:
        """Apply the selected imputation method."""
        method_mapping = {
            ImputationMethod.LINEAR.value: self._linear_interpolation,
            ImputationMethod.POLYNOMIAL.value: self._polynomial_interpolation,
            ImputationMethod.ARIMA.value: self._arima_imputation,
            ImputationMethod.MEAN.value: self._mean_imputation,
            ImputationMethod.MEDIAN.value: self._median_imputation,
            ImputationMethod.FORWARD_FILL.value: self._forward_fill,
            ImputationMethod.BACKWARD_FILL.value: self._backward_fill
        }
        
        if method not in method_mapping:
            raise ImputationError(f"Unknown imputation method: {method}")
        
        return method_mapping[method](data, target_column, **kwargs)
    
    def _create_imputation_result(self, target_column: str, original_missing_count: int,
                                imputed_count: int, selected_method: str, processing_time: float,
                                original_shape: Tuple[int, int], imputed_shape: Tuple[int, int]) -> ImputationResult:
        """Create an ImputationResult instance."""
        return ImputationResult.create_success(
            variable_type=self.config.get('variable_type', 'unknown'),
            column_name=target_column,
            original_missing_count=original_missing_count,
            imputed_count=imputed_count,
            imputation_method=selected_method,
            processing_time=processing_time,
            original_data_shape=original_shape,
            imputed_data_shape=imputed_shape
        )
    
    def _add_station_result(self, result: ImputationResult, station_name: str, station_code: str,
                          original_missing_count: int, cleaned_missing_count: int, imputed_count: int,
                          cleaned_total_count: int, total_count: int, selected_method: str,
                          processing_time: float, cleaning_info: Dict[str, Any]) -> None:
        """Add station-specific information to the result."""
        result.add_station_result(station_name, {
            'station_code': station_code,
            'original_missing_count': original_missing_count,
            'cleaned_missing_count': cleaned_missing_count,
            'imputed_count': imputed_count,
            'total_count': cleaned_total_count,
            'rows_removed': total_count - cleaned_total_count,
            'missing_percentage': (cleaned_missing_count / cleaned_total_count) * 100,
            'imputation_percentage': (imputed_count / cleaned_missing_count) * 100 if cleaned_missing_count > 0 else 0,
            'strategy_used': selected_method,
            'processing_time_seconds': processing_time,
            'cleaning_info': cleaning_info,
            'block_analysis': self._analyze_missing_blocks(result, cleaned_total_count)
        })
    
    def _linear_interpolation(self, data: pd.DataFrame, target_column: str, **kwargs) -> pd.DataFrame:
        """Apply linear interpolation to the target column."""
        result_data = data.copy()
        result_data[target_column] = result_data[target_column].interpolate(
            method='linear', limit_direction='both'
        )
        return result_data
    
    def _polynomial_interpolation(self, data: pd.DataFrame, target_column: str, **kwargs) -> pd.DataFrame:
        """Apply polynomial interpolation to the target column."""
        result_data = data.copy()
        try:
            result_data[target_column] = result_data[target_column].interpolate(
                method='polynomial', order=self.imputation_config.polynomial_order, 
                limit_direction='both'
            )
        except Exception:
            # Fallback to linear if polynomial fails
            result_data[target_column] = result_data[target_column].interpolate(
                method='linear', limit_direction='both'
            )
        return result_data
    
    def _arima_imputation(self, data: pd.DataFrame, target_column: str, **kwargs) -> pd.DataFrame:
        """
        Apply sequential imputation by block size:
        1. Small blocks (≤3 days) with linear interpolation
        2. Medium blocks (4-30 days) with polynomial interpolation
        3. Large blocks (>30 days) with ARIMA
        """
        result_data = data.copy()
        series = result_data[target_column]
        
        # Check for sufficient data
        non_missing_mask = series.notna()
        if non_missing_mask.sum() < self.imputation_config.min_data_points_for_arima:
            self.logger.warning(f"Insufficient data for sequential imputation. "
                              f"Need at least {self.imputation_config.min_data_points_for_arima} non-missing values, "
                              f"but only have {non_missing_mask.sum()}. "
                              f"Falling back to polynomial interpolation.")
            return self._polynomial_interpolation(result_data, target_column, **kwargs)
        
        try:
            current_series = series.copy()
            total_imputed = 0
            
            # Step 1: Impute small blocks with linear interpolation
            total_imputed += self._impute_blocks_by_size(
                current_series, BlockSize.SMALL, self._linear_interpolation_step
            )
            
            # Step 2: Impute medium blocks with polynomial interpolation
            total_imputed += self._impute_blocks_by_size(
                current_series, BlockSize.MEDIUM, self._polynomial_interpolation_step
            )
            
            # Step 3: Impute large blocks with ARIMA
            total_imputed += self._impute_blocks_by_size(
                current_series, BlockSize.LARGE, self._arima_interpolation_step, result_data
            )
            
            # Step 4: Final fallback for any remaining NaNs
            total_imputed += self._apply_final_fallback(current_series)
            
            # Assign result
            result_data[target_column] = current_series
            
            self.logger.info(f"Sequential imputation completed: {total_imputed} values imputed in total")
            
            return result_data
            
        except Exception as e:
            self.logger.warning(f"Sequential imputation failed: {e}. Fallback to polynomial.")
            return self._polynomial_interpolation(result_data, target_column, **kwargs)
    
    def _impute_blocks_by_size(self, current_series: pd.Series, block_size: BlockSize, 
                              imputation_func, *args) -> int:
        """Impute blocks of a specific size using the provided function."""
        blocks = self._find_missing_blocks(current_series)
        filtered_blocks = self._filter_blocks_by_size(blocks, block_size)
        
        if not filtered_blocks:
            return 0
        
        self.logger.info(f"Imputing {len(filtered_blocks)} {block_size.value} blocks")
        
        initial_missing = current_series.isnull().sum()
        
        for i, block in enumerate(filtered_blocks):
            imputation_func(current_series, block, i, *args)
        
        final_missing = current_series.isnull().sum()
        imputed_count = initial_missing - final_missing
        
        self.logger.info(f"Completed {block_size.value} blocks: {imputed_count} values imputed")
        return imputed_count
    
    def _filter_blocks_by_size(self, blocks: List[Dict[str, int]], block_size: BlockSize) -> List[Dict[str, int]]:
        """Filter blocks by size category."""
        if block_size == BlockSize.SMALL:
            return [block for block in blocks if block['length'] <= self.imputation_config.small_block_threshold]
        elif block_size == BlockSize.MEDIUM:
            return [block for block in blocks 
                   if self.imputation_config.small_block_threshold < block['length'] <= self.imputation_config.medium_block_threshold]
        else:  # BlockSize.LARGE
            return [block for block in blocks if block['length'] > self.imputation_config.medium_block_threshold]
    
    def _linear_interpolation_step(self, current_series: pd.Series, block: Dict[str, int], 
                                 block_index: int) -> None:
        """Apply linear interpolation to a single block."""
        block_data = current_series.iloc[block['start']:block['end']+1].copy()
        block_data = block_data.interpolate(method='linear', limit_direction='both')
        current_series.iloc[block['start']:block['end']+1] = block_data
    
    def _polynomial_interpolation_step(self, current_series: pd.Series, block: Dict[str, int], 
                                     block_index: int) -> None:
        """Apply polynomial interpolation to a single block."""
        block_data = current_series.iloc[block['start']:block['end']+1].copy()
        try:
            block_data = block_data.interpolate(
                method='polynomial', order=self.imputation_config.polynomial_order, 
                limit_direction='both'
            )
            current_series.iloc[block['start']:block['end']+1] = block_data
        except Exception as e:
            # Fallback to linear if polynomial fails
            self.logger.warning(f"Polynomial interpolation failed for block {block_index+1}, using linear: {e}")
            block_data = block_data.interpolate(method='linear', limit_direction='both')
            current_series.iloc[block['start']:block['end']+1] = block_data
    
    def _arima_interpolation_step(self, current_series: pd.Series, block: Dict[str, int], 
                                block_index: int, result_data: pd.DataFrame) -> None:
        """Apply ARIMA interpolation to a single block."""
        non_missing_data = current_series.dropna()
        
        if len(non_missing_data) < self.imputation_config.seasonal_period * 2:
            self.logger.warning(f"Insufficient data for seasonal decomposition. "
                              f"Need at least {self.imputation_config.seasonal_period * 2} observations.")
            # Fallback to polynomial for large blocks
            self._polynomial_interpolation_step(current_series, block, block_index)
            return
        
        # Seasonal decomposition
        decomposition = seasonal_decompose(non_missing_data, model='additive', 
                                         period=self.imputation_config.seasonal_period)
        
        # Process trend, seasonality, and residuals
        trend_interp = self._process_trend_component(decomposition, current_series)
        seasonal_series = self._process_seasonal_component(decomposition, current_series, result_data)
        resid = current_series - trend_interp - seasonal_series
        
        # Impute residual with ARIMA
        self._impute_residual_with_arima(resid, block, block_index)
        
        # Reconstruct series
        current_series.iloc[block['start']:block['end']+1] = (
            trend_interp.iloc[block['start']:block['end']+1] + 
            seasonal_series.iloc[block['start']:block['end']+1] + 
            resid.iloc[block['start']:block['end']+1]
        )
    
    def _process_trend_component(self, decomposition, current_series: pd.Series) -> pd.Series:
        """Process trend component from decomposition."""
        trend = decomposition.trend.reindex(current_series.index)
        return trend.interpolate(method='linear', limit_direction='both')
    
    def _process_seasonal_component(self, decomposition, current_series: pd.Series, 
                                  result_data: pd.DataFrame) -> pd.Series:
        """Process seasonal component from decomposition."""
        seasonal = decomposition.seasonal
        
        if 'Fecha' in result_data.columns:
            fechas = pd.to_datetime(result_data['Fecha'])
            dayofyear = fechas.dt.dayofyear.values
            seasonal_cycle = seasonal.values[:self.imputation_config.seasonal_period]
            seasonal_full = np.array([seasonal_cycle[(d-1)%self.imputation_config.seasonal_period] 
                                    for d in dayofyear])
            return pd.Series(seasonal_full, index=current_series.index)
        else:
            return seasonal.reindex(current_series.index, fill_value=0)
    
    def _impute_residual_with_arima(self, resid: pd.Series, block: Dict[str, int], 
                                   block_index: int) -> None:
        """Impute residual component using ARIMA."""
        # Get context for ARIMA
        context_before = resid.iloc[max(0, block['start'] - 100):block['start']].dropna()
        context_after = resid.iloc[block['end'] + 1:min(len(resid), block['end'] + 101)].dropna()
        combined_data = pd.concat([context_before, context_after]).dropna()
        
        if len(combined_data) < self.imputation_config.min_context_for_arima:
            # Fallback to polynomial if insufficient context
            block_data = resid.iloc[block['start']:block['end']+1].copy()
            block_data = block_data.interpolate(
                method='polynomial', order=self.imputation_config.polynomial_order, 
                limit_direction='both'
            )
            resid.iloc[block['start']:block['end']+1] = block_data
        else:
            try:
                # Simple ARIMA for residual
                model = SARIMAX(combined_data, order=self.imputation_config.arima_order, 
                               seasonal_order=(0,0,0,0), enforce_stationarity=False, 
                               enforce_invertibility=False)
                fitted = model.fit(disp=False)
                
                try:
                    forecast = fitted.forecast(steps=block['length'])
                except AttributeError:
                    forecast = fitted.predict(start=len(combined_data), 
                                            end=len(combined_data)+block['length']-1)
                
                resid.iloc[block['start']:block['end']+1] = forecast
            except Exception as e:
                # Fallback to polynomial if ARIMA fails
                self.logger.warning(f"ARIMA failed for block {block_index+1}, using polynomial: {e}")
                block_data = resid.iloc[block['start']:block['end']+1].copy()
                block_data = block_data.interpolate(
                    method='polynomial', order=self.imputation_config.polynomial_order, 
                    limit_direction='both'
                )
                resid.iloc[block['start']:block['end']+1] = block_data
    
    def _apply_final_fallback(self, current_series: pd.Series) -> int:
        """Apply final fallback methods for any remaining NaNs."""
        if not current_series.isnull().any():
            return 0
        
        remaining_nans = current_series.isnull().sum()
        self.logger.info(f"Applying final fallback for {remaining_nans} remaining NaN values")
        
        # Interpolate in final series
        current_series.interpolate(method='linear', limit_direction='both', inplace=True)
        
        # If NaNs still remain, apply forward/backward fill
        if current_series.isnull().any():
            current_series.fillna(method='ffill', inplace=True)
            current_series.fillna(method='bfill', inplace=True)
        
        final_imputed = remaining_nans - current_series.isnull().sum()
        self.logger.info(f"Final fallback completed: {final_imputed} values imputed")
        return final_imputed
    
    def _mean_imputation(self, data: pd.DataFrame, target_column: str, **kwargs) -> pd.DataFrame:
        """Apply mean imputation to the target column."""
        result_data = data.copy()
        mean_value = result_data[target_column].mean()
        result_data[target_column] = result_data[target_column].fillna(mean_value)
        return result_data
    
    def _median_imputation(self, data: pd.DataFrame, target_column: str, **kwargs) -> pd.DataFrame:
        """Apply median imputation to the target column."""
        result_data = data.copy()
        median_value = result_data[target_column].median()
        result_data[target_column] = result_data[target_column].fillna(median_value)
        return result_data
    
    def _forward_fill(self, data: pd.DataFrame, target_column: str, **kwargs) -> pd.DataFrame:
        """Apply forward fill imputation to the target column."""
        result_data = data.copy()
        result_data[target_column] = result_data[target_column].fillna(method='ffill')
        return result_data
    
    def _backward_fill(self, data: pd.DataFrame, target_column: str, **kwargs) -> pd.DataFrame:
        """Apply backward fill imputation to the target column."""
        result_data = data.copy()
        result_data[target_column] = result_data[target_column].fillna(method='bfill')
        return result_data
    
    def _validate_station_data(self, station_data: pd.DataFrame, target_column: str) -> None:
        """Validate station data for imputation."""
        if not isinstance(station_data, pd.DataFrame):
            raise ValidationError("Station data must be a pandas DataFrame")
        
        if station_data.empty:
            raise ValidationError("Station data cannot be empty")
        
        if target_column not in station_data.columns:
            raise ValidationError(f"Target column '{target_column}' not found in station data")
        
        if not pd.api.types.is_numeric_dtype(station_data[target_column]):
            self.logger.warning(f"Target column '{target_column}' is not numeric. "
                              f"Some imputation methods may not work correctly.")
    
    def get_imputation_summary(self) -> Dict[str, Any]:
        """Get a summary of all imputation results."""
        if not self.imputation_results:
            return {'message': 'No imputation results available'}
        
        total_stations = len(self.imputation_results)
        successful_stations = sum(1 for result in self.imputation_results.values() 
                                if result.is_successful())
        failed_stations = total_stations - successful_stations
        
        total_original_missing = sum(result.original_missing_count 
                                   for result in self.imputation_results.values())
        total_imputed = sum(result.imputed_count 
                           for result in self.imputation_results.values())
        
        overall_imputation_rate = (total_imputed / total_original_missing * 100) if total_original_missing > 0 else 0
        
        method_usage = {}
        for result in self.imputation_results.values():
            method = result.imputation_method
            method_usage[method] = method_usage.get(method, 0) + 1
        
        return {
            'total_stations': total_stations,
            'successful_stations': successful_stations,
            'failed_stations': failed_stations,
            'success_rate': (successful_stations / total_stations * 100) if total_stations > 0 else 0,
            'total_original_missing': total_original_missing,
            'total_imputed': total_imputed,
            'overall_imputation_rate': overall_imputation_rate,
            'method_usage': method_usage,
            'average_processing_time': np.mean([result.processing_time_seconds 
                                              for result in self.imputation_results.values()])
        }
    
    def _convert_to_json_serializable(self, obj):
        """Convert object to JSON serializable format."""
        try:
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [self._convert_to_json_serializable(item) for item in obj]
            else:
                return obj
        except Exception:
            return str(obj)
    
    def save_results(self, output_path: Path) -> None:
        """Save imputation results to a file."""
        if not self.imputation_results:
            self.logger.warning("No imputation results to save")
            return
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            'summary': self.get_imputation_summary(),
            'station_results': {
                station: result.to_dict() 
                for station, result in self.imputation_results.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        results_dict = self._convert_to_json_serializable(results_dict)
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Imputation results saved to: {output_path}")
    
    def reset(self) -> None:
        """Reset the service state."""
        self.imputation_results.clear()
        self.logger.info("Service state reset")
    
    def _remove_large_missing_blocks(self, data: pd.DataFrame, target_column: str, 
                                   max_block_days: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Remove large blocks of missing data and all data before them."""
        result_data = data.copy()
        series = result_data[target_column]
        
        missing_blocks = self._find_missing_blocks(series)
        large_blocks = [block for block in missing_blocks if block['length'] > max_block_days]
        
        cleaning_info = {
            'total_blocks_found': len(missing_blocks),
            'large_blocks_found': len(large_blocks),
            'max_block_days': max_block_days,
            'blocks_removed': []
        }
        
        if not large_blocks:
            return result_data, cleaning_info
        
        large_blocks.sort(key=lambda x: x['start'])
        last_block_end = max(block['end'] for block in large_blocks)
        
        rows_to_remove = set(range(0, last_block_end + 1))
        
        for i, block in enumerate(large_blocks):
            cleaning_info['blocks_removed'].append({
                'block_index': i + 1,
                'start_position': block['start'],
                'end_position': block['end'],
                'length_days': block['length'],
                'rows_removed': block['end'] - block['start'] + 1
            })
        
        if rows_to_remove:
            rows_to_keep = [i for i in range(len(result_data)) if i not in rows_to_remove]
            result_data = result_data.iloc[rows_to_keep].reset_index(drop=True)
        
        return result_data, cleaning_info
    
    def _find_missing_blocks(self, series: pd.Series) -> List[Dict[str, int]]:
        """Find all blocks of consecutive missing values in a series."""
        blocks = []
        if not isinstance(series, pd.Series):
            raise ValidationError("Input must be a pandas Series")
        
        missing_mask = series.isna()
        
        if not missing_mask.any():
            return blocks
        
        block_start = None
        
        for i, is_missing in enumerate(missing_mask):
            if is_missing and block_start is None:
                block_start = i
            elif not is_missing and block_start is not None:
                block_end = i - 1
                block_length = block_end - block_start + 1
                
                blocks.append({
                    'start': block_start,
                    'end': block_end,
                    'length': block_length
                })
                
                block_start = None
        
        if block_start is not None:
            block_end = len(series) - 1
            block_length = block_end - block_start + 1
            
            blocks.append({
                'start': block_start,
                'end': block_end,
                'length': block_length
            })
        
        return blocks
    
    def _select_imputation_method_by_block_size(self, data: pd.DataFrame, target_column: str) -> str:
        """Select imputation method based on the size of missing blocks."""
        series = data[target_column]
        missing_blocks = self._find_missing_blocks(series)
        
        if not missing_blocks:
            return ImputationMethod.LINEAR.value
        
        small_blocks = [block for block in missing_blocks 
                       if block['length'] <= self.imputation_config.small_block_threshold]
        medium_blocks = [block for block in missing_blocks 
                        if self.imputation_config.small_block_threshold < block['length'] <= self.imputation_config.medium_block_threshold]
        large_blocks = [block for block in missing_blocks 
                       if block['length'] > self.imputation_config.medium_block_threshold]
        
        total_small_missing = sum(block['length'] for block in small_blocks)
        total_medium_missing = sum(block['length'] for block in medium_blocks)
        total_large_missing = sum(block['length'] for block in large_blocks)
        
        self.logger.info(f"Block analysis: {len(small_blocks)} small blocks ({total_small_missing} values), "
                        f"{len(medium_blocks)} medium blocks ({total_medium_missing} values), "
                        f"{len(large_blocks)} large blocks ({total_large_missing} values)")
        
        return ImputationMethod.ARIMA.value
    
    def _analyze_missing_blocks(self, result: ImputationResult, total_count: int) -> Dict[str, Any]:
        """Analyze missing blocks in the data."""
        # This method is called from _add_station_result, so we need to get the data differently
        # For now, return a basic structure
        return {
            'total_blocks': 0,
            'small_blocks': 0,
            'medium_blocks': 0,
            'large_blocks': 0,
            'total_missing': 0,
            'small_missing': 0,
            'medium_missing': 0,
            'large_missing': 0
        }