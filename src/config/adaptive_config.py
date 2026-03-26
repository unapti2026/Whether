"""
Adaptive Configuration for Memory Optimization

This module provides adaptive configuration that adjusts parameters
based on data size to optimize memory usage and processing time.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Adaptive configuration that adjusts parameters based on data size."""
    
    series_length: int
    variable_type: str
    
    def __post_init__(self):
        """Calculate adaptive parameters based on series length."""
        self._calculate_adaptive_parameters()
    
    def _calculate_adaptive_parameters(self) -> None:
        """Calculate adaptive parameters based on series length."""
        
        # EEMD Ensembles: Adaptive based on series length
        if self.series_length > 15000:
            self.eemd_nensembles = 5
            self.use_downsampling = True
            self.downsample_freq = '2D'  # Every 2 days
        elif self.series_length > 10000:
            self.eemd_nensembles = 10
            self.use_downsampling = False
            self.downsample_freq = None
        elif self.series_length > 5000:
            self.eemd_nensembles = 15
            self.use_downsampling = False
            self.downsample_freq = None
        else:
            self.eemd_nensembles = 20
            self.use_downsampling = False
            self.downsample_freq = None
        
        # EEMD sd_thresh values: Reduce for large series
        if self.series_length > 10000:
            self.eemd_sd_thresh_values = [0.1, 0.15]  # Only 2 values
        else:
            self.eemd_sd_thresh_values = [0.05, 0.1, 0.15, 0.2]  # All 4 values
        
        # SARIMAX Configuration: Optimized for memory with DATA LIMITATION
        self._calculate_sarimax_parameters()
        
        # SVR Configuration: Adaptive training sample
        if self.series_length > 8000:
            self.svr_training_sample = min(5000, self.series_length // 2)
            self.use_svr_sampling = True
        else:
            self.svr_training_sample = self.series_length
            self.use_svr_sampling = False
        
        # Prediction Configuration: Adaptive window
        self.prediction_window = min(100, max(20, self.series_length // 20))
        
        # Memory Management
        self.enable_garbage_collection = True
        self.chunk_size = min(3000, self.series_length // 3) if self.series_length > 5000 else self.series_length
        
        logger.info(f"Adaptive config for {self.variable_type}: {self.series_length} points, "
                   f"EEMD ensembles: {self.eemd_nensembles}, SARIMAX temporal validation: {self.sarimax_max_data_points} days")
    
    def _calculate_sarimax_parameters(self) -> None:
        """Calculate SARIMAX parameters optimized for memory usage with DATA LIMITATION."""
        
        # Minimum records for SARIMAX: 365 (1 year of daily data)
        self.sarimax_min_records = 365
        
        # CRITICAL: DATA LIMITATION FOR SARIMAX
        # Limit the amount of data used for SARIMAX training to prevent memory issues
        # TEMPORAL VALIDATION: Maximum 365 days for SARIMAX training (parametrizable)
        # Note: This is different from max_missing_block_days (548) which is for imputation
        self.sarimax_max_data_points = 365  # Maximum 1 year of daily data for SARIMAX training
        self.sarimax_data_years = 1.0  # Always 1 year
        
        # Additional parametrizable settings
        self.sarimax_data_limit_days = 365  # Parametrizable limit in days for SARIMAX training
        self.sarimax_use_temporal_validation = True  # Enable temporal validation
        
        # SARIMAX Model Complexity: Adaptive based on data size
        # METEOROLOGICAL DATA: Always use annual seasonality (365 days) for proven annual cycles
        if self.series_length > 10000:
            # Ultra-light configuration for very large series
            self.sarimax_order = (1, 0, 0)  # Simple AR(1)
            self.sarimax_seasonal_order = (1, 0, 0, 365)  # Annual seasonality (365 days)
            self.sarimax_max_iter = 30  # Reduced iterations
            self.sarimax_use_simple = True
            self.sarimax_timeout = 20  # Reduced timeout
        elif self.series_length > 5000:
            # Light configuration for large series
            self.sarimax_order = (1, 1, 0)  # ARIMA(1,1,0)
            self.sarimax_seasonal_order = (1, 0, 0, 365)  # Annual seasonality (365 days)
            self.sarimax_max_iter = 50  # Reduced iterations
            self.sarimax_use_simple = False
            self.sarimax_timeout = 40  # Reduced timeout
        else:
            # Standard configuration for smaller series
            self.sarimax_order = (1, 1, 0)  # ARIMA(1,1,0)
            self.sarimax_seasonal_order = (1, 0, 0, 365)  # Annual seasonality (365 days)
            self.sarimax_max_iter = 100  # Standard iterations
            self.sarimax_use_simple = False
            self.sarimax_timeout = 60  # Standard timeout
        
        # Additional SARIMAX settings for memory optimization
        self.sarimax_enforce_stationarity = False  # Faster training
        self.sarimax_enforce_invertibility = False  # Faster training
        self.sarimax_disp = False  # No verbose output
        self.sarimax_max_models_per_station = 1  # Maximum 1 SARIMAX model per station
        
        # Memory optimization flags
        self.sarimax_enable_memory_cleanup = True
        self.sarimax_force_garbage_collection = True
        self.sarimax_use_chunked_training = self.series_length > 3000


def create_adaptive_config(series_length: int, variable_type: str) -> AdaptiveConfig:
    """
    Create adaptive configuration based on series length.
    
    Args:
        series_length: Length of the time series
        variable_type: Type of meteorological variable
        
    Returns:
        AdaptiveConfig instance with optimized parameters
    """
    return AdaptiveConfig(series_length=series_length, variable_type=variable_type) 