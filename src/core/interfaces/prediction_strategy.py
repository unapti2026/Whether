"""
Prediction Strategy Interface

This module defines the abstract interface for prediction strategies.
All prediction strategies must implement this interface to ensure consistency.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PredictionConfig:
    """Configuration class for prediction strategies."""
    variable_type: str
    prediction_steps: int = 30
    num_lags: int = 7
    train_test_split: float = 0.8
    max_stations: Optional[int] = None
    
    # Prediction Horizon Configuration
    # New: Fixed horizon system (replaces percentage-based approach)
    use_fixed_horizon: bool = True  # Use fixed horizon instead of percentage
    prediction_horizon_weeks: int = 3  # Horizon in weeks (default: 3 weeks = 21 days)
    prediction_horizon_days: Optional[int] = None  # If specified, overrides weeks
    legacy_horizon_ratio: float = 0.2  # Legacy: percentage of series size (for backward compatibility)
    max_horizon_days: int = 60  # Maximum reasonable horizon (safety limit)
    
    # EEMD Configuration
    eemd_sd_thresh_values: List[float] = None
    eemd_nensembles: int = 20
    eemd_noise_factor: float = 0.1
    eemd_max_imfs: int = 10
    eemd_orthogonality_threshold: float = 0.1
    eemd_correlation_threshold: float = 0.1
    
    # Model Configuration
    svr_kernel: str = 'rbf'
    svr_c: float = 1.0
    svr_gamma: str = 'scale'
    sarimax_order: tuple = (1, 1, 0)
    sarimax_seasonal_order: tuple = (1, 0, 0, 365)
    
    # Temporal Weighting Configuration (Fase 2)
    use_temporal_weighting: bool = True
    temporal_weighting_method: str = "exponential"  # "exponential", "linear", "windowed"
    temporal_decay_factor: float = 0.1  # For exponential weighting (higher = more weight to recent)
    temporal_increment_factor: float = 1.0  # For linear weighting
    temporal_recent_window_days: int = 30  # Days with full weight (for windowed method)
    temporal_weighting_strength: float = 0.5  # Overall strength of weighting (0-1, 1 = full effect)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.eemd_sd_thresh_values is None:
            self.eemd_sd_thresh_values = [0.05, 0.1, 0.15, 0.2]
        
        # Validate horizon configuration
        if self.prediction_horizon_weeks < 1:
            raise ValueError("prediction_horizon_weeks must be at least 1")
        if self.prediction_horizon_days is not None and self.prediction_horizon_days < 1:
            raise ValueError("prediction_horizon_days must be at least 1")
        if not 0 < self.legacy_horizon_ratio < 1:
            raise ValueError("legacy_horizon_ratio must be between 0 and 1")
        if self.max_horizon_days < 1:
            raise ValueError("max_horizon_days must be at least 1")
        
        # Validate temporal weighting configuration
        if self.temporal_weighting_method not in ["exponential", "linear", "windowed"]:
            raise ValueError(f"temporal_weighting_method must be one of: exponential, linear, windowed")
        if self.temporal_decay_factor <= 0:
            raise ValueError("temporal_decay_factor must be positive")
        if self.temporal_increment_factor < 0:
            raise ValueError("temporal_increment_factor must be non-negative")
        if self.temporal_recent_window_days < 1:
            raise ValueError("temporal_recent_window_days must be at least 1")
        if not 0 <= self.temporal_weighting_strength <= 1:
            raise ValueError("temporal_weighting_strength must be between 0 and 1")
    
    def calculate_prediction_steps(self, series_length: int) -> int:
        """
        Calculate prediction steps based on configuration.
        
        Args:
            series_length: Length of the time series
            
        Returns:
            Number of prediction steps to generate
        """
        if self.use_fixed_horizon:
            # Use fixed horizon (weeks or days)
            if self.prediction_horizon_days is not None:
                horizon_days = self.prediction_horizon_days
            else:
                horizon_days = self.prediction_horizon_weeks * 7
            
            # Apply safety limits
            max_reasonable = min(int(series_length * 0.1), self.max_horizon_days)
            prediction_steps = min(horizon_days, max_reasonable)
            
            # Ensure at least 1 step
            return max(1, prediction_steps)
        else:
            # Legacy mode: use percentage
            return max(1, int(series_length * self.legacy_horizon_ratio))


@dataclass
class EEMDResult:
    """Data class representing EEMD decomposition result."""
    imfs: np.ndarray
    correlations: List[float]
    variance_explained: pd.DataFrame
    best_sd_thresh: float
    orthogonality_score: float
    decomposition_quality: Dict[str, float]
    
    @property
    def num_imfs(self) -> int:
        """Get the number of IMFs."""
        return self.imfs.shape[1]


@dataclass
class ModelTrainingResult:
    """Data class representing model training result."""
    svr_models: Dict[int, Any]  # IMF index -> SVR model
    sarimax_model: Dict[int, Any]  # IMF index -> SARIMAX model
    selected_imf_for_sarimax: int
    training_time: float
    success: bool
    error_message: Optional[str] = None
    imf_classifications: Optional[Dict[str, List[int]]] = None  # New field for IMF classifications


@dataclass
class PredictionResult:
    """Data class representing prediction result for a station."""
    station_name: str
    station_code: str
    original_data: pd.DataFrame
    imf_predictions: Dict[int, np.ndarray]
    final_prediction: np.ndarray
    future_dates: pd.DatetimeIndex
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    model_performance: Optional[Dict[str, float]] = None
    prediction_quality_metrics: Optional[Dict[str, float]] = None
    component_classification: Optional[Dict[str, List[int]]] = None
    reconstructed_signal: Optional[np.ndarray] = None
    
    @property
    def prediction_length(self) -> int:
        """Get the length of the prediction."""
        return len(self.final_prediction)


class PredictionStrategyInterface(ABC):
    """
    Abstract interface for prediction strategies.
    
    This interface defines the contract that all prediction strategies must implement.
    It ensures consistency across different types of prediction methods.
    """
    
    @abstractmethod
    def decompose_time_series(self, time_series: pd.Series) -> EEMDResult:
        """
        Decompose a time series using EEMD.
        
        Args:
            time_series: Input time series to decompose
            
        Returns:
            EEMDResult containing IMFs and metadata
            
        Raises:
            PredictionError: If decomposition fails
        """
        pass
    
    @abstractmethod
    def train_models(self, eemd_result: EEMDResult, time_series: pd.Series) -> ModelTrainingResult:
        """
        Train prediction models on EEMD components.
        
        Args:
            eemd_result: EEMD decomposition result
            time_series: Original time series
            
        Returns:
            ModelTrainingResult containing trained models
            
        Raises:
            PredictionError: If training fails
        """
        pass
    
    @abstractmethod
    def generate_predictions(self, 
                           eemd_result: EEMDResult,
                           model_result: ModelTrainingResult,
                           time_series: pd.Series) -> PredictionResult:
        """
        Generate future predictions using trained models.
        
        Args:
            eemd_result: EEMD decomposition result
            model_result: Trained models result
            time_series: Original time series
            
        Returns:
            PredictionResult containing predictions and metadata
            
        Raises:
            PredictionError: If prediction generation fails
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data is suitable for prediction.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid for prediction, False otherwise
            
        Raises:
            ValidationError: If validation fails
        """
        pass
    
    @abstractmethod
    def get_prediction_info(self) -> Dict[str, Any]:
        """
        Get information about the prediction strategy.
        
        Returns:
            Dictionary containing prediction metadata
        """
        pass
    
    @abstractmethod
    def save_results(self, result: PredictionResult, output_path: str) -> None:
        """
        Save prediction results to files.
        
        Args:
            result: Prediction result to save
            output_path: Path where to save the results
            
        Raises:
            PredictionError: If saving fails
        """
        pass 