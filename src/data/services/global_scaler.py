"""
Global Scaler Service for Consistent Model Scaling.

This service provides:
- Consistent scaling across different model types (SARIMAX, SVR)
- Proper handling of scale differences in reconstruction
- Validation of scale consistency
- Adaptive scaling based on IMF characteristics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import warnings

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Different scaling strategies for different model types."""
    NONE = "none"  # No scaling (for SARIMAX)
    STANDARD = "standard"  # StandardScaler (for SVR)
    ROBUST = "robust"  # RobustScaler (for noisy data)
    MINMAX = "minmax"  # MinMaxScaler (for bounded data)


@dataclass
class GlobalScalingConfig:
    """Configuration for global scaling strategy."""
    # Default scaling strategies
    sarimax_scaling: ScalingStrategy = ScalingStrategy.NONE
    svr_scaling: ScalingStrategy = ScalingStrategy.STANDARD
    
    # Consistency validation
    enable_scale_validation: bool = True
    scale_tolerance: float = 0.1  # Maximum allowed scale difference
    
    # Adaptive scaling
    enable_adaptive_scaling: bool = True
    noise_threshold: float = 0.3  # Threshold for noisy IMFs
    variance_threshold: float = 0.1  # Threshold for low variance
    
    # Reconstruction settings
    normalize_reconstruction: bool = True
    reconstruction_method: str = "weighted_sum"  # "weighted_sum", "simple_sum"


@dataclass
class ScalingResult:
    """Result of scaling operation."""
    original_data: np.ndarray
    scaled_data: np.ndarray
    scaler: Optional[BaseEstimator]
    strategy: ScalingStrategy
    scale_factors: Dict[str, float]
    is_consistent: bool
    validation_metrics: Dict[str, float]


class GlobalScaler:
    """
    Global scaler for ensuring consistency across different model types.
    
    This scaler:
    - Applies appropriate scaling for each model type
    - Validates scale consistency across components
    - Provides reconstruction normalization
    - Handles adaptive scaling based on data characteristics
    """
    
    def __init__(self, config: Optional[GlobalScalingConfig] = None):
        """
        Initialize the global scaler.
        
        Args:
            config: Configuration for global scaling
        """
        self.config = config or GlobalScalingConfig()
        self.logger = logging.getLogger(__name__)
        self.scalers: Dict[int, BaseEstimator] = {}
        self.scale_factors: Dict[int, Dict[str, float]] = {}
        self.scaling_strategies: Dict[int, ScalingStrategy] = {}
    
    def scale_imf_for_model(
        self, 
        imf_data: np.ndarray, 
        imf_idx: int, 
        model_type: str,
        imf_properties: Optional[Dict[str, float]] = None
    ) -> ScalingResult:
        """
        Scale IMF data for specific model type.
        
        Args:
            imf_data: IMF series data
            imf_idx: Index of the IMF
            model_type: Type of model ('sarimax', 'svr')
            imf_properties: Optional IMF properties for adaptive scaling
            
        Returns:
            ScalingResult with scaled data and metadata
        """
        self.logger.debug(f"Scaling IMF {imf_idx + 1} for {model_type.upper()}")
        
        # Determine scaling strategy
        strategy = self._determine_scaling_strategy(model_type, imf_properties)
        
        # Apply scaling
        if strategy == ScalingStrategy.NONE:
            # No scaling for SARIMAX
            scaled_data = imf_data.copy()
            scaler = None
            scale_factors = {
                'mean': np.mean(imf_data),
                'std': np.std(imf_data),
                'min': np.min(imf_data),
                'max': np.max(imf_data)
            }
        else:
            # Apply appropriate scaler
            scaler = self._create_scaler(strategy)
            scaled_data = scaler.fit_transform(imf_data.reshape(-1, 1)).flatten()
            
            # Calculate scale factors
            scale_factors = {
                'mean': scaler.mean_[0] if hasattr(scaler, 'mean_') else np.mean(imf_data),
                'std': scaler.scale_[0] if hasattr(scaler, 'scale_') else np.std(imf_data),
                'min': scaler.data_min_[0] if hasattr(scaler, 'data_min_') else np.min(imf_data),
                'max': scaler.data_max_[0] if hasattr(scaler, 'data_max_') else np.max(imf_data)
            }
        
        # Store scaling information
        self.scalers[imf_idx] = scaler
        self.scale_factors[imf_idx] = scale_factors
        self.scaling_strategies[imf_idx] = strategy
        
        # Validate consistency
        is_consistent = self._validate_scale_consistency(imf_idx, scale_factors)
        
        # Calculate validation metrics
        validation_metrics = self._calculate_validation_metrics(imf_data, scaled_data, strategy)
        
        self.logger.debug(f"  IMF {imf_idx + 1} scaled with {strategy.value} strategy")
        
        return ScalingResult(
            original_data=imf_data,
            scaled_data=scaled_data,
            scaler=scaler,
            strategy=strategy,
            scale_factors=scale_factors,
            is_consistent=is_consistent,
            validation_metrics=validation_metrics
        )
    
    def scale_features_for_svr(
        self, 
        features: np.ndarray, 
        imf_idx: int,
        fit: bool = True
    ) -> np.ndarray:
        """
        Scale features for SVR training/prediction.
        
        Args:
            features: Feature matrix for SVR
            imf_idx: Index of the IMF
            fit: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            Scaled feature matrix
        """
        if imf_idx not in self.scalers or self.scalers[imf_idx] is None:
            # Use default StandardScaler for features
            scaler = StandardScaler()
            if fit:
                return scaler.fit_transform(features)
            else:
                self.logger.warning(f"No scaler found for IMF {imf_idx + 1}, using default scaling")
                return features  # Return unscaled if no scaler available
        
        # Use the stored scaler
        scaler = self.scalers[imf_idx]
        if fit:
            return scaler.fit_transform(features)
        else:
            return scaler.transform(features)
    
    def normalize_reconstruction(
        self, 
        predictions: Dict[int, np.ndarray],
        original_series: np.ndarray
    ) -> np.ndarray:
        """
        Normalize the final reconstruction to ensure consistency.
        
        Args:
            predictions: Dictionary of predictions by IMF index
            original_series: Original time series for reference
            
        Returns:
            Normalized reconstruction
        """
        if not self.config.normalize_reconstruction:
            # Simple sum without normalization
            reconstruction = np.sum(list(predictions.values()), axis=0)
        else:
            # Weighted sum based on scale factors
            reconstruction = self._weighted_reconstruction(predictions, original_series)
        
        # Validate reconstruction scale
        self._validate_reconstruction_scale(reconstruction, original_series)
        
        return reconstruction
    
    def _determine_scaling_strategy(
        self, 
        model_type: str, 
        imf_properties: Optional[Dict[str, float]] = None
    ) -> ScalingStrategy:
        """Determine appropriate scaling strategy based on model type and IMF properties."""
        
        if model_type.lower() == 'sarimax':
            return ScalingStrategy.NONE
        
        elif model_type.lower() == 'svr':
            if not self.config.enable_adaptive_scaling or imf_properties is None:
                return self.config.svr_scaling
            
            # Adaptive scaling based on IMF properties
            variance = imf_properties.get('variance', 1.0)
            complexity = imf_properties.get('complexity', 0.5)
            
            # Use robust scaling for noisy IMFs
            if complexity > self.config.noise_threshold:
                return ScalingStrategy.ROBUST
            
            # Use minmax scaling for low variance IMFs
            elif variance < self.config.variance_threshold:
                return ScalingStrategy.MINMAX
            
            # Default to standard scaling
            else:
                return ScalingStrategy.STANDARD
        
        else:
            self.logger.warning(f"Unknown model type: {model_type}, using standard scaling")
            return ScalingStrategy.STANDARD
    
    def _create_scaler(self, strategy: ScalingStrategy) -> BaseEstimator:
        """Create appropriate scaler based on strategy."""
        if strategy == ScalingStrategy.STANDARD:
            return StandardScaler()
        elif strategy == ScalingStrategy.ROBUST:
            return RobustScaler()
        elif strategy == ScalingStrategy.MINMAX:
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling strategy: {strategy}")
    
    def _validate_scale_consistency(self, imf_idx: int, scale_factors: Dict[str, float]) -> bool:
        """Validate scale consistency across IMFs."""
        if not self.config.enable_scale_validation:
            return True
        
        if len(self.scale_factors) < 2:
            return True
        
        # Compare with other IMFs
        for other_idx, other_factors in self.scale_factors.items():
            if other_idx == imf_idx:
                continue
            
            # Check scale difference
            scale_diff = abs(scale_factors['std'] - other_factors['std']) / max(scale_factors['std'], 1e-8)
            
            if scale_diff > self.config.scale_tolerance:
                self.logger.warning(f"Scale inconsistency detected: IMF {imf_idx + 1} vs IMF {other_idx + 1}")
                return False
        
        return True
    
    def _calculate_validation_metrics(
        self, 
        original: np.ndarray, 
        scaled: np.ndarray, 
        strategy: ScalingStrategy
    ) -> Dict[str, float]:
        """Calculate validation metrics for scaling."""
        metrics = {
            'original_mean': np.mean(original),
            'original_std': np.std(original),
            'scaled_mean': np.mean(scaled),
            'scaled_std': np.std(scaled),
            'preservation_ratio': np.std(scaled) / max(np.std(original), 1e-8)
        }
        
        if strategy != ScalingStrategy.NONE:
            metrics['scaling_factor'] = np.std(original) / max(np.std(scaled), 1e-8)
        
        return metrics
    
    def _weighted_reconstruction(
        self, 
        predictions: Dict[int, np.ndarray], 
        original_series: np.ndarray
    ) -> np.ndarray:
        """Perform weighted reconstruction based on scale factors."""
        if self.config.reconstruction_method == "weighted_sum":
            # Weight by inverse of scale factor (smaller scale = higher weight)
            weights = {}
            total_weight = 0
            
            for imf_idx, prediction in predictions.items():
                if imf_idx in self.scale_factors:
                    # Weight inversely proportional to scale
                    scale_factor = self.scale_factors[imf_idx]['std']
                    weight = 1.0 / max(scale_factor, 1e-8)
                    weights[imf_idx] = weight
                    total_weight += weight
                else:
                    weights[imf_idx] = 1.0
                    total_weight += 1.0
            
            # Normalize weights
            for imf_idx in weights:
                weights[imf_idx] /= total_weight
            
            # Weighted sum
            reconstruction = np.zeros_like(list(predictions.values())[0])
            for imf_idx, prediction in predictions.items():
                reconstruction += weights[imf_idx] * prediction
            
        else:
            # Simple sum
            reconstruction = np.sum(list(predictions.values()), axis=0)
        
        return reconstruction
    
    def _validate_reconstruction_scale(self, reconstruction: np.ndarray, original: np.ndarray):
        """Validate that reconstruction scale is reasonable."""
        recon_std = np.std(reconstruction)
        orig_std = np.std(original)
        
        scale_ratio = recon_std / max(orig_std, 1e-8)
        
        if scale_ratio > 10 or scale_ratio < 0.1:
            self.logger.warning(f"Reconstruction scale ratio ({scale_ratio:.3f}) is outside expected range")
        
        self.logger.debug(f"Reconstruction scale ratio: {scale_ratio:.3f}")
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get summary of all scaling operations."""
        summary = {
            'total_imfs': len(self.scalers),
            'scaling_strategies': {f"IMF_{idx+1}": strategy.value for idx, strategy in self.scaling_strategies.items()},
            'scale_factors': self.scale_factors,
            'consistency_check': all(self._validate_scale_consistency(idx, factors) 
                                   for idx, factors in self.scale_factors.items())
        }
        
        return summary
    
    def reset(self):
        """Reset all scaling information."""
        self.scalers.clear()
        self.scale_factors.clear()
        self.scaling_strategies.clear()


def create_global_scaling_config(
    model_type: str = 'hybrid',
    series_length: int = 1000,
    imf_count: int = 7
) -> GlobalScalingConfig:
    """
    Create adaptive global scaling configuration.
    
    Args:
        model_type: Type of model ('hybrid', 'svr_only', 'sarimax_only')
        series_length: Length of the time series
        imf_count: Number of IMFs
        
    Returns:
        GlobalScalingConfig with adaptive settings
    """
    config = GlobalScalingConfig()
    
    # Adjust based on series length
    if series_length < 500:
        config.scale_tolerance = 0.15  # More tolerant for short series
        config.enable_adaptive_scaling = False
    elif series_length > 5000:
        config.scale_tolerance = 0.08  # Stricter for long series
        config.enable_adaptive_scaling = True
    
    # Adjust based on IMF count
    if imf_count > 10:
        config.normalize_reconstruction = True
        config.reconstruction_method = "weighted_sum"
    else:
        config.normalize_reconstruction = False
        config.reconstruction_method = "simple_sum"
    
    # Adjust based on model type
    if model_type == 'svr_only':
        config.sarimax_scaling = ScalingStrategy.STANDARD  # Apply to all
    elif model_type == 'sarimax_only':
        config.svr_scaling = ScalingStrategy.NONE  # No scaling needed
    
    return config
			