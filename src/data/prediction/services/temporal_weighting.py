"""
Temporal Weighting Module

This module provides functions for calculating temporal weights to prioritize
recent data when training prediction models.

The module supports multiple weighting strategies:
- Exponential: Exponential decay weighting
- Linear: Linear increasing weighting
- Windowed: Windowed sampling with different weights for different time periods
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TemporalWeighting:
    """
    Class for calculating temporal weights for time series data.
    """
    
    @staticmethod
    def calculate_exponential_weights(
        series_length: int,
        decay_factor: float = 0.1,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Calculate exponential decay weights for time series.
        
        Formula: w(t) = exp(-α * (T - t))
        Where:
        - t = temporal index (0 = oldest, T = most recent)
        - T = index of most recent data point
        - α = decay factor (higher = more weight to recent data)
        
        Args:
            series_length: Length of the time series
            decay_factor: Decay factor (α). Higher values give more weight to recent data.
                          Typical range: 0.01-0.5. Default: 0.1
            normalize: If True, normalize weights to sum to series_length
            
        Returns:
            Array of weights (most recent has highest weight)
        """
        if series_length <= 0:
            raise ValueError("Series length must be positive")
        
        if decay_factor <= 0:
            raise ValueError("Decay factor must be positive")
        
        # Create indices: 0 = oldest, series_length-1 = most recent
        t = np.arange(series_length)
        T = series_length - 1
        
        # Calculate exponential weights: w(t) = exp(-α * (T - t))
        # More recent data (higher t) gets higher weight
        weights = np.exp(-decay_factor * (T - t))
        
        # Normalize if requested (so sum equals series_length)
        if normalize:
            weights = weights * (series_length / weights.sum())
        
        logger.debug(f"Calculated exponential weights: min={weights.min():.4f}, max={weights.max():.4f}, "
                    f"mean={weights.mean():.4f}, decay_factor={decay_factor}")
        
        return weights
    
    @staticmethod
    def calculate_linear_weights(
        series_length: int,
        increment_factor: float = 1.0,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Calculate linear increasing weights for time series.
        
        Formula: w(t) = 1 + β * (t / T)
        Where:
        - t = temporal index (0 = oldest, T = most recent)
        - T = index of most recent data point
        - β = increment factor
        
        Args:
            series_length: Length of the time series
            increment_factor: Increment factor (β). Higher values give more weight to recent data.
                              Typical range: 0.5-5.0. Default: 1.0
            normalize: If True, normalize weights to sum to series_length
            
        Returns:
            Array of weights (most recent has highest weight)
        """
        if series_length <= 0:
            raise ValueError("Series length must be positive")
        
        if increment_factor < 0:
            raise ValueError("Increment factor must be non-negative")
        
        # Create indices: 0 = oldest, series_length-1 = most recent
        t = np.arange(series_length)
        T = max(1, series_length - 1)  # Avoid division by zero
        
        # Calculate linear weights: w(t) = 1 + β * (t / T)
        # More recent data (higher t) gets higher weight
        weights = 1.0 + increment_factor * (t / T)
        
        # Normalize if requested
        if normalize:
            weights = weights * (series_length / weights.sum())
        
        logger.debug(f"Calculated linear weights: min={weights.min():.4f}, max={weights.max():.4f}, "
                    f"mean={weights.mean():.4f}, increment_factor={increment_factor}")
        
        return weights
    
    @staticmethod
    def apply_windowed_sampling(
        series: np.ndarray,
        recent_window_days: int = 30,
        sampling_ratios: Optional[dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply windowed sampling to prioritize recent data.
        
        Strategy: Use different sampling ratios for different time windows:
        - Last N days: 100% (all data)
        - Days N+1 to 2N: 80% (random sampling)
        - Days 2N+1 to 3N: 60% (random sampling)
        - Days 3N+1 to 6N: 40% (random sampling)
        - Rest: 20% (random sampling)
        
        Args:
            series: Time series data
            recent_window_days: Number of recent days to include 100%
            sampling_ratios: Optional dict with custom sampling ratios.
                           Format: {window_name: (start_offset, end_offset, ratio)}
                           Example: {'recent': (0, 30, 1.0), 'medium': (30, 60, 0.8)}
            
        Returns:
            Tuple of (sampled_series, original_indices)
        """
        n = len(series)
        
        if n <= recent_window_days:
            # Series too short, return all data
            logger.debug(f"Series length ({n}) <= recent_window_days ({recent_window_days}), returning all data")
            return series, np.arange(n)
        
        # Default sampling ratios if not provided
        if sampling_ratios is None:
            sampling_ratios = {
                'recent': (0, recent_window_days, 1.0),      # Last 30 days: 100%
                'medium': (recent_window_days, 2 * recent_window_days, 0.8),  # 31-60: 80%
                'far': (2 * recent_window_days, 3 * recent_window_days, 0.6),  # 61-90: 60%
                'distant': (3 * recent_window_days, 6 * recent_window_days, 0.4),  # 91-180: 40%
                'very_distant': (6 * recent_window_days, None, 0.2)  # Rest: 20%
            }
        
        indices = []
        np.random.seed(42)  # For reproducibility
        
        # Apply sampling for each window
        for window_name, (start_offset, end_offset, ratio) in sampling_ratios.items():
            if end_offset is None:
                # Last window: from start_offset to beginning
                window_start = max(0, n - start_offset)
                window_end = 0
            else:
                window_start = max(0, n - end_offset)
                window_end = max(0, n - start_offset)
            
            if window_start <= window_end:
                continue  # Invalid window
            
            window_indices = list(range(window_end, window_start))
            
            if len(window_indices) == 0:
                continue
            
            # Calculate sample size
            sample_size = max(1, int(len(window_indices) * ratio))
            sample_size = min(sample_size, len(window_indices))
            
            # Sample indices
            if sample_size == len(window_indices):
                # Include all
                indices.extend(window_indices)
            else:
                sampled = np.random.choice(window_indices, size=sample_size, replace=False)
                indices.extend(sampled.tolist())
        
        # Sort indices to maintain temporal order
        indices = sorted(set(indices))
        
        # Extract sampled data
        sampled_series = series[indices]
        
        logger.info(f"Windowed sampling: {len(series)} -> {len(sampled_series)} points "
                   f"({len(sampled_series)/len(series)*100:.1f}%)")
        
        return sampled_series, np.array(indices)
    
    @staticmethod
    def calculate_sample_weights_for_svr(
        series: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "exponential",
        decay_factor: float = 0.1,
        increment_factor: float = 1.0
    ) -> np.ndarray:
        """
        Calculate sample weights for SVR training based on temporal position.
        
        Args:
            series: Original time series (for determining temporal position)
            X: Feature matrix (samples x features)
            y: Target vector (samples)
            method: Weighting method ("exponential" or "linear")
            decay_factor: For exponential weighting
            increment_factor: For linear weighting
            
        Returns:
            Array of sample weights (one per sample in X/y)
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if len(X) == 0:
            return np.array([])
        
        # Determine temporal position of each sample
        # Each sample in X corresponds to a position in the original series
        # The position is: num_lags + sample_index
        # We need to map from sample index to original series position
        
        # For samples created from series with num_lags, the first sample
        # corresponds to position num_lags in the original series
        # We'll use the length of y to infer the starting position
        # Actually, we need to know num_lags, but we can infer from series and y lengths
        # If series has length N and we use num_lags L, then y has length N - L
        # So: num_lags = len(series) - len(y)
        
        num_lags = len(series) - len(y)
        if num_lags < 0:
            logger.warning(f"Cannot determine num_lags: series_len={len(series)}, y_len={len(y)}")
            # Fallback: assume all samples are from recent data
            num_lags = 0
        
        # Calculate weights for the portion of series that corresponds to y
        # y corresponds to series[num_lags:]
        relevant_series = series[num_lags:]
        
        if len(relevant_series) != len(y):
            logger.warning(f"Length mismatch: relevant_series={len(relevant_series)}, y={len(y)}")
            # Use y length as reference
            relevant_series = series[-len(y):]
        
        # Calculate weights for relevant portion
        if method == "exponential":
            weights = TemporalWeighting.calculate_exponential_weights(
                len(relevant_series),
                decay_factor=decay_factor,
                normalize=False  # Don't normalize here, we'll normalize after
            )
        elif method == "linear":
            weights = TemporalWeighting.calculate_linear_weights(
                len(relevant_series),
                increment_factor=increment_factor,
                normalize=False
            )
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        # Normalize so that mean weight is 1.0 (standard practice for sample_weight)
        weights = weights / weights.mean()
        
        logger.debug(f"Calculated SVR sample weights: min={weights.min():.4f}, max={weights.max():.4f}, "
                    f"mean={weights.mean():.4f}, method={method}")
        
        return weights



