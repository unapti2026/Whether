"""
Advanced Temporal Features Service for SVR Models.

This service provides sophisticated temporal features for time series prediction,
including rolling statistics, seasonality detection, trend analysis, and cyclical patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)


class AdvancedTemporalFeatures:
    """
    Service for generating advanced temporal features for time series prediction.
    
    Features include:
    - Rolling statistics (mean, std, min, max, quantiles, skewness, kurtosis)
    - Seasonality features (annual, monthly, weekly patterns)
    - Trend features (linear, quadratic, exponential trends)
    - Cyclical patterns (autocorrelation, periodicity)
    - Volatility features (GARCH-like, realized volatility)
    - Momentum features (rate of change, acceleration)
    - Statistical features (percentiles, z-scores, outliers)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Advanced Temporal Features service.
        
        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for temporal features."""
        return {
            'rolling_windows': [3, 7, 14, 30, 90],  # Multiple rolling windows
            'seasonal_periods': [7, 30, 365],       # Weekly, monthly, annual
            'trend_windows': [7, 14, 30, 90],       # Trend analysis windows
            'volatility_windows': [5, 10, 20],      # Volatility calculation windows
            'momentum_windows': [1, 3, 7, 14],      # Momentum calculation windows
            'percentiles': [5, 25, 50, 75, 95],     # Percentile features
            'autocorr_lags': [1, 7, 14, 30],        # Autocorrelation lags
            'enable_fft_features': True,            # FFT-based features
            'enable_wavelet_features': False,       # Wavelet features (if available)
            'max_features': 100                     # Maximum number of features
        }
    
    def generate_advanced_features(self, series: np.ndarray, num_lags: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate comprehensive advanced temporal features for time series.
        
        Args:
            series: Time series data
            num_lags: Number of basic lagged features
            
        Returns:
            Tuple of (X, y) arrays with advanced features
        """
        self.logger.info(f"Generating advanced temporal features for series with {len(series)} points")
        
        X, y = [], []
        
        for i in range(num_lags, len(series)):
            # Basic lagged features
            lagged_features = list(series[i-num_lags:i])
            
            # Advanced temporal features
            advanced_features = []
            
            # 1. Rolling Statistics Features
            rolling_features = self._generate_rolling_features(series[:i+1])
            advanced_features.extend(rolling_features)
            
            # 2. Seasonality Features
            seasonality_features = self._generate_seasonality_features(series[:i+1])
            advanced_features.extend(seasonality_features)
            
            # 3. Trend Features
            trend_features = self._generate_trend_features(series[:i+1])
            advanced_features.extend(trend_features)
            
            # 4. Volatility Features
            volatility_features = self._generate_volatility_features(series[:i+1])
            advanced_features.extend(volatility_features)
            
            # 5. Momentum Features
            momentum_features = self._generate_momentum_features(series[:i+1])
            advanced_features.extend(momentum_features)
            
            # 6. Statistical Features
            statistical_features = self._generate_statistical_features(series[:i+1])
            advanced_features.extend(statistical_features)
            
            # 7. Cyclical Features
            cyclical_features = self._generate_cyclical_features(series[:i+1])
            advanced_features.extend(cyclical_features)
            
            # 8. FFT Features (if enabled)
            if self.config.get('enable_fft_features', True):
                fft_features = self._generate_fft_features(series[:i+1])
                advanced_features.extend(fft_features)
            
            # Combine all features
            all_features = lagged_features + advanced_features
            X.append(all_features)
            y.append(series[i])
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        self.logger.info(f"Generated {X_array.shape[1]} features (basic lags: {num_lags}, advanced: {X_array.shape[1] - num_lags})")
        
        return X_array, y_array
    
    def _generate_rolling_features(self, series: np.ndarray) -> List[float]:
        """Generate rolling statistics features."""
        features = []
        
        for window in self.config['rolling_windows']:
            if len(series) >= window:
                window_data = series[-window:]
                
                # Basic statistics
                features.extend([
                    np.mean(window_data),      # Rolling mean
                    np.std(window_data),       # Rolling std
                    np.min(window_data),       # Rolling min
                    np.max(window_data),       # Rolling max
                    np.ptp(window_data),       # Rolling range
                    np.median(window_data),    # Rolling median
                    stats.skew(window_data),   # Rolling skewness
                    stats.kurtosis(window_data) # Rolling kurtosis
                ])
                
                # Quantile features
                for q in self.config['percentiles']:
                    features.append(np.percentile(window_data, q))
                
                # Additional rolling features
                features.extend([
                    np.var(window_data),       # Rolling variance
                    np.mean(np.abs(window_data - np.mean(window_data))),  # Mean absolute deviation
                    np.sum(np.diff(window_data) > 0) / (len(window_data) - 1),  # Upward trend ratio
                ])
            else:
                # Pad with zeros if not enough data
                features.extend([0.0] * (8 + len(self.config['percentiles']) + 3))
        
        return features
    
    def _generate_seasonality_features(self, series: np.ndarray) -> List[float]:
        """Generate seasonality-related features."""
        features = []
        
        for period in self.config['seasonal_periods']:
            if len(series) >= period * 2:
                # Seasonal decomposition approximation
                seasonal_pattern = self._extract_seasonal_pattern(series, period)
                features.extend([
                    np.mean(seasonal_pattern),     # Seasonal mean
                    np.std(seasonal_pattern),      # Seasonal std
                    np.max(seasonal_pattern),      # Seasonal max
                    np.min(seasonal_pattern),      # Seasonal min
                    np.ptp(seasonal_pattern),      # Seasonal range
                ])
                
                # Seasonal autocorrelation
                if len(seasonal_pattern) > 1:
                    autocorr = np.corrcoef(seasonal_pattern[:-1], seasonal_pattern[1:])[0, 1]
                    features.append(autocorr if not np.isnan(autocorr) else 0.0)
                else:
                    features.append(0.0)
            else:
                features.extend([0.0] * 6)
        
        return features
    
    def _extract_seasonal_pattern(self, series: np.ndarray, period: int) -> np.ndarray:
        """Extract seasonal pattern from time series."""
        if len(series) < period:
            return np.array([0.0])
        
        # Simple seasonal pattern extraction
        seasonal_values = []
        for i in range(period):
            indices = range(i, len(series), period)
            if indices:
                seasonal_values.append(np.mean(series[indices]))
        
        return np.array(seasonal_values)
    
    def _generate_trend_features(self, series: np.ndarray) -> List[float]:
        """Generate trend-related features."""
        features = []
        
        for window in self.config['trend_windows']:
            if len(series) >= window:
                window_data = series[-window:]
                x = np.arange(len(window_data))
                
                # Linear trend
                slope_linear, intercept_linear = np.polyfit(x, window_data, 1)
                features.extend([slope_linear, intercept_linear])
                
                # Quadratic trend
                if len(window_data) >= 3:
                    coeffs_quad = np.polyfit(x, window_data, 2)
                    features.extend([coeffs_quad[0], coeffs_quad[1]])  # a, b coefficients
                else:
                    features.extend([0.0, 0.0])
                
                # Trend strength (R-squared)
                y_pred_linear = slope_linear * x + intercept_linear
                r_squared = 1 - np.sum((window_data - y_pred_linear) ** 2) / np.sum((window_data - np.mean(window_data)) ** 2)
                features.append(r_squared if not np.isnan(r_squared) else 0.0)
                
                # Trend direction
                features.append(1.0 if slope_linear > 0 else -1.0)
                
            else:
                features.extend([0.0] * 6)
        
        return features
    
    def _generate_volatility_features(self, series: np.ndarray) -> List[float]:
        """Generate volatility-related features."""
        features = []
        
        for window in self.config['volatility_windows']:
            if len(series) >= window:
                window_data = series[-window:]
                returns = np.diff(window_data) / window_data[:-1] if len(window_data) > 1 else np.array([0.0])
                
                if len(returns) > 0:
                    features.extend([
                        np.std(returns),           # Realized volatility
                        np.mean(np.abs(returns)),  # Mean absolute returns
                        np.max(np.abs(returns)),   # Max absolute returns
                        np.var(returns),           # Return variance
                        stats.skew(returns),       # Return skewness
                        stats.kurtosis(returns),   # Return kurtosis
                    ])
                    
                    # GARCH-like features
                    squared_returns = returns ** 2
                    features.extend([
                        np.mean(squared_returns),  # Mean squared returns
                        np.std(squared_returns),   # Std of squared returns
                    ])
                else:
                    features.extend([0.0] * 8)
            else:
                features.extend([0.0] * 8)
        
        return features
    
    def _generate_momentum_features(self, series: np.ndarray) -> List[float]:
        """Generate momentum-related features."""
        features = []
        
        for window in self.config['momentum_windows']:
            if len(series) >= window + 1:
                # Rate of change
                roc = (series[-1] - series[-window-1]) / series[-window-1] if series[-window-1] != 0 else 0.0
                features.append(roc)
                
                # Acceleration (second derivative)
                if len(series) >= window * 2 + 1:
                    roc_prev = (series[-window-1] - series[-window*2-1]) / series[-window*2-1] if series[-window*2-1] != 0 else 0.0
                    acceleration = roc - roc_prev
                    features.append(acceleration)
                else:
                    features.append(0.0)
                
                # Momentum strength
                momentum_strength = np.sum(np.diff(series[-window:]) > 0) / (window - 1) if window > 1 else 0.0
                features.append(momentum_strength)
                
            else:
                features.extend([0.0] * 3)
        
        return features
    
    def _generate_statistical_features(self, series: np.ndarray) -> List[float]:
        """Generate statistical features."""
        features = []
        
        if len(series) > 0:
            # Z-score of current value
            z_score = (series[-1] - np.mean(series)) / np.std(series) if np.std(series) > 0 else 0.0
            features.append(z_score)
            
            # Percentile rank
            percentile_rank = np.sum(series < series[-1]) / len(series)
            features.append(percentile_rank)
            
            # Outlier detection (using IQR)
            q1, q3 = np.percentile(series, [25, 75])
            iqr = q3 - q1
            is_outlier = 1.0 if (series[-1] < q1 - 1.5 * iqr or series[-1] > q3 + 1.5 * iqr) else 0.0
            features.append(is_outlier)
            
            # Coefficient of variation
            cv = np.std(series) / np.mean(series) if np.mean(series) != 0 else 0.0
            features.append(cv)
            
            # Entropy (discretized)
            if len(series) > 10:
                hist, _ = np.histogram(series, bins=min(10, len(series)//10))
                hist = hist[hist > 0]  # Remove zero bins
                if len(hist) > 0:
                    entropy = -np.sum(hist * np.log(hist / np.sum(hist)))
                    features.append(entropy)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        else:
            features.extend([0.0] * 5)
        
        return features
    
    def _generate_cyclical_features(self, series: np.ndarray) -> List[float]:
        """Generate cyclical pattern features."""
        features = []
        
        for lag in self.config['autocorr_lags']:
            if len(series) >= lag + 1:
                # Autocorrelation
                autocorr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                features.append(autocorr if not np.isnan(autocorr) else 0.0)
                
                # Partial autocorrelation approximation
                if len(series) >= lag * 2:
                    # Simple partial autocorrelation approximation
                    residuals = series[lag:] - np.mean(series[lag:])
                    residuals_lag = series[:-lag] - np.mean(series[:-lag])
                    pacf = np.corrcoef(residuals, residuals_lag)[0, 1]
                    features.append(pacf if not np.isnan(pacf) else 0.0)
                else:
                    features.append(0.0)
            else:
                features.extend([0.0] * 2)
        
        # Periodicity detection
        if len(series) >= 20:
            # Find peaks in autocorrelation
            autocorr_full = [np.corrcoef(series[:-i], series[i:])[0, 1] for i in range(1, min(20, len(series)//2))]
            autocorr_full = [x if not np.isnan(x) else 0.0 for x in autocorr_full]
            
            if len(autocorr_full) > 3:
                peaks, _ = find_peaks(autocorr_full, height=0.1)
                if len(peaks) > 0:
                    dominant_period = peaks[0] + 1
                    features.append(dominant_period)
                    features.append(autocorr_full[peaks[0]])  # Peak autocorrelation
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _generate_fft_features(self, series: np.ndarray) -> List[float]:
        """Generate FFT-based features."""
        features = []
        
        if len(series) >= 16:  # Minimum length for meaningful FFT
            # Apply FFT
            fft_values = np.fft.fft(series)
            fft_magnitude = np.abs(fft_values)
            
            # Keep only positive frequencies (first half)
            n = len(fft_magnitude) // 2
            fft_magnitude = fft_magnitude[:n]
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(fft_magnitude[1:]) + 1  # Skip DC component
            dominant_freq = dominant_freq_idx / len(series)
            features.append(dominant_freq)
            
            # Spectral power
            total_power = np.sum(fft_magnitude ** 2)
            features.append(total_power)
            
            # Spectral centroid
            if total_power > 0:
                spectral_centroid = np.sum(np.arange(len(fft_magnitude)) * fft_magnitude ** 2) / total_power
                features.append(spectral_centroid)
            else:
                features.append(0.0)
            
            # Spectral bandwidth
            if total_power > 0 and len(fft_magnitude) > 1:
                freq_axis = np.arange(len(fft_magnitude))
                spectral_bandwidth = np.sqrt(np.sum((freq_axis - spectral_centroid) ** 2 * fft_magnitude ** 2) / total_power)
                features.append(spectral_bandwidth)
            else:
                features.append(0.0)
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            if total_power > 0:
                cumulative_power = np.cumsum(fft_magnitude ** 2)
                rolloff_threshold = 0.85 * total_power
                rolloff_idx = np.where(cumulative_power >= rolloff_threshold)[0]
                if len(rolloff_idx) > 0:
                    spectral_rolloff = rolloff_idx[0] / len(series)
                    features.append(spectral_rolloff)
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
            
        else:
            features.extend([0.0] * 5)
        
        return features
    
    def generate_dynamic_features(self, dynamic_series: np.ndarray, num_lags: int = 7) -> np.ndarray:
        """
        Generate features for dynamic prediction (when we have historical + predicted values).
        
        Args:
            dynamic_series: Series containing historical + predicted values
            num_lags: Number of basic lagged features
            
        Returns:
            Feature array for next prediction
        """
        # Get the last num_lags values
        last_values = dynamic_series[-num_lags:]
        
        # Generate advanced features for the entire dynamic series
        advanced_features = []
        
        # Rolling features
        rolling_features = self._generate_rolling_features(dynamic_series)
        advanced_features.extend(rolling_features)
        
        # Seasonality features
        seasonality_features = self._generate_seasonality_features(dynamic_series)
        advanced_features.extend(seasonality_features)
        
        # Trend features
        trend_features = self._generate_trend_features(dynamic_series)
        advanced_features.extend(trend_features)
        
        # Volatility features
        volatility_features = self._generate_volatility_features(dynamic_series)
        advanced_features.extend(volatility_features)
        
        # Momentum features
        momentum_features = self._generate_momentum_features(dynamic_series)
        advanced_features.extend(momentum_features)
        
        # Statistical features
        statistical_features = self._generate_statistical_features(dynamic_series)
        advanced_features.extend(statistical_features)
        
        # Cyclical features
        cyclical_features = self._generate_cyclical_features(dynamic_series)
        advanced_features.extend(cyclical_features)
        
        # FFT features
        if self.config.get('enable_fft_features', True):
            fft_features = self._generate_fft_features(dynamic_series)
            advanced_features.extend(fft_features)
        
        # Combine all features
        all_features = list(last_values) + advanced_features
        return np.array(all_features).reshape(1, -1)
    
    def get_feature_names(self, num_lags: int = 7) -> List[str]:
        """
        Get names of all generated features.
        
        Args:
            num_lags: Number of basic lagged features
            
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Basic lagged features
        for i in range(num_lags):
            feature_names.append(f"lag_{i+1}")
        
        # Rolling features
        for window in self.config['rolling_windows']:
            feature_names.extend([
                f"rolling_mean_{window}", f"rolling_std_{window}", f"rolling_min_{window}",
                f"rolling_max_{window}", f"rolling_range_{window}", f"rolling_median_{window}",
                f"rolling_skew_{window}", f"rolling_kurt_{window}"
            ])
            for q in self.config['percentiles']:
                feature_names.append(f"rolling_p{q}_{window}")
            feature_names.extend([
                f"rolling_var_{window}", f"rolling_mad_{window}", f"rolling_up_ratio_{window}"
            ])
        
        # Seasonality features
        for period in self.config['seasonal_periods']:
            feature_names.extend([
                f"seasonal_mean_{period}", f"seasonal_std_{period}", f"seasonal_max_{period}",
                f"seasonal_min_{period}", f"seasonal_range_{period}", f"seasonal_autocorr_{period}"
            ])
        
        # Trend features
        for window in self.config['trend_windows']:
            feature_names.extend([
                f"trend_slope_{window}", f"trend_intercept_{window}", f"trend_quad_a_{window}",
                f"trend_quad_b_{window}", f"trend_r2_{window}", f"trend_direction_{window}"
            ])
        
        # Volatility features
        for window in self.config['volatility_windows']:
            feature_names.extend([
                f"vol_std_{window}", f"vol_mean_abs_{window}", f"vol_max_abs_{window}",
                f"vol_var_{window}", f"vol_skew_{window}", f"vol_kurt_{window}",
                f"vol_mean_sq_{window}", f"vol_std_sq_{window}"
            ])
        
        # Momentum features
        for window in self.config['momentum_windows']:
            feature_names.extend([
                f"momentum_roc_{window}", f"momentum_accel_{window}", f"momentum_strength_{window}"
            ])
        
        # Statistical features
        feature_names.extend([
            "z_score", "percentile_rank", "is_outlier", "coef_variation", "entropy"
        ])
        
        # Cyclical features
        for lag in self.config['autocorr_lags']:
            feature_names.extend([f"autocorr_{lag}", f"pacf_{lag}"])
        feature_names.extend(["dominant_period", "peak_autocorr"])
        
        # FFT features
        if self.config.get('enable_fft_features', True):
            feature_names.extend([
                "fft_dominant_freq", "fft_total_power", "fft_spectral_centroid",
                "fft_spectral_bandwidth", "fft_spectral_rolloff"
            ])
        
        return feature_names


def create_advanced_features_config(series_length: int) -> Dict[str, Any]:
    """
    Create adaptive configuration for advanced temporal features.
    
    Args:
        series_length: Length of the time series
        
    Returns:
        Configuration dictionary
    """
    # Adaptive configuration based on series length
    if series_length < 100:
        # Small series: minimal features
        return {
            'rolling_windows': [3, 7],
            'seasonal_periods': [7],
            'trend_windows': [7, 14],
            'volatility_windows': [5, 10],
            'momentum_windows': [1, 3, 7],
            'percentiles': [25, 50, 75],
            'autocorr_lags': [1, 7],
            'enable_fft_features': False,
            'max_features': 50
        }
    elif series_length < 1000:
        # Medium series: moderate features
        return {
            'rolling_windows': [3, 7, 14, 30],
            'seasonal_periods': [7, 30],
            'trend_windows': [7, 14, 30],
            'volatility_windows': [5, 10, 20],
            'momentum_windows': [1, 3, 7, 14],
            'percentiles': [5, 25, 50, 75, 95],
            'autocorr_lags': [1, 7, 14],
            'enable_fft_features': True,
            'max_features': 80
        }
    else:
        # Large series: full features
        return {
            'rolling_windows': [3, 7, 14, 30, 90],
            'seasonal_periods': [7, 30, 365],
            'trend_windows': [7, 14, 30, 90],
            'volatility_windows': [5, 10, 20],
            'momentum_windows': [1, 3, 7, 14],
            'percentiles': [5, 25, 50, 75, 95],
            'autocorr_lags': [1, 7, 14, 30],
            'enable_fft_features': True,
            'max_features': 100
        }

