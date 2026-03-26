"""
EEMD Service

This service handles EEMD (Ensemble Empirical Mode Decomposition) operations
for time series data with automatic parameter optimization.
Implements variable-agnostic interfaces for complete independence from
specific meteorological variables.
"""

import logging
import os
import emd
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from itertools import combinations
from pathlib import Path

from src.core.interfaces.prediction_strategy import EEMDResult, PredictionConfig
from src.core.interfaces.variable_agnostic_interfaces import (
    IVariableAgnosticProcessor, ProcessingConfig, ProcessingResult
)
from src.core.exceptions.processing_exceptions import DecompositionError, ValidationError

logger = logging.getLogger(__name__)


class EEMDService(IVariableAgnosticProcessor):
    """
    Service for performing EEMD decomposition with automatic parameter optimization.

    This service implements the Single Responsibility Principle by focusing solely
    on EEMD decomposition operations.
    """

    def __init__(self, config: PredictionConfig):
        """
        Initialize the EEMD service.
        
        Args:
            config: Prediction configuration containing EEMD parameters
        """
        self.config = config
        self.logger = logger

        # Validate configuration
        self._validate_config()

        self.logger.info(f"Initialized EEMDService for {config.variable_type}")

    def _validate_config(self) -> None:
        """Validate EEMD configuration parameters."""
        if not self.config.eemd_sd_thresh_values:
            raise ValueError("EEMD sd_thresh values must be provided")

        if self.config.eemd_nensembles <= 0:
            raise ValueError("EEMD nensembles must be positive")

        if self.config.eemd_noise_factor <= 0:
            raise ValueError("EEMD noise factor must be positive")

    def validate_time_series(self, time_series: pd.Series) -> bool:
        """
        Validate that the time series is suitable for EEMD decomposition.
        
        Args:
            time_series: Time series to validate
            
        Returns:
            True if time series is valid for decomposition
            
        Raises:
            ValidationError: If validation fails
        """
        from src.core.validators.data_validator import DataValidator
        validator = DataValidator()
        return validator.validate_time_series(time_series)

    def decompose_time_series(self, time_series: pd.Series, adaptive_config=None) -> EEMDResult:
        """
        Perform EEMD decomposition with automatic parameter optimization and memory optimization.

        Args:
            time_series: Input time series to decompose
            adaptive_config: Optional adaptive configuration for memory optimization

        Returns:
            EEMDResult containing IMFs, correlations, and metadata

        Raises:
            DecompositionError: If decomposition fails
        """
        try:
            # Validate input
            if not self.validate_time_series(time_series):
                raise ValidationError("Time series validation failed")
            
            self.logger.info(f"Starting EEMD decomposition for series with {len(time_series)} points")

            # Apply downsampling if configured
            if adaptive_config and adaptive_config.use_downsampling:
                original_length = len(time_series)
                
                # Check if index is datetime-like
                if not isinstance(time_series.index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
                    self.logger.warning(f"Index is not datetime-like, skipping downsampling. Index type: {type(time_series.index)}")
                else:
                    time_series = time_series.resample(adaptive_config.downsample_freq).mean()
                    self.logger.info(f"Applied downsampling: {original_length} -> {len(time_series)} points")

            # Find optimal parameters using adaptive optimization
            best_imfs, best_stats = self._find_optimal_parameters(time_series, adaptive_config)

            if best_imfs is None:
                raise DecompositionError("Failed to find optimal EEMD parameters")

            # Calculate correlations and variance explained
            correlations = self._calculate_correlations(best_imfs, time_series)
            variance_explained = self._calculate_variance_explained(best_imfs, time_series)

            # Calculate comprehensive quality metrics
            decomposition_quality = self._calculate_comprehensive_quality(
                best_imfs, time_series, best_stats['best_sd_thresh']
            )

            self.logger.info(f"EEMD decomposition completed successfully with {best_imfs.shape[1]} IMFs")

            # Filter low-quality IMFs
            filtered_imfs, filtered_stats = self._filter_low_quality_imfs(
                best_imfs, decomposition_quality, variance_explained
            )
            
            # Analyze meteorological patterns
            meteorological_patterns = self._analyze_meteorological_patterns(filtered_imfs, time_series)
            filtered_stats['meteorological_patterns'] = meteorological_patterns
            
            # Classify IMFs for modeling
            imf_classifications = self.classify_imfs_for_modeling(filtered_imfs, meteorological_patterns, variance_explained, time_series)
            filtered_stats['imf_classifications'] = imf_classifications
            
            return EEMDResult(
                imfs=filtered_imfs,
                correlations=correlations[:filtered_imfs.shape[1]],
                variance_explained=variance_explained.head(filtered_imfs.shape[1]),
                best_sd_thresh=best_stats['best_sd_thresh'],
                orthogonality_score=filtered_stats['orthogonality_score'],
                decomposition_quality=filtered_stats
            )

        except Exception as e:
            self.logger.error(f"EEMD decomposition failed: {e}")
            raise DecompositionError(f"EEMD decomposition failed: {e}")
    
    def _find_optimal_parameters(self, time_series: pd.Series, adaptive_config=None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Find optimal EEMD parameters using adaptive optimization with early stopping.
        
        Args:
            time_series: Input time series
            adaptive_config: Optional adaptive configuration for memory optimization
            
        Returns:
            Tuple of (best_imfs, best_stats)
        """
        best_score = float('inf')
        best_imfs = None
        best_stats = {}
        
        # Calculate adaptive parameters based on series characteristics
        series_std = np.std(time_series.dropna())
        series_length = len(time_series.dropna())
        
        # Simple optimization without early stopping
        
        # Use adaptive config if provided for memory optimization
        if adaptive_config:
            nensembles = adaptive_config.eemd_nensembles
            sd_thresh_values = adaptive_config.eemd_sd_thresh_values
            noise_factor = 0.1  # Fixed for memory optimization
        else:
            # Enhanced adaptive noise factor based on series characteristics
            if series_length < 1000:
                noise_factor = 0.03  # Very low noise for short series
            elif series_length < 3000:
                noise_factor = 0.05  # Low noise for medium series
            elif series_length < 10000:
                noise_factor = 0.1   # Standard noise
            else:
                noise_factor = 0.15  # Higher noise for very long series
            
            # Enhanced adaptive ensemble size based on series complexity
            if series_std < 1.5:
                nensembles = 8   # Very simple series
            elif series_std < 3.0:
                nensembles = 15  # Simple series
            elif series_std < 5.0:
                nensembles = 25  # Moderate complexity
            else:
                nensembles = 35  # Complex series
            
            # Enhanced adaptive sd_thresh range based on series characteristics
            if series_std > 6.0:
                sd_thresh_values = [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25]  # More options for very complex series
            elif series_std > 4.0:
                sd_thresh_values = [0.05, 0.08, 0.1, 0.15, 0.2, 0.25]  # More options for complex series
            else:
                sd_thresh_values = [0.05, 0.1, 0.15, 0.2]
        
        ensemble_noise = series_std * noise_factor
        
        self.logger.info(f"Memory-optimized adaptive parameters:")
        self.logger.info(f"  - Series length: {series_length}, std: {series_std:.4f}")
        self.logger.info(f"  - Noise factor: {noise_factor}, ensemble noise: {ensemble_noise:.4f}")
        self.logger.info(f"  - Ensembles: {nensembles}, sd_thresh range: {sd_thresh_values}")
        
        # First pass: test all parameters with early stopping
        for i, thresh in enumerate(sd_thresh_values):
            self.logger.debug(f"Testing sd_thresh={thresh:.3f} (iteration {i+1}/{len(sd_thresh_values)})")
            
            try:
                # Perform EEMD decomposition with adaptive parameters
                imfs = emd.sift.ensemble_sift(
                    np.array(time_series.dropna()),
                    nensembles=nensembles,
                    nprocesses=os.cpu_count(),
                    ensemble_noise=ensemble_noise,
                    imf_opts={'sd_thresh': thresh}
                )
                
                # Calculate comprehensive quality metrics
                quality_metrics = self._calculate_comprehensive_quality(imfs, time_series, thresh)
                
                self.logger.debug(f"sd_thresh={thresh:.3f} - Quality score: {quality_metrics['composite_score']:.4f}")
                
                if quality_metrics['composite_score'] < best_score:
                    best_score = quality_metrics['composite_score']
                    best_imfs = imfs
                    best_stats = {
                        'best_sd_thresh': thresh,
                        'nensembles': nensembles,
                        'noise_factor': noise_factor,
                        'ensemble_noise': ensemble_noise,
                        **quality_metrics
                    }
                    break
                    
            except Exception as e:
                self.logger.warning(f"Error with sd_thresh={thresh}: {e}")
                continue
        
        # Fine-tuning: if quality is poor, try additional parameters (only for non-adaptive config)
        if (best_imfs is not None and 
            best_stats.get('composite_score', float('inf')) > 0.5 and 
            adaptive_config is None and
            not early_stopping.history):  # Only if early stopping didn't trigger
            self.logger.info("Quality below threshold, performing fine-tuning...")
            
            # Try with increased ensemble size
            fine_tuned_nensembles = min(nensembles * 2, 50)
            fine_tuned_noise = ensemble_noise * 0.8  # Reduce noise slightly
            
            for thresh in [best_stats['best_sd_thresh'] * 0.8, best_stats['best_sd_thresh'] * 1.2]:
                if 0.01 <= thresh <= 0.3:  # Keep within reasonable bounds
                    try:
                        imfs = emd.sift.ensemble_sift(
                            np.array(time_series.dropna()),
                            nensembles=fine_tuned_nensembles,
                            nprocesses=os.cpu_count(),
                            ensemble_noise=fine_tuned_noise,
                            imf_opts={'sd_thresh': thresh}
                        )
                        
                        quality_metrics = self._calculate_comprehensive_quality(imfs, time_series, thresh)
                        
                        if quality_metrics['composite_score'] < best_score:
                            best_score = quality_metrics['composite_score']
                            best_imfs = imfs
                            best_stats.update({
                                'best_sd_thresh': thresh,
                                'nensembles': fine_tuned_nensembles,
                                'noise_factor': fine_tuned_noise / series_std,
                                'ensemble_noise': fine_tuned_noise,
                                **quality_metrics
                            })
                            self.logger.info(f"Fine-tuning improved quality: {best_score:.4f}")
                            
                    except Exception as e:
                        self.logger.debug(f"Fine-tuning failed for thresh={thresh:.3f}: {e}")
                        continue
        
        if best_imfs is None:
            self.logger.error("No successful EEMD decomposition found")
            return None, {}
        
        self.logger.info(f"Best configuration:")
        self.logger.info(f"  - sd_thresh: {best_stats['best_sd_thresh']:.3f}")
        self.logger.info(f"  - Ensembles: {best_stats['nensembles']}")
        self.logger.info(f"  - Quality score: {best_stats['composite_score']:.4f}")
        
        # Optimization completed successfully
        
        return best_imfs, best_stats
    
    def _filter_low_quality_imfs(self, imfs: np.ndarray, quality_metrics: Dict[str, Any], 
                                variance_explained: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Filter out low-quality IMFs based on multiple criteria.
        
        Args:
            imfs: Original IMF array
            quality_metrics: Quality metrics dictionary
            variance_explained: Variance explained DataFrame
            
        Returns:
            Tuple of (filtered_imfs, updated_quality_metrics)
        """
        if imfs.shape[1] <= 3:
            # Keep at least 3 IMFs
            return imfs, quality_metrics
        
        # Criteria for filtering IMFs
        keep_indices = []
        imf_quality_scores = quality_metrics.get('imf_quality_scores', [])
        
        for i in range(imfs.shape[1]):
            keep_imf = True
            
            # 1. Variance threshold (keep IMFs with > 0.5% variance)
            variance_ratio = variance_explained.iloc[i]['explained_ratio']
            if variance_ratio < 0.005:
                keep_imf = False
                self.logger.debug(f"Filtering IMF {i+1}: Low variance ({variance_ratio:.4f})")
            
            # 2. Quality score threshold
            if i < len(imf_quality_scores) and imf_quality_scores[i] < 0.3:
                keep_imf = False
                self.logger.debug(f"Filtering IMF {i+1}: Low quality score ({imf_quality_scores[i]:.4f})")
            
            if keep_imf:
                keep_indices.append(i)
        
        # Ensure we keep at least 3 IMFs
        if len(keep_indices) < 3:
            keep_indices = list(range(min(3, imfs.shape[1])))
        
        filtered_imfs = imfs[:, keep_indices]
        
        # Update quality metrics
        updated_metrics = quality_metrics.copy()
        updated_metrics['num_imfs'] = filtered_imfs.shape[1]
        updated_metrics['orthogonality_score'] = self._calculate_orthogonality(filtered_imfs)
        
        # Recalculate variance explained for filtered IMFs
        total_var = np.var(quality_metrics.get('original_series', np.zeros(100)))
        filtered_variance_scores = []
        for i in keep_indices:
            imf_var = np.var(imfs[:, i])
            filtered_variance_scores.append(imf_var / total_var if total_var > 0 else 0)
        
        updated_metrics['top3_variance'] = sum(sorted(filtered_variance_scores, reverse=True)[:3])
        
        self.logger.info(f"Filtered {imfs.shape[1] - filtered_imfs.shape[1]} low-quality IMFs")
        self.logger.info(f"Kept {filtered_imfs.shape[1]} high-quality IMFs")
        
        return filtered_imfs, updated_metrics
    
    def _analyze_meteorological_patterns(self, imfs: np.ndarray, time_series: pd.Series) -> Dict[str, Any]:
        """
        Analyze comprehensive meteorological-specific patterns in IMFs.
        
        Args:
            imfs: IMF array
            time_series: Original time series
            
        Returns:
            Dictionary with meteorological pattern analysis
        """
        original_clean = time_series.dropna()
        patterns = {}
        
        # 1. Annual seasonality detection (365 days)
        annual_seasonality = []
        for i in range(min(imfs.shape[1], 5)):
            imf_series = imfs[:len(original_clean), i]
            if len(imf_series) > 365:
                # Calculate autocorrelation at lag 365
                autocorr = np.corrcoef(imf_series[:-365], imf_series[365:])[0, 1]
                annual_seasonality.append(abs(autocorr))
            else:
                annual_seasonality.append(0)
        
        patterns['annual_seasonality'] = annual_seasonality
        patterns['strongest_annual_imf'] = np.argmax(annual_seasonality) if annual_seasonality else -1
        
        # 2. Monthly patterns (30 days)
        monthly_patterns = []
        for i in range(min(imfs.shape[1], 5)):
            imf_series = imfs[:len(original_clean), i]
            if len(imf_series) > 30:
                autocorr = np.corrcoef(imf_series[:-30], imf_series[30:])[0, 1]
                monthly_patterns.append(abs(autocorr))
            else:
                monthly_patterns.append(0)
        
        patterns['monthly_patterns'] = monthly_patterns
        patterns['strongest_monthly_imf'] = np.argmax(monthly_patterns) if monthly_patterns else -1
        
        # 3. Weekly patterns (7 days)
        weekly_patterns = []
        for i in range(min(imfs.shape[1], 5)):
            imf_series = imfs[:len(original_clean), i]
            if len(imf_series) > 7:
                autocorr = np.corrcoef(imf_series[:-7], imf_series[7:])[0, 1]
                weekly_patterns.append(abs(autocorr))
            else:
                weekly_patterns.append(0)
        
        patterns['weekly_patterns'] = weekly_patterns
        patterns['strongest_weekly_imf'] = np.argmax(weekly_patterns) if weekly_patterns else -1
        
        # 4. Enhanced Trend analysis (STEP 2.1: More comprehensive trend detection)
        trend_scores = []
        quadratic_trend_scores = []
        trend_consistency_scores = []
        
        for i in range(imfs.shape[1]):
            imf_series = imfs[:len(original_clean), i]
            
            # Linear trend strength
            x = np.arange(len(imf_series))
            slope, intercept = np.polyfit(x, imf_series, 1)
            trend_strength = abs(slope) / np.std(imf_series) if np.std(imf_series) > 0 else 0
            trend_scores.append(trend_strength)
            
            # Quadratic trend (captures non-linear trends)
            if len(imf_series) > 10:
                coeffs = np.polyfit(x, imf_series, 2)
                quadratic_trend = abs(coeffs[0]) / np.std(imf_series) if np.std(imf_series) > 0 else 0
                quadratic_trend_scores.append(quadratic_trend)
            else:
                quadratic_trend_scores.append(0)
            
            # Trend consistency (how well the trend fits)
            if len(imf_series) > 10:
                trend_line = slope * x + intercept
                residuals = imf_series - trend_line
                trend_consistency = 1.0 - (np.std(residuals) / np.std(imf_series)) if np.std(imf_series) > 0 else 0
                trend_consistency = max(0, trend_consistency)  # Ensure non-negative
                trend_consistency_scores.append(trend_consistency)
            else:
                trend_consistency_scores.append(0)
        
        patterns['trend_scores'] = trend_scores
        patterns['quadratic_trend_scores'] = quadratic_trend_scores
        patterns['trend_consistency_scores'] = trend_consistency_scores
        patterns['strongest_trend_imf'] = np.argmax(trend_scores) if trend_scores else -1
        patterns['strongest_quadratic_trend_imf'] = np.argmax(quadratic_trend_scores) if quadratic_trend_scores else -1
        
        # 5. Extreme events detection (heat waves, cold spells)
        extreme_events = self._detect_extreme_events(imfs, original_clean)
        patterns.update(extreme_events)
        
        # 6. Long-term climate trends
        climate_trends = self._analyze_climate_trends(imfs, original_clean)
        patterns.update(climate_trends)
        
        # 7. Noise identification
        noise_scores = []
        for i in range(imfs.shape[1]):
            imf_series = imfs[:len(original_clean), i]
            
            # High frequency components are likely noise
            # Calculate the ratio of high-frequency power to total power
            fft = np.fft.fft(imf_series)
            power = np.abs(fft) ** 2
            high_freq_power = np.sum(power[len(power)//4:])  # Upper 75% of frequencies
            total_power = np.sum(power)
            noise_ratio = high_freq_power / total_power if total_power > 0 else 0
            noise_scores.append(noise_ratio)
        
        patterns['noise_scores'] = noise_scores
        patterns['noisiest_imf'] = np.argmax(noise_scores) if noise_scores else -1
        
        return patterns
    
    def _detect_extreme_events(self, imfs: np.ndarray, time_series: pd.Series) -> Dict[str, Any]:
        """
        Detect extreme weather events in IMFs.
        
        Args:
            imfs: IMF array
            time_series: Original time series
            
        Returns:
            Dictionary with extreme event analysis
        """
        extreme_events = {}
        
        # Calculate percentiles for extreme event detection
        series_95th = np.percentile(time_series, 95)
        series_5th = np.percentile(time_series, 5)
        
        # Analyze each IMF for extreme event patterns
        heat_wave_scores = []
        cold_spell_scores = []
        
        for i in range(imfs.shape[1]):
            imf_series = imfs[:, i]
            
            # Heat wave detection (high frequency, high amplitude)
            high_temp_events = np.sum(imf_series > series_95th)
            heat_wave_score = high_temp_events / len(imf_series) if len(imf_series) > 0 else 0
            heat_wave_scores.append(heat_wave_score)
            
            # Cold spell detection (low frequency, low amplitude)
            low_temp_events = np.sum(imf_series < series_5th)
            cold_spell_score = low_temp_events / len(imf_series) if len(imf_series) > 0 else 0
            cold_spell_scores.append(cold_spell_score)
        
        extreme_events['heat_wave_scores'] = heat_wave_scores
        extreme_events['cold_spell_scores'] = cold_spell_scores
        extreme_events['strongest_heat_wave_imf'] = np.argmax(heat_wave_scores) if heat_wave_scores else -1
        extreme_events['strongest_cold_spell_imf'] = np.argmax(cold_spell_scores) if cold_spell_scores else -1
        
        return extreme_events
    
    def _analyze_climate_trends(self, imfs: np.ndarray, time_series: pd.Series) -> Dict[str, Any]:
        """
        Analyze long-term climate trends and complex patterns in IMFs.
        
        STEP 2.1: Enhanced climate trend analysis for flexible SARIMAX criteria
        
        Args:
            imfs: IMF array
            time_series: Original time series
            
        Returns:
            Dictionary with climate trend analysis
        """
        climate_trends = {}
        
        # Long-term trend analysis (using longer periods)
        long_term_trends = []
        decadal_trends = []
        seasonal_trends = []
        cyclical_patterns = []
        
        for i in range(imfs.shape[1]):
            imf_series = imfs[:, i]
            
            # Long-term trend (using entire series)
            x = np.arange(len(imf_series))
            slope, intercept = np.polyfit(x, imf_series, 1)
            long_term_trend = slope * len(imf_series) / np.std(imf_series) if np.std(imf_series) > 0 else 0
            long_term_trends.append(abs(long_term_trend))
            
            # Decadal trend (using 10-year periods if available)
            if len(imf_series) > 3650:  # More than 10 years of daily data
                decade_length = min(3650, len(imf_series) // 2)
                recent_decade = imf_series[-decade_length:]
                x_decade = np.arange(len(recent_decade))
                slope_decade, _ = np.polyfit(x_decade, recent_decade, 1)
                decadal_trend = slope_decade * decade_length / np.std(recent_decade) if np.std(recent_decade) > 0 else 0
                decadal_trends.append(abs(decadal_trend))
            else:
                decadal_trends.append(0)
            
            # STEP 2.1: Seasonal trend analysis (NEW)
            # Detect if there's a trend in the seasonal patterns
            if len(imf_series) > 730:  # At least 2 years
                # Split into seasonal periods and analyze trend within each
                seasonal_periods = []
                for year in range(0, len(imf_series) - 365, 365):
                    period = imf_series[year:year + 365]
                    if len(period) == 365:
                        seasonal_periods.append(np.mean(period))
                
                if len(seasonal_periods) > 2:
                    x_seasonal = np.arange(len(seasonal_periods))
                    slope_seasonal, _ = np.polyfit(x_seasonal, seasonal_periods, 1)
                    seasonal_trend = abs(slope_seasonal) / np.std(seasonal_periods) if np.std(seasonal_periods) > 0 else 0
                    seasonal_trends.append(seasonal_trend)
                else:
                    seasonal_trends.append(0)
            else:
                seasonal_trends.append(0)
            
            # STEP 2.1: Cyclical pattern detection (NEW)
            # Detect cyclical patterns that are not strictly seasonal
            if len(imf_series) > 100:
                # Use autocorrelation to detect cyclical patterns
                autocorr = np.correlate(imf_series, imf_series, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Look for peaks in autocorrelation beyond lag 1
                peaks = []
                for lag in range(10, min(100, len(autocorr))):
                    if (autocorr[lag] > autocorr[lag-1] and 
                        autocorr[lag] > autocorr[lag+1] and 
                        autocorr[lag] > 0.1):  # Significant correlation
                        peaks.append(lag)
                
                if peaks:
                    # Calculate average cyclical strength
                    cyclical_strength = np.mean([autocorr[peak] for peak in peaks])
                    cyclical_patterns.append(cyclical_strength)
                else:
                    cyclical_patterns.append(0)
            else:
                cyclical_patterns.append(0)
        
        climate_trends['long_term_trends'] = long_term_trends
        climate_trends['decadal_trends'] = decadal_trends
        climate_trends['seasonal_trends'] = seasonal_trends
        climate_trends['cyclical_patterns'] = cyclical_patterns
        climate_trends['strongest_long_term_trend_imf'] = np.argmax(long_term_trends) if long_term_trends else -1
        climate_trends['strongest_decadal_trend_imf'] = np.argmax(decadal_trends) if decadal_trends else -1
        climate_trends['strongest_seasonal_trend_imf'] = np.argmax(seasonal_trends) if seasonal_trends else -1
        climate_trends['strongest_cyclical_imf'] = np.argmax(cyclical_patterns) if cyclical_patterns else -1
        
        return climate_trends
    
    def classify_imfs_for_modeling(self, imfs: np.ndarray, patterns: Dict[str, Any], 
                                  variance_explained: pd.DataFrame, original_series: pd.Series) -> Dict[str, List[int]]:
        """
        Classify IMFs using COMPOSITE SCORING SYSTEM.
        
        NEW LOGIC:
        1. Calculate composite score for each IMF
        2. Assign EXACTLY ONE IMF to SARIMAX (best score)
        3. Distribute remaining IMFs intelligently
        
        Args:
            imfs: IMF array
            patterns: Meteorological patterns dictionary
            variance_explained: Variance explained DataFrame
            original_series: Original time series for correlation analysis
            
        Returns:
            Dictionary with IMF classifications for different modeling approaches
        """
        self.logger.info("🎯 Starting COMPOSITE SCORING CLASSIFICATION...")
        
        # Initialize classifications
        classifications = {
            'sarimax_imfs': [],           # EXACTLY ONE - Best composite score
            'svr_imfs': [],               # High frequency components
            'extrapolation_imfs': [],     # Simple extrapolation
            'noise_imfs': []              # Discard or simple methods
        }
        
        # Step 1: Calculate composite scores for all IMFs
        imf_scores = []
        imf_properties_list = []
        
        for i in range(imfs.shape[1]):
            imf_series = imfs[:, i]
            imf_properties = self._analyze_imf_properties(imf_series, original_series.values, i)
            
            # Calculate SARIMAX-specific score based on SEASONALITY
            sarimax_score = self._calculate_sarimax_composite_score(
                imf_properties, 
                variance_explained.iloc[i],
                patterns,
                i
            )
            
            imf_scores.append(sarimax_score)
            imf_properties_list.append(imf_properties)
            
            self.logger.info(f"IMF {i+1} SARIMAX Score: {sarimax_score:.4f}")
            self.logger.debug(f"  - Annual Seasonality: {patterns.get('annual_seasonality', [0]*10)[i] if i < len(patterns.get('annual_seasonality', [])) else 0:.3f}")
            self.logger.debug(f"  - Monthly Seasonality: {patterns.get('monthly_patterns', [0]*10)[i] if i < len(patterns.get('monthly_patterns', [])) else 0:.3f}")
            self.logger.debug(f"  - Stability: {imf_properties['stability']:.3f}")
            self.logger.debug(f"  - Noise Level: {imf_properties['noise_level']:.3f}")
        
        # Step 2: Assign EXACTLY ONE IMF to SARIMAX (best score)
        best_sarimax_idx = int(np.argmax(imf_scores))  # Convert to regular Python int
        classifications['sarimax_imfs'].append(best_sarimax_idx)
        
        self.logger.info(f"[SARIMAX] Assignment: IMF {best_sarimax_idx + 1} (Best Seasonality Score: {imf_scores[best_sarimax_idx]:.4f})")
        
        # Step 3: Classify remaining IMFs
        for i in range(imfs.shape[1]):
            if i == best_sarimax_idx:
                continue  # Already assigned to SARIMAX
                
            imf_properties = imf_properties_list[i]
            imf_classification = self._classify_remaining_imf(imf_properties, i)
            classifications[imf_classification].append(i)
            
            self.logger.debug(f"IMF {i+1} → {imf_classification}")
        
        # Sort for consistency and convert to regular Python ints
        for key in classifications:
            classifications[key] = [int(idx) for idx in sorted(classifications[key])]
        
        # Step 4: Log final classification
        self.logger.info("[SUMMARY] FINAL CLASSIFICATION SUMMARY:")
        for classification_type, imf_indices in classifications.items():
            if imf_indices:
                imf_numbers = [i+1 for i in imf_indices]
                self.logger.info(f"  - {classification_type}: IMFs {imf_numbers}")
        
        return classifications
    
    def _calculate_sarimax_composite_score(self, imf_properties: Dict[str, float], variance_info: pd.Series, 
                                         patterns: Dict[str, Any], imf_idx: int) -> float:
        """
        Calculate SARIMAX-specific score based on COMPREHENSIVE pattern analysis.
        
        STEP 2.1: IMPLEMENT FLEXIBLE CRITERIA - Include trends and complex patterns
        
        SARIMAX can handle multiple pattern types, so we now consider:
        1. Seasonal patterns (annual, monthly, weekly)
        2. Trend patterns (long-term, decadal, linear trends)
        3. Complex patterns (climate trends, extreme events)
        4. Temporal stability and autocorrelation
        5. Predictability and model complexity
        
        Args:
            imf_properties: IMF properties dictionary
            variance_info: Variance explained information
            patterns: Meteorological patterns analysis
            imf_idx: IMF index
            
        Returns:
            SARIMAX-specific score (0-1, higher is better for SARIMAX)
        """
        # Extract SEASONALITY properties
        annual_seasonality = patterns.get('annual_seasonality', [0] * 10)[imf_idx] if imf_idx < len(patterns.get('annual_seasonality', [])) else 0
        monthly_seasonality = patterns.get('monthly_patterns', [0] * 10)[imf_idx] if imf_idx < len(patterns.get('monthly_patterns', [])) else 0
        weekly_seasonality = patterns.get('weekly_patterns', [0] * 10)[imf_idx] if imf_idx < len(patterns.get('weekly_patterns', [])) else 0
        
        # STEP 2.1: Extract TREND properties (NEW)
        trend_strength = patterns.get('trend_scores', [0] * 10)[imf_idx] if imf_idx < len(patterns.get('trend_scores', [])) else 0
        long_term_trend = patterns.get('long_term_trends', [0] * 10)[imf_idx] if imf_idx < len(patterns.get('long_term_trends', [])) else 0
        decadal_trend = patterns.get('decadal_trends', [0] * 10)[imf_idx] if imf_idx < len(patterns.get('decadal_trends', [])) else 0
        quadratic_trend = patterns.get('quadratic_trend_scores', [0] * 10)[imf_idx] if imf_idx < len(patterns.get('quadratic_trend_scores', [])) else 0
        trend_consistency = patterns.get('trend_consistency_scores', [0] * 10)[imf_idx] if imf_idx < len(patterns.get('trend_consistency_scores', [])) else 0
        seasonal_trend = patterns.get('seasonal_trends', [0] * 10)[imf_idx] if imf_idx < len(patterns.get('seasonal_trends', [])) else 0
        cyclical_pattern = patterns.get('cyclical_patterns', [0] * 10)[imf_idx] if imf_idx < len(patterns.get('cyclical_patterns', [])) else 0
        
        # Extract COMPLEX PATTERN properties (NEW)
        heat_wave_score = patterns.get('heat_wave_scores', [0] * 10)[imf_idx] if imf_idx < len(patterns.get('heat_wave_scores', [])) else 0
        cold_spell_score = patterns.get('cold_spell_scores', [0] * 10)[imf_idx] if imf_idx < len(patterns.get('cold_spell_scores', [])) else 0
        
        # Extract basic properties
        stability = imf_properties['stability']
        autocorrelation = imf_properties.get('autocorrelation', 0.0)
        noise_level = imf_properties['noise_level']
        dominant_freq = imf_properties['dominant_frequency']
        trend_strength_imf = imf_properties['trend_strength']
        
        # STEP 2.1: COMPREHENSIVE SCORING SYSTEM
        
        # 1. SEASONAL PATTERNS (still important for SARIMAX)
        annual_score = annual_seasonality  # Already 0-1
        monthly_score = monthly_seasonality  # Already 0-1
        weekly_score = weekly_seasonality * 0.5  # Reduced weight
        
        # 2. TREND PATTERNS (NEW - SARIMAX can model trends)
        # Normalize trend scores to 0-1 range
        trend_score = min(trend_strength / 0.1, 1.0) if trend_strength > 0 else 0  # Normalize to reasonable range
        long_term_trend_score = min(long_term_trend / 0.1, 1.0) if long_term_trend > 0 else 0
        decadal_trend_score = min(decadal_trend / 0.1, 1.0) if decadal_trend > 0 else 0
        quadratic_trend_score = min(quadratic_trend / 0.1, 1.0) if quadratic_trend > 0 else 0
        trend_consistency_score = trend_consistency  # Already 0-1
        seasonal_trend_score = min(seasonal_trend / 0.1, 1.0) if seasonal_trend > 0 else 0
        cyclical_pattern_score = min(cyclical_pattern / 0.5, 1.0) if cyclical_pattern > 0 else 0  # Different normalization for autocorr
        
        # 3. COMPLEX PATTERNS (NEW - SARIMAX can handle complex dynamics)
        # Extreme events can be modeled by SARIMAX with appropriate parameters
        extreme_event_score = max(heat_wave_score, cold_spell_score)  # Take the stronger extreme pattern
        
        # 4. TEMPORAL STABILITY (important for SARIMAX)
        stability_score = stability  # Already 0-1
        
        # 5. AUTOCORRELATION (important for SARIMAX)
        autocorr_score = min(autocorrelation * 2, 1.0)  # Scale to 0-1
        
        # 6. NOISE LEVEL (should be low for SARIMAX)
        noise_score = 1.0 - min(noise_level * 2, 1.0)  # Lower noise = better
        
        # 7. FREQUENCY ANALYSIS (SARIMAX prefers lower frequencies)
        if dominant_freq < 0.001:  # Very low frequency (trend)
            frequency_score = 1.0
        elif dominant_freq < 0.01:  # Low frequency (seasonal)
            frequency_score = 0.9
        elif dominant_freq < 0.05:  # Moderate frequency
            frequency_score = 0.6
        else:  # High frequency (noise)
            frequency_score = 0.2
        
        # 8. PREDICTABILITY (NEW - based on variance explained)
        variance_ratio = variance_info.get('explained_ratio', 0.0)
        predictability_score = min(variance_ratio * 2, 1.0)  # Scale to 0-1
        
        # STEP 2.1: FLEXIBLE WEIGHTS (balanced between seasonality and trends)
        weights = {
            # Seasonal patterns (still important)
            'annual_seasonality': 0.20,    # Reduced from 0.25
            'monthly_seasonality': 0.12,   # Reduced from 0.15
            'weekly_seasonality': 0.04,    # Reduced from 0.05
            
            # Trend patterns (NEW - equally important)
            'trend_strength': 0.12,        # Reduced from 0.15
            'long_term_trend': 0.08,       # Reduced from 0.10
            'decadal_trend': 0.04,         # Reduced from 0.05
            'quadratic_trend': 0.04,       # NEW
            'trend_consistency': 0.04,     # NEW
            'seasonal_trend': 0.04,        # NEW
            'cyclical_pattern': 0.04,      # NEW
            
            # Complex patterns (NEW)
            'extreme_events': 0.04,        # Reduced from 0.05
            
            # Stability and correlation
            'stability': 0.08,             # Reduced from 0.10
            'autocorrelation': 0.04,       # Reduced from 0.05
            
            # Quality indicators
            'noise': 0.02,                 # Same
            'frequency': 0.02,             # Same
            'predictability': 0.04         # Reduced from 0.06
        }
        
        # Calculate COMPREHENSIVE SARIMAX score
        sarimax_score = (
            # Seasonal components
            annual_score * weights['annual_seasonality'] +
            monthly_score * weights['monthly_seasonality'] +
            weekly_score * weights['weekly_seasonality'] +
            
            # Trend components (NEW)
            trend_score * weights['trend_strength'] +
            long_term_trend_score * weights['long_term_trend'] +
            decadal_trend_score * weights['decadal_trend'] +
            quadratic_trend_score * weights['quadratic_trend'] +
            trend_consistency_score * weights['trend_consistency'] +
            seasonal_trend_score * weights['seasonal_trend'] +
            cyclical_pattern_score * weights['cyclical_pattern'] +
            
            # Complex pattern components (NEW)
            extreme_event_score * weights['extreme_events'] +
            
            # Stability and correlation
            stability_score * weights['stability'] +
            autocorr_score * weights['autocorrelation'] +
            
            # Quality indicators
            noise_score * weights['noise'] +
            frequency_score * weights['frequency'] +
            predictability_score * weights['predictability']
        )
        
        return sarimax_score
    
    def _classify_remaining_imf(self, imf_properties: Dict[str, float], imf_idx: int) -> str:
        """
        Classify remaining IMFs after SARIMAX assignment.
        
        Args:
            imf_properties: IMF properties dictionary
            imf_idx: IMF index
            
        Returns:
            Classification string
        """
        # Extract properties
        correlation = abs(imf_properties['correlation_with_original'])
        complexity = imf_properties['complexity']
        noise_level = imf_properties['noise_level']
        stability = imf_properties['stability']
        dominant_freq = imf_properties['dominant_frequency']
        variance = imf_properties['variance']
        
        # SVR CLASSIFICATION (High frequency components)
        # Criteria: High complexity, high frequency, moderate correlation
        if (complexity > 0.08 and           # High complexity
            dominant_freq > 0.005 and       # High frequency
            correlation > 0.05 and          # Some correlation
            noise_level < 0.8):             # Not pure noise
            return 'svr_imfs'
        
        # NOISE CLASSIFICATION (Pure noise or very low correlation)
        # Criteria: Very high noise, very low correlation, very low variance
        if (noise_level > 0.7 or           # High noise
            correlation < 0.02 or           # Very low correlation
            variance < 0.0001):             # Very low variance
            return 'noise_imfs'
        
        # EXTRAPOLATION CLASSIFICATION (Simple patterns)
        # Criteria: Low complexity, moderate stability, simple patterns
        if (complexity < 0.05 and          # Low complexity
            stability > 0.6 and            # Moderate stability
            correlation > 0.02):            # Some correlation
            return 'extrapolation_imfs'
        
        # DEFAULT: Assign to SVR if no clear classification
        return 'svr_imfs'
    
    def _perform_single_eemd(self, time_series: pd.Series, sd_thresh: float, ensemble_noise: float) -> np.ndarray:
        """
        Perform single EEMD decomposition with given parameters.
        
        Args:
            time_series: Input time series
            sd_thresh: Sifting threshold
            ensemble_noise: Ensemble noise level
            
        Returns:
            IMF array
        """
        return emd.sift.ensemble_sift(
            np.array(time_series.dropna()),
            nensembles=self.config.eemd_nensembles,
            nprocesses=os.cpu_count(),
            ensemble_noise=ensemble_noise,
            imf_opts={'sd_thresh': sd_thresh}
        )
    
    def _calculate_orthogonality(self, imfs: np.ndarray) -> float:
        """
        Calculate orthogonality score between IMFs.
        
        Args:
            imfs: IMF array
            
        Returns:
            Average orthogonality score (0 = perfect orthogonality, 1 = no orthogonality)
        """
        if imfs.shape[1] < 2:
            return 0.0
        
        scores = []
        for i, j in combinations(range(imfs.shape[1]), 2):
            # Calculate correlation between IMFs
            corr = np.abs(np.corrcoef(imfs[:, i], imfs[:, j])[0, 1])
            scores.append(corr)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_correlations(self, imfs: np.ndarray, original: pd.Series) -> List[float]:
        """
        Calculate correlations between IMFs and original series.
        
        Args:
            imfs: IMF array
            original: Original time series
            
        Returns:
            List of correlation coefficients
        """
        correlations = []
        original_clean = original.dropna()
        
        for i in range(imfs.shape[1]):
            # Ensure same length
            imf_clean = imfs[:len(original_clean), i]
            correlation = np.corrcoef(imf_clean, original_clean)[0, 1]
            correlations.append(correlation)
        
        return correlations
    
    def _calculate_variance_explained(self, imfs: np.ndarray, original: pd.Series) -> pd.DataFrame:
        """
        Calculate variance explained by each IMF.
        
        Args:
            imfs: IMF array
            original: Original time series
            
        Returns:
            DataFrame with variance statistics
        """
        original_clean = original.dropna()
        total_var = np.var(original_clean)
        
        variance_data = []
        cumulative_ratio = 0.0
        
        for i in range(imfs.shape[1]):
            imf_clean = imfs[:len(original_clean), i]
            imf_var = np.var(imf_clean)
            explained_ratio = imf_var / total_var if total_var > 0 else 0
            cumulative_ratio += explained_ratio
            
            variance_data.append({
                'imf_index': i + 1,
                'variance': imf_var,
                'explained_ratio': explained_ratio,
                'cumulative_ratio': cumulative_ratio
            })
        
        return pd.DataFrame(variance_data)
    
    def _calculate_comprehensive_quality(self, imfs: np.ndarray, time_series: pd.Series, sd_thresh: float) -> Dict[str, float]:
        """
        Calculate comprehensive quality metrics for EEMD decomposition.
        
        Args:
            imfs: IMF array
            time_series: Original time series
            sd_thresh: Sifting threshold used
            
        Returns:
            Dictionary with comprehensive quality metrics
        """
        original_clean = time_series.dropna()
        
        # 1. Basic orthogonality
        orthogonality = self._calculate_orthogonality(imfs)
        
        # 2. Variance analysis
        variance_df = self._calculate_variance_explained(imfs, original_clean)
        top3_variance = variance_df['explained_ratio'].head(3).sum()
        variance_concentration = variance_df['explained_ratio'].head(5).sum()  # Top 5
        
        # 3. Correlation analysis
        correlations = self._calculate_correlations(imfs, time_series)
        max_correlation = max([abs(corr) for corr in correlations]) if correlations else 0
        mean_correlation = np.mean([abs(corr) for corr in correlations]) if correlations else 0
        
        # 4. Reconstruction quality
        reconstructed = np.sum(imfs, axis=1)
        if len(reconstructed) < len(original_clean):
            reconstructed = reconstructed[:len(original_clean)]
        elif len(reconstructed) > len(original_clean):
            reconstructed = reconstructed[:len(original_clean)]
        
        reconstruction_error = np.mean(np.abs(reconstructed - original_clean))
        reconstruction_correlation = np.corrcoef(reconstructed, original_clean)[0, 1]
        
        # 5. IMF quality assessment
        imf_quality_scores = []
        for i in range(imfs.shape[1]):
            imf_series = imfs[:len(original_clean), i]
            
            # IMF should have zero mean
            zero_mean_score = 1.0 - min(abs(np.mean(imf_series)) / np.std(imf_series), 1.0) if np.std(imf_series) > 0 else 0
            
            # IMF should have reasonable variance
            variance_score = min(variance_df.iloc[i]['explained_ratio'] * 10, 1.0)  # Scale up small variances
            
            # IMF should not be too noisy (smoothness)
            smoothness_score = 1.0 - min(np.std(np.diff(imf_series)) / np.std(imf_series), 1.0) if np.std(imf_series) > 0 else 0
            
            imf_quality = (zero_mean_score + variance_score + smoothness_score) / 3
            imf_quality_scores.append(imf_quality)
        
        mean_imf_quality = np.mean(imf_quality_scores)
        
        # 6. Meteorological-specific metrics
        # Seasonality detection (for temperature data)
        seasonality_scores = []
        for i in range(min(imfs.shape[1], 5)):  # Check first 5 IMFs
            imf_series = imfs[:len(original_clean), i]
            
            # Check for annual seasonality (365 days)
            if len(imf_series) > 365:
                # Calculate autocorrelation at lag 365
                autocorr = np.corrcoef(imf_series[:-365], imf_series[365:])[0, 1] if len(imf_series) > 365 else 0
                seasonality_scores.append(abs(autocorr))
            else:
                seasonality_scores.append(0)
        
        seasonality_strength = max(seasonality_scores) if seasonality_scores else 0
        
        # 7. Composite quality score (lower is better)
        # Weighted combination of all metrics
        composite_score = (
            orthogonality * 0.25 +                    # Orthogonality is important
            (1 - top3_variance) * 0.20 +              # Variance concentration
            reconstruction_error * 0.15 +              # Reconstruction accuracy
            (1 - mean_imf_quality) * 0.15 +           # Individual IMF quality
            (1 - reconstruction_correlation) * 0.15 +  # Reconstruction correlation
            (1 - seasonality_strength) * 0.10          # Seasonality preservation
        )
        
        return {
            'orthogonality_score': orthogonality,
            'top3_variance': top3_variance,
            'variance_concentration': variance_concentration,
            'max_correlation': max_correlation,
            'mean_correlation': mean_correlation,
            'reconstruction_error': reconstruction_error,
            'reconstruction_correlation': reconstruction_correlation,
            'mean_imf_quality': mean_imf_quality,
            'seasonality_strength': seasonality_strength,
            'composite_score': composite_score,
            'num_imfs': imfs.shape[1],
            'imf_quality_scores': imf_quality_scores
        }
    
    def save_eemd_results(self, eemd_result: EEMDResult, output_path: Path, station_name: str) -> None:
        """
        Save EEMD results to files.
        
        Args:
            eemd_result: EEMD decomposition result
            output_path: Output directory path
            station_name: Name of the station
        """
        try:
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save IMFs as CSV
            imfs_df = pd.DataFrame(eemd_result.imfs, 
                                  columns=[f'imf_{i+1}' for i in range(eemd_result.num_imfs)])
            imfs_file = output_path / f"{station_name}_imfs.csv"
            imfs_df.to_csv(imfs_file, index=False)
            
            # Save variance explained
            variance_file = output_path / f"{station_name}_variance_explained.csv"
            eemd_result.variance_explained.to_csv(variance_file, index=False)
            
            # Save correlations
            correlations_df = pd.DataFrame({
                'imf_index': range(1, eemd_result.num_imfs + 1),
                'correlation': eemd_result.correlations
            })
            correlations_file = output_path / f"{station_name}_correlations.csv"
            correlations_df.to_csv(correlations_file, index=False)
            
            # Save quality metrics
            quality_df = pd.DataFrame([eemd_result.decomposition_quality])
            quality_file = output_path / f"{station_name}_quality_metrics.csv"
            quality_df.to_csv(quality_file, index=False)
            
            self.logger.info(f"EEMD results saved for station {station_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save EEMD results for station {station_name}: {e}")
            raise DecompositionError(f"Failed to save EEMD results: {e}") 

    def _analyze_imf_properties(self, imf_series: np.ndarray, original_series: np.ndarray, imf_idx: int) -> Dict[str, float]:
        """
        Analyze fundamental properties of an IMF for scientific classification.
        
        Args:
            imf_series: IMF time series data
            original_series: Original time series data
            imf_idx: Index of the IMF
            
        Returns:
            Dictionary with analyzed properties
        """
        properties = {}
        
        # 1. CORRELATION WITH ORIGINAL SERIES
        correlation = np.corrcoef(imf_series, original_series)[0, 1]
        properties['correlation_with_original'] = correlation if not np.isnan(correlation) else 0.0
        
        # 2. VARIANCE ANALYSIS
        variance = np.var(imf_series)
        properties['variance'] = variance
        
        # 3. FREQUENCY CHARACTERISTICS
        # Calculate dominant frequency using FFT
        fft_values = np.fft.fft(imf_series)
        frequencies = np.fft.fftfreq(len(imf_series))
        dominant_freq_idx = np.argmax(np.abs(fft_values[1:len(fft_values)//2])) + 1
        dominant_frequency = frequencies[dominant_freq_idx]
        properties['dominant_frequency'] = abs(dominant_frequency)
        
        # 4. SEASONALITY STRENGTH
        # Check for annual seasonality (frequency ~ 1/365 for daily data)
        annual_freq = 1/365
        freq_tolerance = 0.1
        annual_seasonality = abs(abs(dominant_frequency) - annual_freq) < freq_tolerance
        properties['annual_seasonality'] = annual_seasonality
        
        # Check for monthly seasonality (frequency ~ 1/30)
        monthly_freq = 1/30
        monthly_seasonality = abs(abs(dominant_frequency) - monthly_freq) < freq_tolerance
        properties['monthly_seasonality'] = monthly_seasonality
        
        # 5. TREND STRENGTH
        # Calculate linear trend
        x = np.arange(len(imf_series))
        slope, _ = np.polyfit(x, imf_series, 1)
        trend_strength = abs(slope) / np.std(imf_series) if np.std(imf_series) > 0 else 0
        properties['trend_strength'] = trend_strength
        
        # 6. NOISE LEVEL
        # Calculate noise as high-frequency components
        high_freq_components = np.abs(fft_values[len(fft_values)//4:])
        noise_level = np.mean(high_freq_components) / np.mean(np.abs(fft_values))
        properties['noise_level'] = noise_level
        
        # 7. STABILITY (low variance in variance)
        rolling_var = np.array([np.var(imf_series[max(0, i-30):i+1]) for i in range(len(imf_series))])
        stability = 1 / (1 + np.std(rolling_var[30:]))  # Higher stability = lower std
        properties['stability'] = stability
        
        # 8. COMPLEXITY (number of zero crossings)
        zero_crossings = np.sum(np.diff(np.sign(imf_series)) != 0)
        complexity = zero_crossings / len(imf_series)
        properties['complexity'] = complexity
        
        # 9. AMPLITUDE CHARACTERISTICS
        amplitude = np.max(imf_series) - np.min(imf_series)
        properties['amplitude'] = amplitude
        
        # 10. MEAN VALUE
        mean_value = np.mean(imf_series)
        properties['mean_value'] = mean_value
        
        # 11. AUTOCORRELATION (important for SARIMAX)
        # Calculate lag-1 autocorrelation
        if len(imf_series) > 1:
            autocorr = np.corrcoef(imf_series[:-1], imf_series[1:])[0, 1]
            properties['autocorrelation'] = autocorr if not np.isnan(autocorr) else 0.0
        else:
            properties['autocorrelation'] = 0.0
        
        return properties
    
    # ============================================================================
    # IVariableAgnosticProcessor Interface Implementation
    # ============================================================================
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validar datos de entrada genéricos para EEMD.
        
        Args:
            data: DataFrame con datos temporales
            
        Returns:
            True si los datos son válidos para EEMD
        """
        try:
            # Verificar que el DataFrame no esté vacío
            if data.empty:
                self.logger.error("DataFrame is empty")
                return False
            
            # Verificar que tenga al menos una columna numérica
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                self.logger.error("No numeric columns found in data")
                return False
            
            # Verificar que no tenga demasiados valores faltantes
            for col in numeric_columns:
                missing_ratio = data[col].isnull().sum() / len(data)
                if missing_ratio > 0.5:
                    self.logger.error(f"Too many missing values in column {col}: {missing_ratio:.2%}")
                    return False
            
            self.logger.info("EEMD data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"EEMD data validation error: {e}")
            return False
    
    def preprocess_data(self, data: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
        """
        Preprocesar datos genéricos para EEMD.
        
        Args:
            data: DataFrame con datos originales
            config: Configuración de procesamiento
            
        Returns:
            DataFrame preprocesado para EEMD
        """
        try:
            self.logger.info("Starting EEMD data preprocessing")
            
            # Crear copia de los datos
            processed_data = data.copy()
            
            # Seleccionar columna objetivo
            if config.target_column not in processed_data.columns:
                # Buscar columna numérica por defecto
                numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    config.target_column = numeric_columns[0]
                    self.logger.info(f"Using default target column: {config.target_column}")
                else:
                    raise ValueError("No suitable target column found")
            
            # Manejar valores faltantes en la columna objetivo
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
            
            self.logger.info(f"EEMD data preprocessing completed: {len(processed_data)} points")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"EEMD data preprocessing error: {e}")
            raise
    
    def decompose_series(self, series: pd.Series, config: ProcessingConfig) -> Any:
        """
        Descomponer serie temporal usando EEMD.
        
        Args:
            series: Serie temporal a descomponer
            config: Configuración de procesamiento
            
        Returns:
            Resultado de descomposición EEMD
        """
        try:
            self.logger.info("Starting EEMD series decomposition")
            
            # Configurar parámetros EEMD
            eemd_params = {
                'ensembles': config.eemd_ensembles,
                'noise_factor': config.eemd_noise_factor,
                'sd_thresh_range': config.eemd_sd_thresh_range,
                'max_imfs': config.eemd_max_imfs,
                'quality_threshold': config.eemd_quality_threshold
            }
            
            # Realizar descomposición EEMD
            decomposition_result = self.decompose_time_series(series)
            
            self.logger.info("EEMD series decomposition completed")
            return decomposition_result
            
        except Exception as e:
            self.logger.error(f"EEMD series decomposition error: {e}")
            raise
    
    def classify_components(self, decomposition_result: Any, config: ProcessingConfig) -> Dict[str, List[int]]:
        """
        Clasificar componentes IMF de la descomposición EEMD.
        
        Args:
            decomposition_result: Resultado de descomposición EEMD
            config: Configuración de procesamiento
            
        Returns:
            Diccionario con clasificación de componentes
        """
        try:
            self.logger.info("Starting EEMD component classification")
            
            # Usar el método de clasificación existente
            if hasattr(decomposition_result, 'classify_imfs'):
                classifications = decomposition_result.classify_imfs()
            else:
                # Clasificación por defecto
                num_imfs = decomposition_result.imfs.shape[1] if hasattr(decomposition_result, 'imfs') else 0
                classifications = {
                    'sarimax_imfs': [num_imfs // 2] if num_imfs > 0 else [],
                    'svr_imfs': list(range(1, min(4, num_imfs))) if num_imfs > 1 else [],
                    'extrapolation_imfs': list(range(max(4, num_imfs - 2), num_imfs)) if num_imfs > 4 else [],
                    'noise_imfs': [0] if num_imfs > 0 else []
                }
            
            self.logger.info(f"EEMD component classification completed: {classifications}")
            return classifications
            
        except Exception as e:
            self.logger.error(f"EEMD component classification error: {e}")
            raise
    
    def train_models(self, 
                    decomposition_result: Any, 
                    classifications: Dict[str, List[int]], 
                    config: ProcessingConfig) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Entrenar modelos usando componentes EEMD.
        
        Args:
            decomposition_result: Resultado de descomposición EEMD
            classifications: Clasificación de componentes
            config: Configuración de procesamiento
            
        Returns:
            Tupla con modelos entrenados y métricas
        """
        try:
            self.logger.info("Starting EEMD-based model training")
            
            # Para EEMD, el entrenamiento de modelos se delega a otros servicios
            # Este método se mantiene por compatibilidad con la interfaz
            trained_models = {}
            model_metrics = {
                'eemd_imfs_count': decomposition_result.imfs.shape[1] if hasattr(decomposition_result, 'imfs') else 0,
                'classification_count': sum(len(imfs) for imfs in classifications.values())
            }
            
            self.logger.info("EEMD-based model training completed")
            return trained_models, model_metrics
            
        except Exception as e:
            self.logger.error(f"EEMD-based model training error: {e}")
            raise
    
    def generate_predictions(self, 
                           models: Dict[str, Any], 
                           decomposition_result: Any,
                           config: ProcessingConfig) -> Tuple[pd.Series, Optional[Tuple[pd.Series, pd.Series]]]:
        """
        Generar predicciones usando componentes EEMD.
        
        Args:
            models: Modelos entrenados
            decomposition_result: Resultado de descomposición EEMD
            config: Configuración de procesamiento
            
        Returns:
            Tupla con predicciones e intervalos de confianza
        """
        try:
            self.logger.info("Starting EEMD-based prediction generation")
            
            # Para EEMD, la generación de predicciones se delega a otros servicios
            # Este método se mantiene por compatibilidad con la interfaz
            predictions = pd.Series()
            confidence_intervals = None
            
            self.logger.info("EEMD-based prediction generation completed")
            return predictions, confidence_intervals
            
        except Exception as e:
            self.logger.error(f"EEMD-based prediction generation error: {e}")
            raise
    
    def evaluate_quality(self, 
                        input_data: pd.DataFrame, 
                        predictions: pd.Series, 
                        config: ProcessingConfig) -> Dict[str, float]:
        """
        Evaluar calidad de las predicciones basadas en EEMD.
        
        Args:
            input_data: Datos de entrada originales
            predictions: Predicciones generadas
            config: Configuración de procesamiento
            
        Returns:
            Diccionario con métricas de calidad
        """
        try:
            self.logger.info("Starting EEMD-based quality evaluation")
            
            # Métricas específicas para EEMD
            quality_metrics = {
                'eemd_quality_score': 0.0,
                'imf_count': 0,
                'decomposition_quality': 0.0
            }
            
            self.logger.info("EEMD-based quality evaluation completed")
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"EEMD-based quality evaluation error: {e}")
            raise
    
    def save_results(self, 
                    result: ProcessingResult, 
                    output_dir: Path, 
                    config: ProcessingConfig) -> Dict[str, str]:
        """
        Guardar resultados de EEMD.
        
        Args:
            result: Resultado de procesamiento
            output_dir: Directorio de salida
            config: Configuración de procesamiento
            
        Returns:
            Diccionario con rutas de archivos guardados
        """
        try:
            self.logger.info("Starting EEMD results saving")
            
            # Crear directorio de salida
            output_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = {}
            
            # Guardar resultados de EEMD si están disponibles
            if hasattr(result, 'eemd_result') and result.eemd_result is not None:
                eemd_file = output_dir / "eemd_results.json"
                import json
                with open(eemd_file, 'w') as f:
                    json.dump({
                        'num_imfs': result.eemd_result.num_imfs if hasattr(result.eemd_result, 'num_imfs') else 0,
                        'quality_score': result.eemd_result.decomposition_quality if hasattr(result.eemd_result, 'decomposition_quality') else 0.0
                    }, f, indent=2)
                saved_files['eemd_results'] = str(eemd_file)
            
            # Guardar configuración
            config_file = output_dir / "eemd_config.json"
            import json
            with open(config_file, 'w') as f:
                json.dump(config.__dict__, f, indent=2, default=str)
            saved_files['config'] = str(config_file)
            
            self.logger.info(f"EEMD results saving completed: {len(saved_files)} files")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"EEMD results saving error: {e}")
            raise
    
    def process_data(self, 
                    data: pd.DataFrame, 
                    config: Optional[ProcessingConfig] = None,
                    output_dir: Optional[Path] = None) -> ProcessingResult:
        """
        Procesar datos completos usando EEMD.
        
        Args:
            data: DataFrame con datos de entrada
            config: Configuración de procesamiento (opcional)
            output_dir: Directorio de salida (opcional)
            
        Returns:
            Resultado del procesamiento EEMD
        """
        try:
            # Configurar configuración por defecto si no se proporciona
            if config is None:
                config = ProcessingConfig(target_column="value")
            
            # Validar datos
            if not self.validate_data(data):
                raise ValueError("EEMD data validation failed")
            
            # Preprocesar datos
            processed_data = self.preprocess_data(data, config)
            
            # Descomponer serie
            target_series = processed_data[config.target_column]
            decomposition_result = self.decompose_series(target_series, config)
            
            # Clasificar componentes
            classifications = self.classify_components(decomposition_result, config)
            
            # Crear resultado
            result = ProcessingResult(
                input_data=data,
                config=config,
                eemd_result=decomposition_result,
                imf_classifications=classifications
            )
            
            # Guardar resultados si se especifica directorio
            if output_dir:
                saved_files = self.save_results(result, output_dir, config)
                result.output_files = saved_files
            
            result.success = True
            return result
            
        except Exception as e:
            self.logger.error(f"EEMD processing error: {e}")
            raise 