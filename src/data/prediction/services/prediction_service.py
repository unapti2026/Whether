"""
Prediction Service

This service generates future predictions using trained hybrid models.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from scipy.stats import gamma, poisson
import warnings

from src.core.interfaces.prediction_strategy import PredictionResult
from src.core.exceptions.processing_exceptions import PredictionError
# Removed unused service imports for code cleanup


class PredictionService:
    """
    Service for generating future predictions using trained hybrid models.
    """
    
    def __init__(self, variable_type: str):
        """
        Initialize the prediction service.
        
        Args:
            variable_type: Type of meteorological variable
        """
        self.variable_type = variable_type
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized PredictionService for {variable_type}")
        
        # Services will be initialized as needed
        
    def generate_predictions(self, 
                           eemd_result, 
                           model_result, 
                           time_series: pd.Series,
                           config) -> PredictionResult:
        """
        Generate future predictions using trained models.
        
        Args:
            eemd_result: EEMD decomposition result
            model_result: Trained models result
            time_series: Original time series
            config: Prediction configuration
            
        Returns:
            PredictionResult with predictions and metadata
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting prediction generation...")
            
            # Calculate prediction steps using new horizon system
            prediction_steps = config.calculate_prediction_steps(len(time_series))
            
            # Log horizon information
            if config.use_fixed_horizon:
                if config.prediction_horizon_days:
                    horizon_info = f"{config.prediction_horizon_days} days"
                else:
                    horizon_info = f"{config.prediction_horizon_weeks} weeks ({config.prediction_horizon_weeks * 7} days)"
                self.logger.info(f"Predicting {prediction_steps} future values (fixed horizon: {horizon_info})")
            else:
                self.logger.info(f"Predicting {prediction_steps} future values (legacy mode: {config.legacy_horizon_ratio*100:.0f}% of {len(time_series)} data points)")
            
            # Check if this is precipitation and use specialized methods
            if self._is_precipitation_variable():
                self.logger.info("🌧️ Using specialized precipitation prediction methods")
                return self._generate_precipitation_predictions(
                    eemd_result, model_result, time_series, config, prediction_steps, start_time
                )
            else:
                self.logger.info("🌡️ Using standard prediction methods for temperature/humidity")
                return self._generate_standard_predictions(
                    eemd_result, model_result, time_series, config, prediction_steps, start_time
                )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Prediction generation failed: {e}")
            raise PredictionError(f"Failed to generate predictions: {e}")
    
    def _is_precipitation_variable(self) -> bool:
        """
        Check if the current variable is precipitation.
        
        Returns:
            True if this is a precipitation variable
        """
        precipitation_keywords = ['precipitation', 'precipitacion', 'rain', 'lluvia', 'rainfall']
        return any(keyword in self.variable_type.lower() for keyword in precipitation_keywords)
    
    def _generate_precipitation_predictions(self, eemd_result, model_result, time_series: pd.Series,
                                          config, prediction_steps: int, start_time: float) -> PredictionResult:
        """
        Generate specialized predictions for precipitation data.
        
        Args:
            eemd_result: EEMD decomposition result
            model_result: Trained models result
            time_series: Original precipitation time series
            config: Prediction configuration
            prediction_steps: Number of steps to predict
            start_time: Start time for processing
            
        Returns:
            PredictionResult with precipitation-specific predictions
        """
        self.logger.info("🌧️ Starting specialized precipitation prediction...")
        
        # Step 1: Analyze precipitation characteristics
        precip_characteristics = self._analyze_precipitation_characteristics(time_series)
        self.logger.info(f"Precipitation characteristics: {precip_characteristics}")
        
        # Step 2: Generate base predictions using standard methods
        imf_predictions = self._predict_imfs(eemd_result, model_result, prediction_steps, config)
        
        # Step 3: Apply precipitation-specific post-processing
        final_prediction = self._post_process_precipitation_predictions(
            imf_predictions, time_series, precip_characteristics, prediction_steps
        )
        
        # Step 4: Generate future dates
        future_dates = self._generate_future_dates(time_series, prediction_steps)
        
        # Step 5: Calculate precipitation-specific quality metrics
        quality_metrics = self._calculate_precipitation_quality_metrics(
            time_series, final_prediction, imf_predictions, precip_characteristics
        )
        
        # Step 6: Calculate confidence intervals for precipitation
        confidence_intervals = self._calculate_precipitation_confidence_intervals(
            final_prediction, precip_characteristics
        )
        
        processing_time = time.time() - start_time
        
        self.logger.info(f"🌧️ Precipitation prediction completed in {processing_time:.2f}s")
        
        return PredictionResult(
            station_name="",  # Will be set by processor
            station_code="",  # Will be set by processor
            original_data=time_series.to_frame(),
            imf_predictions=imf_predictions,
            final_prediction=final_prediction,
            future_dates=future_dates,
            processing_time=processing_time,
            success=True,
            confidence_intervals=confidence_intervals,
            prediction_quality_metrics=quality_metrics,
            component_classification=eemd_result.decomposition_quality.get('imf_classifications', {})
        )
    
    def _generate_standard_predictions(self, eemd_result, model_result, time_series: pd.Series,
                                     config, prediction_steps: int, start_time: float) -> PredictionResult:
        """
        Generate standard predictions for temperature/humidity data.
        
        Args:
            eemd_result: EEMD decomposition result
            model_result: Trained models result
            time_series: Original time series
            config: Prediction configuration
            prediction_steps: Number of steps to predict
            start_time: Start time for processing
            
        Returns:
            PredictionResult with standard predictions
        """
        # Generate predictions for each IMF
        imf_predictions = self._predict_imfs(eemd_result, model_result, prediction_steps, config)
            
        # Reconstruct final prediction with natural scaling based on historical statistics
        final_prediction = self._reconstruct_prediction(imf_predictions, time_series)
        
        # Generate future dates
        future_dates = self._generate_future_dates(time_series, prediction_steps)
            
        # Calculate prediction quality metrics
        quality_metrics = self._calculate_prediction_quality(time_series, final_prediction, imf_predictions)
            
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(imf_predictions, final_prediction)
            
        processing_time = time.time() - start_time
            
        self.logger.info(f"Standard prediction completed in {processing_time:.2f}s")
            
        return PredictionResult(
            station_name="",  # Will be set by processor
            station_code="",  # Will be set by processor
            original_data=time_series.to_frame(),
            imf_predictions=imf_predictions,
            final_prediction=final_prediction,
            future_dates=future_dates,
            processing_time=processing_time,
            success=True,
            confidence_intervals=confidence_intervals,
            prediction_quality_metrics=quality_metrics,
            component_classification=eemd_result.decomposition_quality.get('imf_classifications', {})
        )
            
    def _analyze_precipitation_characteristics(self, time_series: pd.Series) -> Dict[str, Any]:
        """
        Analyze precipitation characteristics for specialized prediction.
        
        Args:
            time_series: Precipitation time series
            
        Returns:
            Dictionary with precipitation characteristics
        """
        # Basic statistics
        total_days = len(time_series)
        dry_days = (time_series == 0).sum()
        wet_days = total_days - dry_days
        dry_fraction = dry_days / total_days
        wet_fraction = wet_days / total_days
        
        # Wet day statistics (excluding zeros)
        wet_day_values = time_series[time_series > 0]
        if len(wet_day_values) > 0:
            wet_day_mean = wet_day_values.mean()
            wet_day_std = wet_day_values.std()
            wet_day_median = wet_day_values.median()
            wet_day_max = wet_day_values.max()
        else:
            wet_day_mean = wet_day_std = wet_day_median = wet_day_max = 0
        
        # Seasonal patterns (if we have enough data)
        seasonal_pattern = {}
        if len(time_series) >= 365:  # At least one year
            try:
                # Group by month
                if hasattr(time_series.index, 'month'):
                    monthly_stats = time_series.groupby(time_series.index.month).agg({
                        'count': 'count',
                        'mean': 'mean',
                        'wet_fraction': lambda x: (x > 0).mean()
                    })
                    seasonal_pattern = {
                        'monthly_wet_fraction': monthly_stats['wet_fraction'].to_dict(),
                        'monthly_mean': monthly_stats['mean'].to_dict()
                    }
            except Exception as e:
                self.logger.warning(f"Could not calculate seasonal patterns: {e}")
        
        # Distribution fitting
        distribution_info = self._fit_precipitation_distribution(wet_day_values)
        
        characteristics = {
            'total_days': total_days,
            'dry_days': dry_days,
            'wet_days': wet_days,
            'dry_fraction': dry_fraction,
            'wet_fraction': wet_fraction,
            'wet_day_mean': wet_day_mean,
            'wet_day_std': wet_day_std,
            'wet_day_median': wet_day_median,
            'wet_day_max': wet_day_max,
            'seasonal_pattern': seasonal_pattern,
            'distribution_info': distribution_info,
            'is_very_dry': dry_fraction > 0.8,  # More than 80% dry days
            'is_moderate': 0.3 <= dry_fraction <= 0.8,  # Moderate precipitation
            'is_wet': dry_fraction < 0.3  # Less than 30% dry days
        }
        
        return characteristics
    
    def _fit_precipitation_distribution(self, wet_day_values: pd.Series) -> Dict[str, Any]:
        """
        Fit probability distributions to wet day precipitation values.
        
        Args:
            wet_day_values: Precipitation values for wet days only
            
        Returns:
            Dictionary with fitted distribution parameters
        """
        if len(wet_day_values) < 10:
            return {'best_distribution': 'insufficient_data'}
        
        distribution_info = {}
        
        try:
            # Try Gamma distribution (most common for precipitation)
            gamma_params = gamma.fit(wet_day_values)
            gamma_aic = self._calculate_aic(wet_day_values, gamma, gamma_params)
            distribution_info['gamma'] = {
                'params': gamma_params,
                'aic': gamma_aic
            }
            
            # Try Exponential distribution
            exp_params = stats.expon.fit(wet_day_values)
            exp_aic = self._calculate_aic(wet_day_values, stats.expon, exp_params)
            distribution_info['exponential'] = {
                'params': exp_params,
                'aic': exp_aic
            }
            
            # Try Log-normal distribution
            lognorm_params = stats.lognorm.fit(wet_day_values)
            lognorm_aic = self._calculate_aic(wet_day_values, stats.lognorm, lognorm_params)
            distribution_info['lognormal'] = {
                'params': lognorm_params,
                'aic': lognorm_aic
            }
            
            # Find best distribution
            aic_scores = {k: v['aic'] for k, v in distribution_info.items()}
            best_dist = min(aic_scores, key=aic_scores.get)
            
            distribution_info['best_distribution'] = best_dist
            distribution_info['best_aic'] = aic_scores[best_dist]
            
        except Exception as e:
            self.logger.warning(f"Distribution fitting failed: {e}")
            distribution_info['best_distribution'] = 'fitting_failed'
        
        return distribution_info
    
    def _calculate_aic(self, data: pd.Series, distribution, params) -> float:
        """
        Calculate AIC for a fitted distribution.
        
        Args:
            data: Data series
            distribution: Distribution class
            params: Fitted parameters
            
        Returns:
            AIC value
        """
        try:
            log_likelihood = distribution.logpdf(data, *params).sum()
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            return aic
        except:
            return float('inf')
    
    def _post_process_precipitation_predictions(self, imf_predictions: Dict[int, np.ndarray],
                                              time_series: pd.Series,
                                              precip_characteristics: Dict[str, Any],
                                              prediction_steps: int) -> np.ndarray:
        """
        Post-process predictions to ensure they are realistic for precipitation.
        
        Args:
            imf_predictions: Raw IMF predictions
            time_series: Original precipitation series
            precip_characteristics: Precipitation characteristics
            prediction_steps: Number of prediction steps
            
        Returns:
            Post-processed precipitation predictions
        """
        self.logger.info("🌧️ Post-processing precipitation predictions...")
        
        # Step 1: Get base prediction from IMF reconstruction
        if not imf_predictions:
            return np.zeros(prediction_steps)
        
        # Sum all IMF predictions
        base_prediction = np.zeros(prediction_steps)
        for imf_pred in imf_predictions.values():
            base_prediction += imf_pred
        
        # Step 2: Apply precipitation-specific transformations
        processed_prediction = self._apply_precipitation_transformations(
            base_prediction, time_series, precip_characteristics
        )
        
        # Step 3: Ensure non-negative values
        processed_prediction = np.maximum(processed_prediction, 0)
        
        # Step 4: Apply realistic bounds based on historical data
        processed_prediction = self._apply_precipitation_bounds(
            processed_prediction, precip_characteristics
        )
        
        # Step 5: Validate precipitation patterns
        processed_prediction = self._validate_precipitation_patterns(
            processed_prediction, precip_characteristics
        )
        
        self.logger.info(f"🌧️ Post-processing completed. "
                        f"Mean: {np.mean(processed_prediction):.2f}, "
                        f"Max: {np.max(processed_prediction):.2f}, "
                        f"Dry days: {(processed_prediction == 0).sum()}/{len(processed_prediction)}")
        
        return processed_prediction
    
    def _apply_precipitation_transformations(self, base_prediction: np.ndarray,
                                           time_series: pd.Series,
                                           precip_characteristics: Dict[str, Any]) -> np.ndarray:
        """
        Apply transformations specific to precipitation data.
        
        Args:
            base_prediction: Base prediction from IMF reconstruction
            time_series: Original precipitation series
            precip_characteristics: Precipitation characteristics
            
        Returns:
            Transformed prediction
        """
        # Get historical statistics
        historical_mean = time_series.mean()
        historical_std = time_series.std()
        historical_max = time_series.max()
        
        # Get precipitation characteristics
        dry_fraction = precip_characteristics['dry_fraction']
        wet_fraction = precip_characteristics['wet_fraction']
        wet_day_mean = precip_characteristics['wet_day_mean']
        
        # Step 1: Scale to match historical mean and variance
        prediction_mean = np.mean(base_prediction)
        prediction_std = np.std(base_prediction)
        
        if prediction_std > 0:
            # Scale to match historical statistics
            scaled_prediction = (base_prediction - prediction_mean) * (historical_std / prediction_std) + historical_mean
        else:
            scaled_prediction = np.full_like(base_prediction, historical_mean)
        
        # Step 2: Apply dry/wet day logic
        if wet_fraction > 0:
            # Calculate threshold for wet days based on historical patterns
            # Use a small threshold to separate dry from wet days
            wet_threshold = wet_day_mean * 0.1  # 10% of wet day mean
            
            # Apply probabilistic wet/dry day assignment
            processed_prediction = np.zeros_like(scaled_prediction)
            
            for i in range(len(scaled_prediction)):
                # Determine if this day should be wet or dry based on historical probability
                if np.random.random() < wet_fraction:
                    # This should be a wet day
                    if scaled_prediction[i] < wet_threshold:
                        # Generate a realistic wet day value
                        if precip_characteristics['distribution_info']['best_distribution'] == 'gamma':
                            params = precip_characteristics['distribution_info']['gamma']['params']
                            processed_prediction[i] = gamma.rvs(*params)
                        else:
                            # Use log-normal or exponential as fallback
                            processed_prediction[i] = np.random.exponential(wet_day_mean)
                    else:
                        # Keep the predicted value but ensure it's reasonable
                        processed_prediction[i] = max(scaled_prediction[i], wet_threshold)
                else:
                    # This should be a dry day
                    processed_prediction[i] = 0
        else:
            # If no wet days in historical data, keep scaled prediction
            processed_prediction = scaled_prediction
        
        return processed_prediction
    
    def _apply_precipitation_bounds(self, prediction: np.ndarray,
                                  precip_characteristics: Dict[str, Any]) -> np.ndarray:
        """
        Apply realistic bounds to precipitation predictions.
        
        Args:
            prediction: Precipitation prediction
            precip_characteristics: Precipitation characteristics
            
        Returns:
            Bounded prediction
        """
        # Get historical bounds
        historical_max = precip_characteristics['wet_day_max']
        
        # Apply reasonable bounds
        # Use historical max with some tolerance for extreme events
        max_bound = historical_max * 1.5  # Allow 50% more than historical max
        
        # Clip to bounds
        bounded_prediction = np.clip(prediction, 0, max_bound)
        
        return bounded_prediction
    
    def _validate_precipitation_patterns(self, prediction: np.ndarray,
                                       precip_characteristics: Dict[str, Any]) -> np.ndarray:
        """
        Validate and correct precipitation patterns for realism.
        
        Args:
            prediction: Precipitation prediction
            precip_characteristics: Precipitation characteristics
            
        Returns:
            Validated prediction
        """
        # Check dry day fraction
        predicted_dry_fraction = (prediction == 0).sum() / len(prediction)
        target_dry_fraction = precip_characteristics['dry_fraction']
        
        # If dry fraction is too different from historical, adjust
        if abs(predicted_dry_fraction - target_dry_fraction) > 0.2:  # 20% tolerance
            self.logger.info(f"Adjusting dry fraction from {predicted_dry_fraction:.3f} to target {target_dry_fraction:.3f}")
            
            # Sort non-zero values
            non_zero_indices = np.where(prediction > 0)[0]
            non_zero_values = prediction[non_zero_indices]
            
            # Calculate how many days should be dry
            target_dry_days = int(len(prediction) * target_dry_fraction)
            target_wet_days = len(prediction) - target_dry_days
            
            if len(non_zero_values) > target_wet_days:
                # Too many wet days, make some dry
                sorted_indices = np.argsort(non_zero_values)
                indices_to_zero = non_zero_indices[sorted_indices[:len(non_zero_values) - target_wet_days]]
                prediction[indices_to_zero] = 0
            elif len(non_zero_values) < target_wet_days:
                # Too many dry days, make some wet
                dry_indices = np.where(prediction == 0)[0]
                if len(dry_indices) > 0:
                    # Generate some wet days
                    wet_day_mean = precip_characteristics['wet_day_mean']
                    num_to_make_wet = min(target_wet_days - len(non_zero_values), len(dry_indices))
                    indices_to_wet = np.random.choice(dry_indices, num_to_make_wet, replace=False)
                    
                    for idx in indices_to_wet:
                        if precip_characteristics['distribution_info']['best_distribution'] == 'gamma':
                            params = precip_characteristics['distribution_info']['gamma']['params']
                            prediction[idx] = gamma.rvs(*params)
                        else:
                            prediction[idx] = np.random.exponential(wet_day_mean)
        
        return prediction
    
    def _calculate_precipitation_quality_metrics(self, time_series: pd.Series,
                                               final_prediction: np.ndarray,
                                               imf_predictions: Dict[int, np.ndarray],
                                               precip_characteristics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate quality metrics specific to precipitation predictions.
        
        Args:
            time_series: Original precipitation series
            final_prediction: Final precipitation prediction
            imf_predictions: IMF predictions
            precip_characteristics: Precipitation characteristics
            
        Returns:
            Dictionary with precipitation-specific quality metrics
        """
        # Basic metrics
        original_dry_fraction = precip_characteristics['dry_fraction']
        predicted_dry_fraction = (final_prediction == 0).sum() / len(final_prediction)
        
        # Wet day statistics
        original_wet_mean = precip_characteristics['wet_day_mean']
        predicted_wet_values = final_prediction[final_prediction > 0]
        predicted_wet_mean = predicted_wet_values.mean() if len(predicted_wet_values) > 0 else 0
        
        # Distribution similarity
        distribution_similarity = 1.0 - abs(original_dry_fraction - predicted_dry_fraction)
        
        # Wet day mean similarity
        if original_wet_mean > 0:
            wet_mean_similarity = 1.0 - abs(predicted_wet_mean - original_wet_mean) / original_wet_mean
        else:
            wet_mean_similarity = 1.0 if predicted_wet_mean == 0 else 0.0
        
        # Overall quality score
        quality_score = (distribution_similarity + wet_mean_similarity) / 2
        
        metrics = {
            'dry_fraction_similarity': distribution_similarity,
            'wet_mean_similarity': wet_mean_similarity,
            'overall_quality_score': quality_score,
            'predicted_dry_fraction': predicted_dry_fraction,
            'predicted_wet_mean': predicted_wet_mean,
            'prediction_realism': self._assess_precipitation_realism(final_prediction, precip_characteristics)
        }
        
        return metrics
    
    def _assess_precipitation_realism(self, prediction: np.ndarray,
                                    precip_characteristics: Dict[str, Any]) -> float:
        """
        Assess how realistic the precipitation prediction is.
        
        Args:
            prediction: Precipitation prediction
            precip_characteristics: Precipitation characteristics
            
        Returns:
            Realism score (0-1)
        """
        realism_score = 1.0
        
        # Check for negative values
        if np.any(prediction < 0):
            realism_score -= 0.3
        
        # Check for unrealistic high values
        historical_max = precip_characteristics['wet_day_max']
        if np.any(prediction > historical_max * 2):
            realism_score -= 0.2
        
        # Check for too many consecutive wet days (unrealistic)
        consecutive_wet = 0
        max_consecutive_wet = 0
        for val in prediction:
            if val > 0:
                consecutive_wet += 1
                max_consecutive_wet = max(max_consecutive_wet, consecutive_wet)
            else:
                consecutive_wet = 0
        
        if max_consecutive_wet > 30:  # More than a month of consecutive rain
            realism_score -= 0.2
        
        # Check for too many consecutive dry days (unrealistic)
        consecutive_dry = 0
        max_consecutive_dry = 0
        for val in prediction:
            if val == 0:
                consecutive_dry += 1
                max_consecutive_dry = max(max_consecutive_dry, consecutive_dry)
            else:
                consecutive_dry = 0
        
        if max_consecutive_dry > 365:  # More than a year without rain
            realism_score -= 0.3
        
        return max(0.0, realism_score)
    
    def _calculate_precipitation_confidence_intervals(self, prediction: np.ndarray,
                                                    precip_characteristics: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals specific to precipitation.
        
        Args:
            prediction: Precipitation prediction
            precip_characteristics: Precipitation characteristics
            
        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        # For precipitation, use asymmetric confidence intervals
        # Lower bound: 0 (can't have negative precipitation)
        # Upper bound: based on historical variability
        
        wet_day_std = precip_characteristics['wet_day_std']
        confidence_factor = 1.96  # 95% confidence interval
        
        lower_bound = np.zeros_like(prediction)  # Precipitation can't be negative
        
        # Upper bound: prediction + variability
        upper_bound = prediction + confidence_factor * wet_day_std
        
        # Ensure upper bound is reasonable
        historical_max = precip_characteristics['wet_day_max']
        upper_bound = np.minimum(upper_bound, historical_max * 2)
        
        return lower_bound, upper_bound
    
    def _predict_imfs(self, eemd_result, model_result, prediction_steps: int, config) -> Dict[int, np.ndarray]:
        """
        Generate predictions for each IMF using NEW CLASSIFICATION SYSTEM.
        
        NEW LOGIC:
        1. Predict SARIMAX IMF (exactly one)
        2. Predict SVR IMFs (high frequency)
        3. Handle extrapolation and noise IMFs
        
        Args:
            eemd_result: EEMD decomposition result
            model_result: Trained models result
            prediction_steps: Number of steps to predict
            config: Prediction configuration
            
        Returns:
            Dictionary mapping IMF index to predictions
        """
        imf_predictions = {}
        
        # Get IMF classifications from new system
        imf_classifications = eemd_result.decomposition_quality.get('imf_classifications', {})
        
        self.logger.info("🎯 Starting IMF predictions with NEW CLASSIFICATION...")
        
        # 1. Predict SARIMAX IMF (exactly one)
        sarimax_imfs = imf_classifications.get('sarimax_imfs', [])
        for imf_idx in sarimax_imfs:
            if imf_idx in model_result.sarimax_model:
                try:
                    self.logger.info(f"🏆 Predicting IMF {imf_idx + 1} with SARIMAX (Best composite score)")
                    prediction = self._predict_with_sarimax(
                        eemd_result.imfs[:, imf_idx],
                        model_result.sarimax_model[imf_idx],
                        prediction_steps
                    )
                    imf_predictions[imf_idx] = prediction
                except Exception as e:
                    self.logger.warning(f"❌ Failed to predict IMF {imf_idx + 1} with SARIMAX: {e}")
                    # Use simple extrapolation as fallback
                    imf_predictions[imf_idx] = self._simple_extrapolation(
                        eemd_result.imfs[:, imf_idx], prediction_steps
                    )
            else:
                self.logger.error(f"❌ No SARIMAX model found for IMF {imf_idx + 1} - this should not happen!")
                imf_predictions[imf_idx] = self._simple_extrapolation(
                    eemd_result.imfs[:, imf_idx], prediction_steps
                )
        
        # 2. Predict SVR IMFs (high frequency components)
        svr_imfs = imf_classifications.get('svr_imfs', [])
        for imf_idx in svr_imfs:
            if imf_idx in model_result.svr_models:
                try:
                    self.logger.info(f"🤖 Predicting IMF {imf_idx + 1} with SVR")
                    prediction = self._predict_with_svr(
                        eemd_result.imfs[:, imf_idx],
                        model_result.svr_models[imf_idx],
                        prediction_steps,
                        config,
                        imf_idx
                    )
                    imf_predictions[imf_idx] = prediction
                except Exception as e:
                    self.logger.warning(f"❌ Failed to predict IMF {imf_idx + 1} with SVR: {e}")
                    # Use simple extrapolation as fallback
                    imf_predictions[imf_idx] = self._simple_extrapolation(
                        eemd_result.imfs[:, imf_idx], prediction_steps
                    )
            else:
                self.logger.warning(f"⚠️ No SVR model for IMF {imf_idx + 1}, using extrapolation")
                imf_predictions[imf_idx] = self._simple_extrapolation(
                    eemd_result.imfs[:, imf_idx], prediction_steps
                )
        
        # 3. Handle extrapolation IMFs (simple patterns)
        extrapolation_imfs = imf_classifications.get('extrapolation_imfs', [])
        for imf_idx in extrapolation_imfs:
            self.logger.info(f"📈 Predicting IMF {imf_idx + 1} with extrapolation (Simple pattern)")
            imf_predictions[imf_idx] = self._simple_extrapolation(
                eemd_result.imfs[:, imf_idx], prediction_steps
            )
        
        # 4. Handle noise IMFs (simple extrapolation or discard)
        noise_imfs = imf_classifications.get('noise_imfs', [])
        for imf_idx in noise_imfs:
            self.logger.info(f"🔇 Predicting IMF {imf_idx + 1} with extrapolation (Noise component)")
            imf_predictions[imf_idx] = self._simple_extrapolation(
                eemd_result.imfs[:, imf_idx], prediction_steps
            )
        
        # Log prediction summary
        self.logger.info("📊 IMF Prediction Summary:")
        self.logger.info(f"  - SARIMAX predictions: {len([i for i in sarimax_imfs if i in imf_predictions])}")
        self.logger.info(f"  - SVR predictions: {len([i for i in svr_imfs if i in imf_predictions])}")
        self.logger.info(f"  - Extrapolation predictions: {len([i for i in extrapolation_imfs if i in imf_predictions])}")
        self.logger.info(f"  - Noise predictions: {len([i for i in noise_imfs if i in imf_predictions])}")
        
        return imf_predictions
    
    def _predict_with_svr(self, imf_series: np.ndarray, svr_model, prediction_steps: int, config, imf_idx: int = None) -> np.ndarray:
        """
        Predict future values using SVR model with DYNAMIC enhanced features.
        
        Args:
            imf_series: IMF time series
            svr_model: Trained SVR model
            prediction_steps: Number of steps to predict
            config: Prediction configuration
            imf_idx: IMF index for context
            
        Returns:
            Array of predicted values
        """
        predictions = []
        
        # Initialize dynamic series with historical data
        dynamic_series = imf_series.copy()
        
        self.logger.info(f"Starting SVR prediction with dynamic features for {prediction_steps} steps")
        
        for step in range(prediction_steps):
            # Prepare input with DYNAMIC features based on current series
            current_input = self._prepare_svr_prediction_input_dynamic(dynamic_series, config.num_lags, imf_idx)
            
            # Scale input using StandardScaler (same as training)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaled_input = scaler.fit_transform(current_input)
            
            # Predict next value
            next_pred = svr_model.predict(scaled_input)[0]
            predictions.append(next_pred)
            
            # Add prediction to dynamic series for next iteration
            dynamic_series = np.append(dynamic_series, next_pred)
            
        self.logger.info(f"SVR prediction completed successfully")
        
        return np.array(predictions)
    
    def _prepare_svr_prediction_input_dynamic(self, dynamic_series: np.ndarray, num_lags: int, imf_idx: int = None) -> np.ndarray:
        """
        Prepare input for SVR prediction with basic temporal features.
        Features are recalculated based on the current state of the dynamic series.
        
        Args:
            dynamic_series: Dynamic series (historical + predictions so far)
            num_lags: Number of lagged features
            imf_idx: IMF index for context
            
        Returns:
            Input array with basic temporal features
        """
        # Ensure we have enough data
        if len(dynamic_series) <= num_lags:
            self.logger.warning(f"Dynamic series too short for {num_lags} lags, using all available data")
            num_lags = max(1, len(dynamic_series) - 1)
        
        # Create basic temporal features
        features = []
        
        # Add lagged features - ensure all have the same length
        for lag in range(1, num_lags + 1):
            if lag < len(dynamic_series):
                # Get the last 'lag' values
                lagged_feature = dynamic_series[-lag]
                features.append(lagged_feature)
            else:
                # Pad with zeros if not enough data
                features.append(0.0)
        
        # Add rolling statistics
        if len(dynamic_series) >= 7:
            rolling_mean = np.mean(dynamic_series[-7:])
            rolling_std = np.std(dynamic_series[-7:])
        else:
            rolling_mean = np.mean(dynamic_series)
            rolling_std = np.std(dynamic_series)
        
        features.extend([rolling_mean, rolling_std])
        
        # Convert to numpy array and reshape to 2D
        features = np.array(features, dtype=float)
        features = features.reshape(1, -1)
        
        return features
    
    def _predict_with_sarimax(self, imf_series: np.ndarray, sarimax_model, prediction_steps: int) -> np.ndarray:
        """
        Generate predictions using SARIMAX model.
        Windows-compatible version without signal.SIGALRM.
        
        Args:
            imf_series: IMF series data
            sarimax_model: Trained SARIMAX model
            prediction_steps: Number of steps to predict
            
        Returns:
            Array of predictions
        """
        try:
            # Generate forecast
            forecast = sarimax_model.forecast(steps=prediction_steps)
            
            # Validate forecast
            if np.any(np.isnan(forecast)) or np.any(np.isinf(forecast)):
                raise ValueError("SARIMAX forecast contains NaN or infinite values")
            
            # Ensure forecast is numpy array
            if not isinstance(forecast, np.ndarray):
                forecast = np.array(forecast)
            
            self.logger.info(f"SARIMAX prediction successful: {len(forecast)} steps")
            return forecast
                
        except Exception as e:
            self.logger.warning(f"SARIMAX prediction failed: {e}")
            raise
    
    def _simple_extrapolation(self, imf_series: np.ndarray, prediction_steps: int) -> np.ndarray:
        """
        Improved extrapolation that maintains realistic scale and characteristics.
        
        Args:
            imf_series: IMF time series
            prediction_steps: Number of steps to predict
            
        Returns:
            Array of predicted values with realistic scale
        """
        # Calculate basic statistics of the IMF
        imf_mean = np.mean(imf_series)
        imf_std = np.std(imf_series)
        imf_range = np.max(imf_series) - np.min(imf_series)
        
        # Determine extrapolation method based on IMF characteristics
        if imf_std < 0.01:  # Very stable IMF
            # Use constant value with small noise
            predictions = np.full(prediction_steps, imf_mean)
            if imf_std > 0:
                predictions += np.random.normal(0, imf_std * 0.1, prediction_steps)
        else:
            # Calculate linear trend
            x = np.arange(len(imf_series))
            slope, intercept = np.polyfit(x, imf_series, 1)
            
            # Extrapolate with trend
            future_x = np.arange(len(imf_series), len(imf_series) + prediction_steps)
            predictions = slope * future_x + intercept
            
            # Add realistic variation based on historical characteristics
            if imf_std > 0:
                # Add noise proportional to historical variation
                noise = np.random.normal(0, imf_std * 0.3, prediction_steps)
                predictions += noise
        
        # Ensure predictions maintain reasonable bounds
        if imf_range > 0:
            # Scale predictions to maintain relative magnitude
            pred_min = np.min(predictions)
            pred_max = np.max(predictions)
            pred_range = pred_max - pred_min
            
            if pred_range > 0:
                # Normalize to historical range
                predictions = (predictions - pred_min) / pred_range * imf_range + np.min(imf_series)
        
        return predictions
    
    def _reconstruct_prediction(self, imf_predictions: Dict[int, np.ndarray], original_series: pd.Series = None) -> np.ndarray:
        """
        Reconstruct final prediction by summing IMF predictions.
        Uses historical statistics for natural scaling instead of forced values.
        CRITICAL: Ensures continuity with the last historical value.
        
        Args:
            imf_predictions: Dictionary of IMF predictions
            original_series: Original time series for historical statistics
            
        Returns:
            Final reconstructed prediction with natural scaling and continuity
        """
        if not imf_predictions:
            return np.array([])
        
        # Get the length of predictions
        prediction_length = len(next(iter(imf_predictions.values())))
        
        # Sum all IMF predictions (natural reconstruction)
        final_prediction = np.zeros(prediction_length)
        for imf_pred in imf_predictions.values():
            final_prediction += imf_pred
        
        # CRITICAL FIX: Ensure continuity with last historical value
        if original_series is not None and len(original_series) > 0:
            last_historical_value = original_series.iloc[-1]
            
            # Calculate historical statistics from original series
            historical_mean = original_series.mean()
            historical_std = original_series.std()
            historical_min = original_series.min()
            historical_max = original_series.max()
            historical_range = historical_max - historical_min
            
            # Check if prediction needs scaling (only if significantly different from historical)
            prediction_mean = np.mean(final_prediction)
            prediction_std = np.std(final_prediction)
            
            # Calculate scale factors based on historical vs prediction statistics
            mean_scale_factor = historical_mean / (prediction_mean + 1e-8)
            std_scale_factor = historical_std / (prediction_std + 1e-8)
            
            # Only scale if prediction is significantly different from historical
            mean_difference = abs(prediction_mean - historical_mean) / (abs(historical_mean) + 1e-8)
            std_difference = abs(prediction_std - historical_std) / (abs(historical_std) + 1e-8)
            
            if mean_difference > 0.5 or std_difference > 0.5:  # 50% difference threshold
                self.logger.info(f"Scaling prediction to match historical statistics")
                
                # Apply natural scaling while preserving patterns
                if prediction_std > 0:
                    # Scale to match historical mean and std
                    final_prediction = (final_prediction - prediction_mean) * std_scale_factor + historical_mean
                else:
                    # If no variation, use historical mean with small variation
                    final_prediction = np.full_like(final_prediction, historical_mean) + np.random.normal(0, historical_std * 0.1, len(final_prediction))
            
            # CRITICAL: Ensure first prediction connects directly to last historical value
            # No gap, no transition - just ensure reasonable continuity
            first_prediction_value = final_prediction[0]
            
            # Calculate the difference
            value_difference = abs(first_prediction_value - last_historical_value)
            historical_std_for_comparison = max(historical_std, 1.0)  # Avoid division by zero
            
            # Only adjust if the difference is unreasonably large (> 2 standard deviations)
            # This ensures the first prediction is reasonable without forcing it to be exactly equal
            if value_difference > 2 * historical_std_for_comparison:
                # Adjust only the first value to be within reasonable range of last historical
                # Use a weighted average: 70% last historical + 30% predicted (smooth but direct)
                final_prediction[0] = 0.7 * last_historical_value + 0.3 * first_prediction_value
                self.logger.info(f"Adjusted first prediction for continuity: {first_prediction_value:.2f} -> {final_prediction[0]:.2f} (last historical: {last_historical_value:.2f})")
            else:
                # Difference is reasonable, keep the prediction as is
                self.logger.debug(f"First prediction is within reasonable range: {first_prediction_value:.2f} (last historical: {last_historical_value:.2f}, diff: {value_difference:.2f})")
        
        # Apply natural bounds based on historical range (if available)
        if original_series is not None:
            # Use historical bounds with small tolerance
            tolerance = historical_range * 0.1  # 10% tolerance
            lower_bound = historical_min - tolerance
            upper_bound = historical_max + tolerance
            
            # Clip to natural bounds
            final_prediction = np.clip(final_prediction, lower_bound, upper_bound)
        else:
            # Fallback to reasonable bounds if no historical data
            final_prediction = np.clip(final_prediction, 10.0, 50.0)
        
        # Basic validation and correction
        if original_series is not None and len(original_series) > 10:
            # Simple statistical validation
            original_mean = original_series.mean()
            original_std = original_series.std()
            prediction_mean = np.mean(final_prediction)
            
            # Check if prediction is within reasonable bounds
            if abs(prediction_mean - original_mean) > 3 * original_std:
                self.logger.warning(f"Prediction mean ({prediction_mean:.2f}) differs significantly from original mean ({original_mean:.2f})")
        
        # Basic coherence check
        if len(imf_predictions) > 1:
            # Simple check: sum of IMF predictions should be close to final prediction
            sum_imfs = sum(imf_predictions.values())
            reconstruction_error = np.mean(np.abs(final_prediction - sum_imfs))
            if reconstruction_error > 0.1 * np.std(final_prediction):
                self.logger.warning(f"High reconstruction error: {reconstruction_error:.4f}")
            else:
                self.logger.info(f"Reconstruction error acceptable: {reconstruction_error:.4f}")
        
        return final_prediction
    
    def _generate_future_dates(self, time_series: pd.Series, prediction_steps: int) -> pd.DatetimeIndex:
        """
        Generate future dates for predictions.
        
        Args:
            time_series: Original time series
            prediction_steps: Number of steps to predict
            
        Returns:
            DatetimeIndex of future dates
        """
        if isinstance(time_series.index, pd.DatetimeIndex):
            last_date = time_series.index[-1]
            # Assume daily frequency
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=prediction_steps,
                freq='D'
            )
        else:
            # If no datetime index, create sequential dates starting from today (date only)
            today = datetime.now().date()
            future_dates = pd.date_range(
                start=today,
                periods=prediction_steps,
                freq='D'
            )
        
        return future_dates
    
    def _calculate_prediction_quality(self, time_series: pd.Series,
                                    final_prediction: np.ndarray,
                                    imf_predictions: Dict[int, np.ndarray]) -> Dict[str, float]:
        """
        Calculate quality metrics for standard predictions (temperature/humidity).
        
        Args:
            time_series: Original time series
            final_prediction: Final prediction
            imf_predictions: IMF predictions
            
        Returns:
            Dictionary with quality metrics
        """
        # Basic statistical metrics
        original_mean = time_series.mean()
        original_std = time_series.std()
        prediction_mean = np.mean(final_prediction)
        prediction_std = np.std(final_prediction)
        
        # Consistency metrics
        mean_consistency = 1.0 - abs(prediction_mean - original_mean) / (abs(original_mean) + 1e-8)
        std_consistency = 1.0 - abs(prediction_std - original_std) / (abs(original_std) + 1e-8)
        
        # Trend consistency
        original_trend = time_series.diff().mean()
        prediction_trend = np.diff(final_prediction).mean()
        trend_consistency = 1.0 - abs(prediction_trend - original_trend) / (abs(original_trend) + 1e-8)
        
        # Diversity score (how much variation is captured)
        diversity_score = prediction_std / (original_std + 1e-8)
        
        # Overall quality score
        quality_score = (mean_consistency + std_consistency + trend_consistency) / 3
        
        metrics = {
            'mean_consistency': mean_consistency,
            'std_consistency': std_consistency,
            'trend_consistency': trend_consistency,
            'diversity_score': diversity_score,
            'overall_quality_score': quality_score,
            'num_imfs_used': len(imf_predictions),
            'prediction_length': len(final_prediction)
        }
    
        return metrics
    
    def _calculate_confidence_intervals(self, imf_predictions: Dict[int, np.ndarray],
                                      final_prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for standard predictions.
        
        Args:
            imf_predictions: IMF predictions
            final_prediction: Final prediction
            
        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        if not imf_predictions:
            return np.array([]), np.array([])
        
        # Calculate prediction uncertainty based on IMF variability
        prediction_length = len(final_prediction)
        
        # Estimate uncertainty from IMF predictions
        imf_std = np.zeros(prediction_length)
        for imf_pred in imf_predictions.values():
            imf_std += np.std(imf_pred) ** 2
        
        imf_std = np.sqrt(imf_std)
        
        # Calculate confidence intervals
        confidence_factor = 1.96  # 95% confidence interval
        lower_bound = final_prediction - confidence_factor * imf_std
        upper_bound = final_prediction + confidence_factor * imf_std
        
        return lower_bound, upper_bound 