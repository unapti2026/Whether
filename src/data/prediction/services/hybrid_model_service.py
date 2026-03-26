"""
Hybrid Model Service

This service implements hybrid SVR + SARIMAX modeling for EEMD components.
Implements variable-agnostic interfaces for complete independence from
specific meteorological variables.
"""

import logging
import time
import gc
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# try:
#     from pmdarima import auto_arima
#     AUTO_ARIMA_AVAILABLE = True
# except ImportError:
#     AUTO_ARIMA_AVAILABLE = False
#     warnings.warn("pmdarima not available. Using fallback SARIMAX parameters.")

from src.core.interfaces.prediction_strategy import ModelTrainingResult
from src.core.interfaces.variable_agnostic_interfaces import (
    IVariableAgnosticProcessor, ProcessingConfig, ProcessingResult
)
from src.core.exceptions.processing_exceptions import ModelTrainingError

logger = logging.getLogger(__name__)


class HybridModelService(IVariableAgnosticProcessor):
    """
    Service for training hybrid SVR + SARIMAX models on EEMD components.
    """
    
    def __init__(self, variable_type: str):
        """
        Initialize the hybrid model service.
        
        Args:
            variable_type: Type of meteorological variable
        """
        self.variable_type = variable_type
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized HybridModelService for {variable_type}")
        
        # Initialize scalers for each IMF
        self.scalers: Dict[int, StandardScaler] = {}
        self.svr_models: Dict[int, SVR] = {}
        self.sarimax_models: Dict[int, Any] = {}
    
    def validate_sarimax_input_data(self, imf_series: np.ndarray, imf_idx: int) -> Dict[str, Any]:
        """
        Validación robusta de datos antes del entrenamiento SARIMAX.
        
        Args:
            imf_series: Serie temporal del IMF
            imf_idx: Índice del IMF
            
        Returns:
            Diccionario con resultados de validación
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # 1. Verificar longitud mínima
            if len(imf_series) < 365:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"IMF {imf_idx}: Insufficient data ({len(imf_series)} < 365 days)")
            
            # 2. Verificar valores faltantes
            missing_count = np.sum(np.isnan(imf_series))
            missing_ratio = missing_count / len(imf_series)
            if missing_ratio > 0.1:  # Más del 10% de valores faltantes
                validation_results['warnings'].append(f"IMF {imf_idx}: High missing data ratio ({missing_ratio:.2%})")
            
            # 3. Verificar valores infinitos
            inf_count = np.sum(np.isinf(imf_series))
            if inf_count > 0:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"IMF {imf_idx}: Found {inf_count} infinite values")
            
            # 4. Verificar varianza
            variance = np.var(imf_series)
            if variance < 1e-6:
                validation_results['warnings'].append(f"IMF {imf_idx}: Very low variance ({variance:.2e})")
            
            # 5. Verificar outliers (método IQR)
            q75, q25 = np.percentile(imf_series, [75, 25])
            iqr = q75 - q25
            if iqr > 0:  # Evitar división por cero
                outlier_threshold = 1.5 * iqr
                outliers = np.sum((imf_series < q25 - outlier_threshold) | (imf_series > q75 + outlier_threshold))
                outlier_ratio = outliers / len(imf_series)
                if outlier_ratio > 0.05:  # Más del 5% outliers
                    validation_results['warnings'].append(f"IMF {imf_idx}: High outlier ratio ({outlier_ratio:.2%})")
            
            # 6. Verificar autocorrelación básica
            if len(imf_series) > 10:
                autocorr = np.corrcoef(imf_series[:-1], imf_series[1:])[0, 1]
                if abs(autocorr) < 0.1:
                    validation_results['warnings'].append(f"IMF {imf_idx}: Low autocorrelation ({autocorr:.3f})")
            
            # 7. Generar recomendaciones
            if len(validation_results['warnings']) > 0:
                validation_results['recommendations'].append("Consider data preprocessing before SARIMAX training")
            
            if missing_ratio > 0.05:
                validation_results['recommendations'].append("Consider imputation for missing values")
            
            self.logger.info(f"IMF {imf_idx}: Input validation completed - Valid: {validation_results['is_valid']}, "
                           f"Warnings: {len(validation_results['warnings'])}, Errors: {len(validation_results['errors'])}")
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"IMF {imf_idx}: Validation failed - {str(e)}")
            self.logger.error(f"IMF {imf_idx}: Input validation error - {e}")
        
        return validation_results
        

        
    def train_models_legacy(self, eemd_result, time_series: pd.Series, config, adaptive_config=None) -> ModelTrainingResult:
        """
        Train hybrid models on EEMD components using NEW CLASSIFICATION SYSTEM.
        
        NEW LOGIC:
        1. Train SARIMAX for EXACTLY ONE IMF (best composite score)
        2. Train SVR for high-frequency components
        3. Handle remaining components appropriately
        
        Args:
            eemd_result: EEMD decomposition result
            time_series: Original time series
            config: Prediction configuration
            adaptive_config: Optional adaptive configuration for memory optimization
            
        Returns:
            ModelTrainingResult with trained models
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting NEW CLASSIFICATION MODEL TRAINING...")
            
            # Get IMF classifications from new system
            imf_classifications = eemd_result.decomposition_quality.get('imf_classifications', {})
            
            self.logger.info("Model Training Distribution:")
            for classification_type, imf_indices in imf_classifications.items():
                if imf_indices:
                    imf_numbers = [i+1 for i in imf_indices]
                    self.logger.info(f"  - {classification_type}: IMFs {imf_numbers}")
            
            # Train SARIMAX model for EXACTLY ONE IMF (best composite score)
            sarimax_imfs = imf_classifications.get('sarimax_imfs', [])
            sarimax_models = {}
            
            if sarimax_imfs:
                # Should be exactly one IMF for SARIMAX
                sarimax_imf_idx = sarimax_imfs[0]
                self.logger.info(f"Training SARIMAX for IMF {sarimax_imf_idx + 1} (Best composite score)")
                
                # CRITICAL: Set force_sarimax_imf flag to ensure SARIMAX is trained
                # regardless of other criteria (user requirement: "SIEMPRE UNO Y SOLO UN")
                if not hasattr(config, 'force_sarimax_imf'):
                    config.force_sarimax_imf = sarimax_imf_idx
                else:
                    config.force_sarimax_imf = sarimax_imf_idx
                
                sarimax_models = self._train_sarimax_models(
                    eemd_result.imfs,
                    [sarimax_imf_idx],  # Only the best IMF
                    config,
                    adaptive_config
                )
            else:
                self.logger.warning("⚠️ No IMF assigned to SARIMAX - this should not happen!")
            
            # Train SVR models for high-frequency IMFs
            svr_imfs = imf_classifications.get('svr_imfs', [])
            svr_models = {}
            
            if svr_imfs:
                self.logger.info(f"🤖 Training SVR models for {len(svr_imfs)} high-frequency IMFs")
                svr_models = self._train_svr_models(
                    eemd_result.imfs, 
                    svr_imfs,
                    config
                )
            
            # Handle extrapolation IMFs (simple methods)
            extrapolation_imfs = imf_classifications.get('extrapolation_imfs', [])
            if extrapolation_imfs:
                imf_numbers = [i+1 for i in extrapolation_imfs]
                self.logger.info(f"📈 Extrapolation IMFs: {imf_numbers} (will use simple extrapolation)")
            
            # Handle noise IMFs (discard or simple methods)
            noise_imfs = imf_classifications.get('noise_imfs', [])
            if noise_imfs:
                imf_numbers = [i+1 for i in noise_imfs]
                self.logger.info(f"🔇 Noise IMFs: {imf_numbers} (will use simple extrapolation)")
            
            # Combine all models
            all_models = {**svr_models, **sarimax_models}
            
            training_time = time.time() - start_time
            
            self.logger.info(f"✅ Model training completed in {training_time:.2f}s")
            self.logger.info(f"📊 Final Model Distribution:")
            self.logger.info(f"  - SARIMAX models: {len(sarimax_models)}")
            self.logger.info(f"  - SVR models: {len(svr_models)}")
            self.logger.info(f"  - Extrapolation IMFs: {len(extrapolation_imfs)}")
            self.logger.info(f"  - Noise IMFs: {len(noise_imfs)}")
            
            # Store all IMF classifications for prediction service
            all_imf_classifications = {
                'sarimax_imfs': sarimax_imfs,
                'svr_imfs': svr_imfs,
                'extrapolation_imfs': extrapolation_imfs,
                'noise_imfs': noise_imfs
            }
            
            return ModelTrainingResult(
                svr_models=svr_models,
                sarimax_model=sarimax_models,
                selected_imf_for_sarimax=sarimax_imfs[0] if sarimax_imfs else 0,
                training_time=training_time,
                success=True,
                imf_classifications=all_imf_classifications  # Store for prediction service
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"❌ Model training failed: {e}")
            raise ModelTrainingError(f"Failed to train models: {e}")
    
    def _train_svr_models(self, imfs: np.ndarray, high_freq_imfs: List[int], config) -> Dict[int, SVR]:
        """
        Train SVR models for high-frequency IMFs with optimized hyperparameters.
        
        Args:
            imfs: IMF array
            high_freq_imfs: List of high-frequency IMF indices
            config: Prediction configuration
            
        Returns:
            Dictionary of trained SVR models
        """
        svr_models = {}
        
        # Get IMF properties for adaptive parameter adjustment
        from src.data.prediction.services.eemd_service import EEMDService
        eemd_service = EEMDService(config)
        
        # Optimize hyperparameters on the first IMF (most representative)
        if high_freq_imfs:
            first_imf_idx = high_freq_imfs[0]
            first_imf_series = imfs[:, first_imf_idx]
            
            self.logger.info(f"🎯 Optimizing SVR hyperparameters on IMF {first_imf_idx + 1}...")
            base_params = self._optimize_svr_hyperparameters(first_imf_series, config.num_lags, high_freq_imfs[0])
            
            # Get IMF properties for adaptive adjustment
            imf_properties = eemd_service._analyze_imf_properties(
                first_imf_series, 
                np.zeros_like(first_imf_series),  # Placeholder for original series
                first_imf_idx
            )
            
            # Adjust base parameters for the first IMF
            adjusted_params = self._adjust_params_for_imf(base_params, imf_properties)
            
            self.logger.info(f"  Base SVR params: {adjusted_params}")
        
        for i, imf_idx in enumerate(high_freq_imfs):
            try:
                self.logger.info(f"Training SVR model for IMF {imf_idx + 1}")
                
                # Prepare data
                imf_series = imfs[:, imf_idx]
                X, y = self._prepare_svr_data(imf_series, config.num_lags, imf_idx)
                
                # Split data
                split_idx = int(len(X) * config.train_test_split)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # FASE 2: Calculate temporal sample weights for SVR training
                sample_weights_train = None
                use_temporal_weighting = getattr(config, 'use_temporal_weighting', True)
                if use_temporal_weighting and len(X_train) > 10:  # Only if enough samples
                    try:
                        from src.data.prediction.services.temporal_weighting import TemporalWeighting
                        
                        weighting_method = getattr(config, 'temporal_weighting_method', 'exponential')
                        decay_factor = getattr(config, 'temporal_decay_factor', 0.1)
                        increment_factor = getattr(config, 'temporal_increment_factor', 1.0)
                        weighting_strength = getattr(config, 'temporal_weighting_strength', 0.5)
                        
                        # Calculate weights for training data
                        # We need to map training samples back to original series positions
                        # X_train corresponds to imf_series[num_lags:num_lags+split_idx]
                        # Pass the full imf_series and let the function calculate positions correctly
                        num_lags = config.num_lags
                        
                        # Calculate weights for the full X/y first, then extract training portion
                        X_full, y_full = self._prepare_svr_data(imf_series, config.num_lags, imf_idx)
                        sample_weights_full = TemporalWeighting.calculate_sample_weights_for_svr(
                            imf_series,
                            X_full,
                            y_full,
                            method=weighting_method,
                            decay_factor=decay_factor,
                            increment_factor=increment_factor
                        )
                        
                        # Extract training portion of weights
                        sample_weights_train = sample_weights_full[:split_idx]
                        
                        # Apply weighting strength: blend with uniform weights
                        if weighting_strength < 1.0:
                            uniform_weights = np.ones(len(sample_weights_train))
                            sample_weights_train = (1 - weighting_strength) * uniform_weights + \
                                                   weighting_strength * sample_weights_train
                        
                        self.logger.info(f"Calculated temporal sample weights for SVR IMF {imf_idx + 1}: "
                                       f"min={sample_weights_train.min():.3f}, max={sample_weights_train.max():.3f}, "
                                       f"mean={sample_weights_train.mean():.3f}, method={weighting_method}")
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate temporal weights for SVR: {e}. Using uniform weights.")
                        sample_weights_train = None
                
                # Scale data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Use optimized parameters for first IMF, adjust for others
                if i == 0:
                    # Use the optimized parameters for the first IMF
                    svr_params = adjusted_params
                else:
                    # Get properties for this IMF and adjust parameters
                    imf_properties = eemd_service._analyze_imf_properties(
                        imf_series, 
                        np.zeros_like(imf_series),  # Placeholder
                        imf_idx
                    )
                    svr_params = self._adjust_params_for_imf(base_params, imf_properties)
                
                self.logger.info(f"  Using SVR params for IMF {imf_idx + 1}: {svr_params}")
                
                # Train SVR model with optimized parameters and temporal weighting
                svr_model = SVR(
                    kernel=svr_params['kernel'],
                    C=svr_params['C'],
                    gamma=svr_params['gamma']
                )
                
                # FASE 2: Fit with temporal sample weights if available
                if sample_weights_train is not None:
                    svr_model.fit(X_train_scaled, y_train, sample_weight=sample_weights_train)
                    self.logger.info(f"Trained SVR IMF {imf_idx + 1} with temporal weighting")
                else:
                    svr_model.fit(X_train_scaled, y_train)
                    self.logger.info(f"Trained SVR IMF {imf_idx + 1} without temporal weighting")
                
                # Evaluate model with enhanced metrics
                y_pred = svr_model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Additional metrics
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100
                
                # Feature importance (for linear kernel)
                if svr_params['kernel'] == 'linear':
                    feature_importance = np.abs(svr_model.coef_[0])
                    top_features = np.argsort(feature_importance)[-3:]  # Top 3 features
                    self.logger.info(f"    Top features: {top_features}")
                
                self.logger.info(f"  SVR IMF {imf_idx + 1} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                self.logger.info(f"    RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
                
                # Store model and scaler
                svr_models[imf_idx] = svr_model
                self.scalers[imf_idx] = scaler
                
            except Exception as e:
                self.logger.warning(f"Failed to train SVR for IMF {imf_idx + 1}: {e}")
                continue
        
        return svr_models
    
    def _train_sarimax_models(self, imfs: np.ndarray, low_freq_imfs: List[int], config, adaptive_config=None) -> Dict[int, Any]:
        """
        Train SARIMAX models for low-frequency IMFs with memory optimization.
        
        Args:
            imfs: IMF array
            low_freq_imfs: List of low-frequency IMF indices
            config: Prediction configuration
            adaptive_config: Optional adaptive configuration for memory optimization
            
        Returns:
            Dictionary of trained SARIMAX models
        """
        sarimax_models = {}
        
        # Get SARIMAX configuration
        if adaptive_config:
            sarimax_config = adaptive_config
        else:
            # Default configuration if adaptive_config not provided
            from src.config.adaptive_config import create_adaptive_config
            sarimax_config = create_adaptive_config(imfs.shape[0], self.variable_type)
        
        # CRITICAL: Pass the force_sarimax_imf flag from config to sarimax_config
        if hasattr(config, 'force_sarimax_imf'):
            sarimax_config.force_sarimax_imf = config.force_sarimax_imf
            self.logger.info(f"🎯 Force SARIMAX flag set: IMF {config.force_sarimax_imf + 1}")
        
        # TEMPORAL VALIDATION: Configure SARIMAX data limitation
        if hasattr(sarimax_config, 'sarimax_data_limit_days'):
            self.logger.info(f"📅 SARIMAX temporal validation: maximum {sarimax_config.sarimax_data_limit_days} days for training")
        
        # Limit number of SARIMAX models per station
        max_models = getattr(sarimax_config, 'sarimax_max_models_per_station', 2)
        models_trained = 0
        
        for imf_idx in low_freq_imfs:
            if models_trained >= max_models:
                self.logger.info(f"Maximum SARIMAX models ({max_models}) reached, skipping remaining IMFs")
                break
                
            try:
                # Validate if SARIMAX should be applied
                if not self._should_apply_sarimax(imf_idx, imfs[:, imf_idx], sarimax_config):
                    self.logger.info(f"Skipping SARIMAX for IMF {imf_idx + 1} (criteria not met)")
                    continue
                
                self.logger.info(f"Training SARIMAX model for IMF {imf_idx + 1}")
                
                # Train with memory management and timeout
                model = self._train_sarimax_with_memory_management(
                    imfs[:, imf_idx], imf_idx, sarimax_config
                )
                
                if model is not None:
                    sarimax_models[imf_idx] = model
                    models_trained += 1
                    self.logger.info(f"  SARIMAX trained successfully for IMF {imf_idx + 1}")
                else:
                    self.logger.warning(f"  SARIMAX failed for IMF {imf_idx + 1}, using extrapolation")
                
            except Exception as e:
                self.logger.warning(f"Failed to train SARIMAX for IMF {imf_idx + 1}: {e}")
                continue
        
        return sarimax_models
    
    def _should_apply_sarimax(self, imf_idx: int, imf_series: np.ndarray, sarimax_config) -> bool:
        """
        Determine if SARIMAX should be applied to this IMF.
        
        NEW LOGIC: If an IMF is explicitly selected for SARIMAX by the composite scoring system,
        we MUST train SARIMAX regardless of other criteria. This ensures the user's requirement
        of "SIEMPRE UNO Y SOLO UN componente asignado con Sarimax" is met.
        
        Args:
            imf_idx: Index of the IMF
            imf_series: IMF series data
            sarimax_config: SARIMAX configuration
            
        Returns:
            True if SARIMAX should be applied
        """
        # CRITICAL: If this IMF was explicitly selected for SARIMAX by composite scoring,
        # we MUST train SARIMAX regardless of other criteria
        if hasattr(sarimax_config, 'force_sarimax_imf') and sarimax_config.force_sarimax_imf == imf_idx:
            self.logger.info(f"🎯 FORCING SARIMAX for IMF {imf_idx + 1} (explicitly selected by composite scoring)")
            return True
        
        # Check minimum records (365 for daily data)
        min_records = getattr(sarimax_config, 'sarimax_min_records', 365)
        if len(imf_series) < min_records:
            self.logger.debug(f"IMF {imf_idx + 1}: Insufficient records ({len(imf_series)} < {min_records})")
            return False
        
        # More permissive frequency criteria - allow IMFs 0, 1, 2 for SARIMAX
        if imf_idx > 2:
            self.logger.debug(f"IMF {imf_idx + 1}: High frequency (index > 2), using SVR instead")
            return False
        
        # Check variance (more permissive)
        variance = np.var(imf_series)
        if variance < 0.001:  # Reduced threshold
            self.logger.debug(f"IMF {imf_idx + 1}: Very low variance ({variance:.6f} < 0.001)")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(imf_series)) or np.any(np.isinf(imf_series)):
            self.logger.debug(f"IMF {imf_idx + 1}: Contains NaN or infinite values")
            return False
        
        # Additional check: ensure the series has some structure
        autocorr = np.corrcoef(imf_series[:-1], imf_series[1:])[0, 1]
        if abs(autocorr) < 0.1:
            self.logger.debug(f"IMF {imf_idx + 1}: Low autocorrelation ({autocorr:.3f}), may be noise")
            return False
        
        return True
    
    def _train_sarimax_with_memory_management(self, imf_series: np.ndarray, imf_idx: int, sarimax_config) -> Optional[Any]:
        """
        Train SARIMAX model with intelligent memory management, TEMPORAL VALIDATION, and TEMPORAL WEIGHTING.
        SARIMAX uses maximum 365 days of data for training (parametrizable).
        FASE 2: Applies temporal weighting to prioritize recent data.
        
        Args:
            imf_series: IMF series data
            imf_idx: Index of the IMF
            sarimax_config: SARIMAX configuration
            
        Returns:
            Trained SARIMAX model or None if failed
        """
        operation_name = f"sarimax_imf_{imf_idx + 1}"
        
        try:
            # TEMPORAL VALIDATION: Limit data to maximum 365 days for SARIMAX training
            # Note: This is different from max_missing_block_days (548) which is for imputation
            data_limit_days = getattr(sarimax_config, 'sarimax_data_limit_days', 365)
            max_data_points = min(data_limit_days, len(imf_series))
            
            # Use only the most recent data for training
            if len(imf_series) > max_data_points:
                training_data = imf_series[-max_data_points:]
                self.logger.info(f"Training SARIMAX for IMF {imf_idx + 1} with TEMPORAL VALIDATION: "
                               f"using last {max_data_points} days from {len(imf_series)} total points")
            else:
                training_data = imf_series
                self.logger.info(f"Training SARIMAX for IMF {imf_idx + 1} with full series ({len(imf_series)} points)")
            
            # FASE 2: Apply temporal weighting if enabled
            use_temporal_weighting = getattr(sarimax_config, 'use_temporal_weighting', True)
            if use_temporal_weighting and len(training_data) > 30:  # Only apply if series is long enough
                weighting_method = getattr(sarimax_config, 'temporal_weighting_method', 'windowed')
                recent_window_days = getattr(sarimax_config, 'temporal_recent_window_days', 30)
                weighting_strength = getattr(sarimax_config, 'temporal_weighting_strength', 0.5)
                
                if weighting_method == 'windowed':
                    # Use windowed sampling to prioritize recent data
                    from src.data.prediction.services.temporal_weighting import TemporalWeighting
                    
                    sampled_data, original_indices = TemporalWeighting.apply_windowed_sampling(
                        training_data,
                        recent_window_days=recent_window_days
                    )
                    
                    # Apply weighting strength: interpolate between original and weighted
                    if weighting_strength < 1.0:
                        # Blend: use (1-strength) of original + strength of weighted
                        # For windowed, we already sampled, so we just use the sampled data
                        # But we can adjust the effect by using more/less aggressive sampling
                        pass  # Windowed sampling already applied
                    
                    training_data = sampled_data
                    self.logger.info(f"Applied temporal weighting (windowed) to SARIMAX training: "
                                   f"{len(imf_series)} -> {len(training_data)} points "
                                   f"(method={weighting_method}, strength={weighting_strength:.2f})")
                else:
                    # For exponential/linear: use windowed as approximation
                    # (SARIMAX doesn't support direct weights)
                    from src.data.prediction.services.temporal_weighting import TemporalWeighting
                    
                    sampled_data, _ = TemporalWeighting.apply_windowed_sampling(
                        training_data,
                        recent_window_days=recent_window_days
                    )
                    training_data = sampled_data
                    self.logger.info(f"Applied temporal weighting ({weighting_method} via windowed) to SARIMAX: "
                                   f"{len(imf_series)} -> {len(training_data)} points")
            
            # Get SARIMAX parameters
            seasonal_order = getattr(sarimax_config, 'sarimax_seasonal_order', (1, 0, 0, 365))
            max_iter = getattr(sarimax_config, 'sarimax_max_iter', 100)
            enforce_stationarity = getattr(sarimax_config, 'sarimax_enforce_stationarity', False)
            enforce_invertibility = getattr(sarimax_config, 'sarimax_enforce_invertibility', False)
            disp = getattr(sarimax_config, 'sarimax_disp', False)
            
            # Use default SARIMAX parameters
            order = getattr(sarimax_config, 'sarimax_order', (1, 1, 0))
            
            self.logger.info(f"SARIMAX parameters: Order {order}, Seasonal {seasonal_order} (annual 365-day cycle), Max iter {max_iter}")
            
            # Create and fit SARIMAX model with temporal validation
            model = SARIMAX(
                training_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility
            )
            
            fitted_model = model.fit(
                disp=disp,
                maxiter=max_iter
            )
            
            # Validate model
            if fitted_model is None or not hasattr(fitted_model, 'forecast'):
                raise ValueError("Invalid SARIMAX model")
            
            self.logger.info(f"SARIMAX trained successfully for IMF {imf_idx + 1} with temporal validation")
            
            # Basic memory cleanup
            if getattr(sarimax_config, 'sarimax_enable_memory_cleanup', True):
                del model  # Clean up the unfitted model
                gc.collect()  # Force garbage collection
            
            return fitted_model
                
        except MemoryError:
            self.logger.warning(f"Memory error during SARIMAX training for IMF {imf_idx + 1}")
            return None
        except Exception as e:
            self.logger.warning(f"SARIMAX training failed for IMF {imf_idx + 1}: {e}")
            return None
    
    def _optimize_sarimax_parameters_with_auto_arima(self, imf_series: np.ndarray, imf_idx: int, sarimax_config) -> Tuple[int, int, int]:
        """
        Optimize SARIMAX parameters using auto_arima with validation temporal.
        
        Args:
            imf_series: IMF series data
            imf_idx: Index of the IMF
            sarimax_config: SARIMAX configuration
            
        Returns:
            Optimal (p, d, q) order tuple
        """
        try:
            self.logger.info(f"Starting auto_arima optimization for IMF {imf_idx + 1}")
            
            # Determine seasonal period based on series characteristics
            seasonal_period = self._determine_seasonal_period_for_auto_arima(imf_series, sarimax_config)
            
            # Configure auto_arima parameters based on series length and characteristics
            max_p = min(3, len(imf_series) // 100)  # Adaptive max_p
            max_d = 2  # Maximum differencing
            max_q = min(3, len(imf_series) // 100)  # Adaptive max_q
            max_P = 1 if seasonal_period > 0 else 0
            max_D = 1 if seasonal_period > 0 else 0
            max_Q = 1 if seasonal_period > 0 else 0
            
            # Ensure minimum values
            max_p = max(1, max_p)
            max_q = max(1, max_q)
            
            self.logger.info(f"Auto_arima parameters: max_p={max_p}, max_d={max_d}, max_q={max_q}, seasonal_period={seasonal_period}")
            
            # Run auto_arima with temporal validation
            auto_model = auto_arima(
                imf_series,
                start_p=0, start_q=0,
                max_p=max_p, max_d=max_d, max_q=max_q,
                max_P=max_P, max_D=max_D, max_Q=max_Q,
                seasonal=seasonal_period > 0,
                m=seasonal_period if seasonal_period > 0 else 1,
                stepwise=True,  # Use stepwise search for efficiency
                suppress_warnings=True,
                error_action='ignore',
                trace=False,  # Reduce logging
                information_criterion='aic',  # Use AIC for model selection
                random_state=42  # For reproducibility
            )
            
            if auto_model is None:
                self.logger.warning(f"Auto_arima failed for IMF {imf_idx + 1}, using fallback parameters")
                return (1, 1, 0)  # Fallback to default
            
            # Extract optimal parameters
            order = auto_model.order
            seasonal_order = auto_model.seasonal_order
            
            self.logger.info(f"Auto_arima optimal parameters for IMF {imf_idx + 1}: Order {order}, Seasonal {seasonal_order}")
            
            # Validate parameters
            if order is None or len(order) != 3:
                self.logger.warning(f"Invalid auto_arima order for IMF {imf_idx + 1}, using fallback")
                return (1, 1, 0)
            
            return order
            
        except Exception as e:
            self.logger.warning(f"Auto_arima optimization failed for IMF {imf_idx + 1}: {e}")
            # Fallback to stationarity-based parameters
            p, d, q = self.stationarity_manager.get_sarimax_order_recommendation(
                imf_series, 
                f"imf_{imf_idx + 1}"
            )
            return (p, d, q)
    
    def _determine_seasonal_period_for_auto_arima(self, imf_series: np.ndarray, sarimax_config) -> int:
        """
        Determine optimal seasonal period for auto_arima based on series characteristics.
        For meteorological data: ALWAYS use annual seasonality (365 days).
        
        Args:
            imf_series: IMF series data
            sarimax_config: SARIMAX configuration
            
        Returns:
            Seasonal period (365 for annual seasonality)
        """
        series_length = len(imf_series)
        
        # For meteorological data: ALWAYS use annual seasonality (365 days)
        # This is based on the proven annual cycles in meteorological data
        if series_length >= 365:  # At least 1 year of data
            return 365  # Annual seasonality (365 days)
        else:
            # If less than 1 year, no seasonality
            return 0
    
    def _prepare_svr_data(self, series: np.ndarray, num_lags: int, imf_idx: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for SVR training with basic temporal features.
        
        Args:
            series: Time series data
            num_lags: Number of lagged features
            imf_idx: IMF index for context
            
        Returns:
            Tuple of (X, y) arrays with basic features
        """
        # Ensure we have enough data
        if len(series) <= num_lags:
            self.logger.warning(f"Series too short for {num_lags} lags, using all available data")
            num_lags = max(1, len(series) - 1)
        
        # Create basic temporal features
        features = []
        
        # Add lagged features - ensure all have the same length
        target_length = len(series) - num_lags
        
        for lag in range(1, num_lags + 1):
            lagged_feature = series[lag:len(series) - num_lags + lag]
            # Ensure the lagged feature has the correct length
            if len(lagged_feature) > target_length:
                lagged_feature = lagged_feature[:target_length]
            elif len(lagged_feature) < target_length:
                # Pad with zeros if too short
                lagged_feature = np.pad(lagged_feature, (0, target_length - len(lagged_feature)), 'constant')
            features.append(lagged_feature)
        
        # Add rolling statistics for the target length
        if target_length >= 7:
            rolling_mean = np.array([np.mean(series[i:i+7]) for i in range(num_lags, len(series) - 6)])
            rolling_std = np.array([np.std(series[i:i+7]) for i in range(num_lags, len(series) - 6)])
            
            # Ensure rolling stats have correct length
            if len(rolling_mean) > target_length:
                rolling_mean = rolling_mean[:target_length]
                rolling_std = rolling_std[:target_length]
            elif len(rolling_mean) < target_length:
                # Pad with mean/std of the series
                pad_size = target_length - len(rolling_mean)
                rolling_mean = np.pad(rolling_mean, (0, pad_size), 'constant', constant_values=np.mean(series))
                rolling_std = np.pad(rolling_std, (0, pad_size), 'constant', constant_values=np.std(series))
        else:
            # Use simple statistics for short series
            rolling_mean = np.full(target_length, np.mean(series))
            rolling_std = np.full(target_length, np.std(series))
        
        features.extend([rolling_mean, rolling_std])
        
        # Create feature matrix
        X = np.column_stack(features)
        y = series[num_lags:num_lags + target_length]
        
        # Ensure X and y have the same length
        min_length = min(len(X), len(y))
        X = X[:min_length]
        y = y[:min_length]
        
        # Remove any rows with NaN values
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.logger.info(f"Generated {X.shape[1]} basic features for SVR training (lags: {num_lags}, samples: {len(X)})")
        
        return X, y
    
    def _create_representative_sample(self, time_series: np.ndarray, sample_size: int = 1500) -> np.ndarray:
        """
        Create a representative sample that preserves key patterns.
        
        Args:
            time_series: Original time series
            sample_size: Target sample size
            
        Returns:
            Representative sample array
        """
        if len(time_series) <= sample_size:
            return time_series
        
        # Strategy 1: Stratified sampling by percentiles
        percentiles = [10, 25, 50, 75, 90]
        samples_per_percentile = sample_size // len(percentiles)
        
        representative_sample = []
        
        for percentile in percentiles:
            threshold = np.percentile(time_series, percentile)
            std_dev = np.std(time_series)
            
            # Find data near this percentile
            near_threshold = time_series[np.abs(time_series - threshold) < std_dev * 0.5]
            
            if len(near_threshold) > 0:
                # Sample from this region
                n_samples = min(samples_per_percentile, len(near_threshold))
                sample_indices = np.random.choice(len(near_threshold), n_samples, replace=False)
                representative_sample.extend(near_threshold[sample_indices])
        
        # Strategy 2: Add some random samples to ensure coverage
        remaining_size = sample_size - len(representative_sample)
        if remaining_size > 0:
            random_indices = np.random.choice(len(time_series), remaining_size, replace=False)
            representative_sample.extend(time_series[random_indices])
        
        return np.array(representative_sample)
    
    def _optimize_svr_hyperparameters(self, imf_series: np.ndarray, num_lags: int = 7, imf_idx: int = None) -> Dict[str, Any]:
        """
        Optimize SVR hyperparameters using simple grid search.
        
        Args:
            imf_series: IMF series data
            num_lags: Number of lagged features
            imf_idx: IMF index for context
            
        Returns:
            Dictionary with optimized hyperparameters
        """
        try:
            # Create representative sample for optimization
            representative_sample = self._create_representative_sample(imf_series, sample_size=1500)
            
            # Prepare data
            X, y = self._prepare_svr_data(representative_sample, num_lags, imf_idx)
            
            if len(X) < 100:  # Too small for meaningful optimization
                self.logger.warning("Sample too small for hyperparameter optimization, using defaults")
                return {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
            
            # Simple grid search
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
            
            best_params = None
            best_score = float('inf')
            
            # Manual grid search
            for C in param_grid['C']:
                for gamma in param_grid['gamma']:
                    for kernel in param_grid['kernel']:
                        try:
                            svr = SVR(C=C, gamma=gamma, kernel=kernel)
                            svr.fit(X, y)
                            y_pred = svr.predict(X)
                            mse = mean_squared_error(y, y_pred)
                            
                            if mse < best_score:
                                best_score = mse
                                best_params = {'C': C, 'gamma': gamma, 'kernel': kernel}
                        except Exception:
                            continue
            
            if best_params is None:
                return {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
            
            self.logger.info(f"  Best SVR params: {best_params} (MSE: {best_score:.6f})")
            return best_params
            
        except Exception as e:
            self.logger.warning(f"Hyperparameter optimization failed: {e}, using defaults")
            return {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'}
    
    # Removed unused methods for code cleanup
    
    def _adjust_params_for_imf(self, base_params: Dict[str, Any], imf_properties: Dict[str, float]) -> Dict[str, Any]:
        """
        Adjust base hyperparameters for specific IMF characteristics.
        
        Args:
            base_params: Base hyperparameters from optimization
            imf_properties: IMF properties dictionary
            
        Returns:
            Adjusted hyperparameters
        """
        adjusted_params = base_params.copy()
        
        # Adjust based on frequency characteristics
        frequency = imf_properties.get('dominant_frequency', 0)
        complexity = imf_properties.get('complexity', 0)
        variance = imf_properties.get('variance', 0)
        
        # High frequency IMFs: More regularization, smaller C
        if frequency > 0.01:
            adjusted_params['C'] = max(0.1, base_params['C'] * 0.5)
            adjusted_params['gamma'] = 'auto'  # Better for high frequency
        
        # Low complexity IMFs: Simpler kernel
        if complexity < 0.05:
            adjusted_params['kernel'] = 'linear'
        
        # High variance IMFs: More regularization
        if variance > 0.1:
            adjusted_params['C'] = max(0.1, base_params['C'] * 0.7)
        
        return adjusted_params
    
    def _determine_seasonal_order(self, series_length: int) -> Tuple[int, int, int, int]:
        """
        Determine seasonal order based on series length.
        For meteorological data: ALWAYS use annual seasonality (365 days).
        
        Args:
            series_length: Length of the time series
            
        Returns:
            Seasonal order tuple (P, D, Q, s)
        """
        if series_length >= 365:  # At least 1 year
            return (1, 0, 0, 365)  # Annual seasonality (365 days)
        else:
            return (0, 0, 0, 0)    # No seasonality 
    
    # Removed unused summary methods for code cleanup
    
    # Removed unused summary methods for code cleanup
    
    def _get_imf_properties(self, imf_series: np.ndarray) -> Dict[str, float]:
        """
        Extract properties from IMF series for adaptive configuration.
        
        Args:
            imf_series: IMF series data
            
        Returns:
            Dictionary with IMF properties
        """
        try:
            # Calculate basic properties
            variance = np.var(imf_series)
            mean = np.mean(imf_series)
            
            # Calculate frequency characteristics using FFT
            fft = np.fft.fft(imf_series)
            freqs = np.fft.fftfreq(len(imf_series))
            
            # Find dominant frequency
            power_spectrum = np.abs(fft) ** 2
            dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
            dominant_frequency = abs(freqs[dominant_freq_idx])
            
            # Calculate complexity (entropy-based)
            hist, _ = np.histogram(imf_series, bins=min(50, len(imf_series)//10))
            hist = hist[hist > 0]  # Remove zero bins
            if len(hist) > 1:
                entropy = -np.sum(hist * np.log(hist / np.sum(hist)))
                complexity = entropy / np.log(len(hist))  # Normalized entropy
            else:
                complexity = 0.0
            
            # Calculate trend strength
            x = np.arange(len(imf_series))
            slope, _, r_value, _, _ = stats.linregress(x, imf_series)
            trend_strength = abs(r_value)
            
            return {
                'variance': variance,
                'mean': mean,
                'dominant_frequency': dominant_frequency,
                'complexity': complexity,
                'trend_strength': trend_strength
            }
        except Exception as e:
            self.logger.warning(f"Failed to extract IMF properties: {e}")
            return {
                'variance': 1.0,
                'mean': 0.0,
                'dominant_frequency': 0.01,
                'complexity': 0.5,
                'trend_strength': 0.5
            }
    
    # ============================================================================
    # IVariableAgnosticProcessor Interface Implementation
    # ============================================================================
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validar datos de entrada genéricos para modelos híbridos.
        
        Args:
            data: DataFrame con datos temporales
            
        Returns:
            True si los datos son válidos para entrenamiento de modelos
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
            
            # Verificar que tenga suficientes datos para entrenamiento
            if len(data) < 100:
                self.logger.error(f"Insufficient data for training: {len(data)} points")
                return False
            
            # Verificar que no tenga demasiados valores faltantes
            for col in numeric_columns:
                missing_ratio = data[col].isnull().sum() / len(data)
                if missing_ratio > 0.3:
                    self.logger.error(f"Too many missing values in column {col}: {missing_ratio:.2%}")
                    return False
            
            self.logger.info("Hybrid model data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Hybrid model data validation error: {e}")
            return False
    
    def preprocess_data(self, data: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
        """
        Preprocesar datos genéricos para modelos híbridos.
        
        Args:
            data: DataFrame con datos originales
            config: Configuración de procesamiento
            
        Returns:
            DataFrame preprocesado para entrenamiento
        """
        try:
            self.logger.info("Starting hybrid model data preprocessing")
            
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
            
            self.logger.info(f"Hybrid model data preprocessing completed: {len(processed_data)} points")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Hybrid model data preprocessing error: {e}")
            raise
    
    def decompose_series(self, series: pd.Series, config: ProcessingConfig) -> Any:
        """
        Descomponer serie temporal para modelos híbridos.
        
        Args:
            series: Serie temporal a descomponer
            config: Configuración de procesamiento
            
        Returns:
            Resultado de descomposición
        """
        try:
            self.logger.info("Starting hybrid model series decomposition")
            
            # Para modelos híbridos, la descomposición se delega al servicio EEMD
            # Este método se mantiene por compatibilidad con la interfaz
            self.logger.info("Hybrid model series decomposition completed")
            return None
            
        except Exception as e:
            self.logger.error(f"Hybrid model series decomposition error: {e}")
            raise
    
    def classify_components(self, decomposition_result: Any, config: ProcessingConfig) -> Dict[str, List[int]]:
        """
        Clasificar componentes para modelos híbridos.
        
        Args:
            decomposition_result: Resultado de descomposición
            config: Configuración de procesamiento
            
        Returns:
            Diccionario con clasificación de componentes
        """
        try:
            self.logger.info("Starting hybrid model component classification")
            
            # Para modelos híbridos, la clasificación se delega al servicio EEMD
            # Este método se mantiene por compatibilidad con la interfaz
            classifications = {}
            
            self.logger.info("Hybrid model component classification completed")
            return classifications
            
        except Exception as e:
            self.logger.error(f"Hybrid model component classification error: {e}")
            raise
    
    def train_models(self, 
                    decomposition_result: Any, 
                    classifications: Dict[str, List[int]], 
                    config: ProcessingConfig) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Entrenar modelos híbridos SVR + SARIMAX.
        
        Args:
            decomposition_result: Resultado de descomposición
            classifications: Clasificación de componentes
            config: Configuración de procesamiento
            
        Returns:
            Tupla con modelos entrenados y métricas
        """
        try:
            self.logger.info("Starting hybrid model training")
            
            # Usar el método de entrenamiento existente
            if hasattr(decomposition_result, 'imfs') and decomposition_result.imfs is not None:
                # Simular una serie temporal para el entrenamiento
                time_series = pd.Series(decomposition_result.imfs[:, 0] if decomposition_result.imfs.shape[1] > 0 else [0])
                
                # Configurar parámetros de entrenamiento
                training_params = {
                    'svr_lags': config.svr_lags,
                    'svr_test_size': config.svr_test_size,
                    'sarimax_max_iter': config.sarimax_max_iter,
                    'sarimax_data_limit_years': config.sarimax_data_limit_years
                }
                
                # Entrenar modelos usando el método existente
                model_result = self.train_models_legacy(decomposition_result, time_series, config)
                
                # Convertir resultado a formato de interfaz
                trained_models = {
                    'svr_models': getattr(model_result, 'svr_models', {}),
                    'sarimax_models': getattr(model_result, 'sarimax_models', {}),
                    'scalers': getattr(model_result, 'scalers', {})
                }
                
                model_metrics = {
                    'training_time': getattr(model_result, 'training_time', 0.0),
                    'models_count': len(trained_models['svr_models']) + len(trained_models['sarimax_models'])
                }
                
            else:
                trained_models = {}
                model_metrics = {'training_time': 0.0, 'models_count': 0}
            
            self.logger.info("Hybrid model training completed")
            return trained_models, model_metrics
            
        except Exception as e:
            self.logger.error(f"Hybrid model training error: {e}")
            raise
    
    def generate_predictions(self, 
                           models: Dict[str, Any], 
                           decomposition_result: Any,
                           config: ProcessingConfig) -> Tuple[pd.Series, Optional[Tuple[pd.Series, pd.Series]]]:
        """
        Generar predicciones usando modelos híbridos.
        
        Args:
            models: Modelos entrenados
            decomposition_result: Resultado de descomposición
            config: Configuración de procesamiento
            
        Returns:
            Tupla con predicciones e intervalos de confianza
        """
        try:
            self.logger.info("Starting hybrid model prediction generation")
            
            # Para modelos híbridos, la generación de predicciones se delega al servicio de predicción
            # Este método se mantiene por compatibilidad con la interfaz
            predictions = pd.Series()
            confidence_intervals = None
            
            self.logger.info("Hybrid model prediction generation completed")
            return predictions, confidence_intervals
            
        except Exception as e:
            self.logger.error(f"Hybrid model prediction generation error: {e}")
            raise
    
    def evaluate_quality(self, 
                        input_data: pd.DataFrame, 
                        predictions: pd.Series, 
                        config: ProcessingConfig) -> Dict[str, float]:
        """
        Evaluar calidad de las predicciones de modelos híbridos.
        
        Args:
            input_data: Datos de entrada originales
            predictions: Predicciones generadas
            config: Configuración de procesamiento
            
        Returns:
            Diccionario con métricas de calidad
        """
        try:
            self.logger.info("Starting hybrid model quality evaluation")
            
            # Métricas específicas para modelos híbridos
            quality_metrics = {
                'hybrid_quality_score': 0.0,
                'svr_models_count': 0,
                'sarimax_models_count': 0,
                'training_quality': 0.0
            }
            
            self.logger.info("Hybrid model quality evaluation completed")
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Hybrid model quality evaluation error: {e}")
            raise
    
    def save_results(self, 
                    result: ProcessingResult, 
                    output_dir: Path, 
                    config: ProcessingConfig) -> Dict[str, str]:
        """
        Guardar resultados de modelos híbridos.
        
        Args:
            result: Resultado de procesamiento
            output_dir: Directorio de salida
            config: Configuración de procesamiento
            
        Returns:
            Diccionario con rutas de archivos guardados
        """
        try:
            self.logger.info("Starting hybrid model results saving")
            
            # Crear directorio de salida
            output_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = {}
            
            # Guardar modelos entrenados si están disponibles
            if hasattr(result, 'trained_models') and result.trained_models:
                models_file = output_dir / "trained_models.json"
                import json
                with open(models_file, 'w') as f:
                    json.dump({
                        'svr_models_count': len(result.trained_models.get('svr_models', {})),
                        'sarimax_models_count': len(result.trained_models.get('sarimax_models', {})),
                        'scalers_count': len(result.trained_models.get('scalers', {}))
                    }, f, indent=2)
                saved_files['trained_models'] = str(models_file)
            
            # Guardar configuración
            config_file = output_dir / "hybrid_model_config.json"
            import json
            with open(config_file, 'w') as f:
                json.dump(config.__dict__, f, indent=2, default=str)
            saved_files['config'] = str(config_file)
            
            self.logger.info(f"Hybrid model results saving completed: {len(saved_files)} files")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Hybrid model results saving error: {e}")
            raise
    
    def process_data(self, 
                    data: pd.DataFrame, 
                    config: Optional[ProcessingConfig] = None,
                    output_dir: Optional[Path] = None) -> ProcessingResult:
        """
        Procesar datos completos usando modelos híbridos.
        
        Args:
            data: DataFrame con datos de entrada
            config: Configuración de procesamiento (opcional)
            output_dir: Directorio de salida (opcional)
            
        Returns:
            Resultado del procesamiento con modelos híbridos
        """
        try:
            # Configurar configuración por defecto si no se proporciona
            if config is None:
                config = ProcessingConfig(target_column="value")
            
            # Validar datos
            if not self.validate_data(data):
                raise ValueError("Hybrid model data validation failed")
            
            # Preprocesar datos
            processed_data = self.preprocess_data(data, config)
            
            # Crear resultado
            result = ProcessingResult(
                input_data=data,
                config=config
            )
            
            # Guardar resultados si se especifica directorio
            if output_dir:
                saved_files = self.save_results(result, output_dir, config)
                result.output_files = saved_files
            
            result.success = True
            return result
            
        except Exception as e:
            self.logger.error(f"Hybrid model processing error: {e}")
            raise 