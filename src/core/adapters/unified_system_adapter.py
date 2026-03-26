"""
Unified System Adapter

Este módulo proporciona el adaptador principal que conecta el sistema de
configuración unificada con el sistema existente, permitiendo una transición
suave hacia las interfaces genéricas.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd

from ..interfaces.variable_agnostic_interfaces import (
    ProcessingConfig, ProcessingResult, IVariableAgnosticProcessor
)
from ..config.unified_configuration_system import (
    UnifiedConfigurationFactory, UnifiedConfigurationValidator,
    UnifiedConfigurationLogger, UnifiedConfigurationMemoryManager
)

logger = logging.getLogger(__name__)


class UnifiedSystemAdapter(IVariableAgnosticProcessor):
    """
    Adaptador principal del sistema unificado.
    
    Este adaptador implementa la interfaz IVariableAgnosticProcessor y
    conecta con el sistema existente para proporcionar funcionalidad
    agnóstica a la variable.
    """
    
    def __init__(self):
        """Inicializar el adaptador del sistema unificado."""
        self.logger = logger
        
        # Inicializar componentes del sistema unificado
        self.config_factory = UnifiedConfigurationFactory()
        self.validator = UnifiedConfigurationValidator()
        self.logger_service = UnifiedConfigurationLogger()
        self.memory_manager = UnifiedConfigurationMemoryManager()
        
        # Estado del procesamiento
        self._current_config: Optional[ProcessingConfig] = None
        self._current_data: Optional[pd.DataFrame] = None
        self._processing_start_time: Optional[float] = None
        
        self.logger.info("UnifiedSystemAdapter initialized")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validar datos de entrada genéricos.
        
        Args:
            data: DataFrame con datos temporales
            
        Returns:
            True si los datos son válidos
        """
        try:
            if self._current_config is None:
                # Crear configuración por defecto si no hay una configurada
                self._current_config = self.config_factory.create_default_config()
            
            # Validar estructura y calidad de datos
            structure_valid = self.validator.validate_data_structure(data, self._current_config)
            quality_valid = self.validator.validate_data_quality(data, self._current_config)
            
            if structure_valid and quality_valid:
                self.logger.info("Data validation passed")
                return True
            else:
                self.logger.error("Data validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return False
    
    def preprocess_data(self, data: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
        """
        Preprocesar datos genéricos.
        
        Args:
            data: DataFrame con datos originales
            config: Configuración de procesamiento
            
        Returns:
            DataFrame preprocesado
        """
        try:
            self.logger.info("Starting data preprocessing")
            
            # Crear copia de los datos
            processed_data = data.copy()
            
            # Asegurar que la columna de fecha sea datetime
            if config.date_column in processed_data.columns:
                processed_data[config.date_column] = pd.to_datetime(processed_data[config.date_column])
            
            # Ordenar por fecha
            if config.date_column in processed_data.columns:
                processed_data = processed_data.sort_values(config.date_column).reset_index(drop=True)
            
            # Manejar valores faltantes en la columna objetivo
            if config.target_column in processed_data.columns:
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
            
            self.logger.info(f"Data preprocessing completed: {len(processed_data)} points")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Data preprocessing error: {e}")
            raise
    
    def decompose_series(self, series: pd.Series, config: ProcessingConfig) -> Any:
        """
        Descomponer serie temporal genérica.
        
        Args:
            series: Serie temporal a descomponer
            config: Configuración de procesamiento
            
        Returns:
            Resultado de descomposición
        """
        try:
            self.logger.info("Starting series decomposition")
            
            # Importar el servicio EEMD existente
            from ...data.prediction.services.eemd_service import EEMDService
            
            # Crear instancia del servicio EEMD
            eemd_service = EEMDService()
            
            # Configurar parámetros EEMD
            eemd_params = {
                'ensembles': config.eemd_ensembles,
                'noise_factor': config.eemd_noise_factor,
                'sd_thresh_range': config.eemd_sd_thresh_range,
                'max_imfs': config.eemd_max_imfs,
                'quality_threshold': config.eemd_quality_threshold
            }
            
            # Realizar descomposición
            decomposition_result = eemd_service.decompose_series(series, **eemd_params)
            
            self.logger.info("Series decomposition completed")
            return decomposition_result
            
        except Exception as e:
            self.logger.error(f"Series decomposition error: {e}")
            raise
    
    def classify_components(self, decomposition_result: Any, config: ProcessingConfig) -> Dict[str, List[int]]:
        """
        Clasificar componentes de la descomposición.
        
        Args:
            decomposition_result: Resultado de descomposición
            config: Configuración de procesamiento
            
        Returns:
            Diccionario con clasificación de componentes
        """
        try:
            self.logger.info("Starting component classification")
            
            # Usar el método de clasificación existente del servicio EEMD
            if hasattr(decomposition_result, 'classify_imfs'):
                classifications = decomposition_result.classify_imfs()
            else:
                # Clasificación por defecto si no hay método específico
                num_imfs = decomposition_result.imfs.shape[1] if hasattr(decomposition_result, 'imfs') else 0
                classifications = {
                    'sarimax_imfs': [num_imfs // 2] if num_imfs > 0 else [],
                    'svr_imfs': list(range(1, min(4, num_imfs))) if num_imfs > 1 else [],
                    'extrapolation_imfs': list(range(max(4, num_imfs - 2), num_imfs)) if num_imfs > 4 else [],
                    'noise_imfs': [0] if num_imfs > 0 else []
                }
            
            self.logger.info(f"Component classification completed: {classifications}")
            return classifications
            
        except Exception as e:
            self.logger.error(f"Component classification error: {e}")
            raise
    
    def train_models(self, 
                    decomposition_result: Any, 
                    classifications: Dict[str, List[int]], 
                    config: ProcessingConfig) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Entrenar modelos genéricos.
        
        Args:
            decomposition_result: Resultado de descomposición
            classifications: Clasificación de componentes
            config: Configuración de procesamiento
            
        Returns:
            Tupla con modelos entrenados y métricas
        """
        try:
            self.logger.info("Starting model training")
            
            # Importar el servicio de modelos híbridos existente
            from ...data.prediction.services.hybrid_model_service import HybridModelService
            
            # Crear instancia del servicio
            model_service = HybridModelService()
            
            # Configurar parámetros de entrenamiento
            training_params = {
                'svr_lags': config.svr_lags,
                'svr_test_size': config.svr_test_size,
                'sarimax_max_iter': config.sarimax_max_iter,
                'sarimax_data_limit_years': config.sarimax_data_limit_years
            }
            
            # Entrenar modelos
            trained_models, model_metrics = model_service.train_models(
                decomposition_result, classifications, **training_params
            )
            
            self.logger.info("Model training completed")
            return trained_models, model_metrics
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            raise
    
    def generate_predictions(self, 
                           models: Dict[str, Any], 
                           decomposition_result: Any,
                           config: ProcessingConfig) -> Tuple[pd.Series, Optional[Tuple[pd.Series, pd.Series]]]:
        """
        Generar predicciones genéricas.
        
        Args:
            models: Modelos entrenados
            decomposition_result: Resultado de descomposición
            config: Configuración de procesamiento
            
        Returns:
            Tupla con predicciones e intervalos de confianza
        """
        try:
            self.logger.info("Starting prediction generation")
            
            # Importar el servicio de predicción existente
            from ...data.prediction.services.prediction_service import PredictionService
            
            # Crear instancia del servicio
            prediction_service = PredictionService()
            
            # Calcular pasos de predicción
            prediction_steps = int(len(self._current_data) * config.prediction_steps_ratio)
            
            # Configurar parámetros de predicción
            prediction_params = {
                'prediction_steps': prediction_steps,
                'confidence_level': config.confidence_level
            }
            
            # Generar predicciones
            predictions, confidence_intervals = prediction_service.generate_predictions(
                models, decomposition_result, **prediction_params
            )
            
            self.logger.info(f"Prediction generation completed: {len(predictions)} steps")
            return predictions, confidence_intervals
            
        except Exception as e:
            self.logger.error(f"Prediction generation error: {e}")
            raise
    
    def evaluate_quality(self, 
                        input_data: pd.DataFrame, 
                        predictions: pd.Series, 
                        config: ProcessingConfig) -> Dict[str, float]:
        """
        Evaluar calidad de las predicciones.
        
        Args:
            input_data: Datos de entrada originales
            predictions: Predicciones generadas
            config: Configuración de procesamiento
            
        Returns:
            Diccionario con métricas de calidad
        """
        try:
            self.logger.info("Starting quality evaluation")
            
            # Calcular métricas básicas
            target_series = input_data[config.target_column]
            
            # Métricas de consistencia
            mean_consistency = abs(predictions.mean() - target_series.mean()) / target_series.std()
            
            # Métricas de tendencia
            target_trend = target_series.diff().mean()
            prediction_trend = predictions.diff().mean()
            trend_consistency = abs(prediction_trend - target_trend) / abs(target_trend) if target_trend != 0 else 0
            
            # Métricas de diversidad
            diversity_score = predictions.std() / target_series.std()
            
            # Calcular score de calidad general
            quality_score = max(0, 1 - (mean_consistency + trend_consistency) / 2)
            
            quality_metrics = {
                'mean_consistency': mean_consistency,
                'trend_consistency': trend_consistency,
                'diversity_score': diversity_score,
                'quality_score': quality_score,
                'prediction_length': len(predictions),
                'target_mean': target_series.mean(),
                'prediction_mean': predictions.mean(),
                'target_std': target_series.std(),
                'prediction_std': predictions.std()
            }
            
            self.logger.info(f"Quality evaluation completed: score = {quality_score:.4f}")
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Quality evaluation error: {e}")
            raise
    
    def save_results(self, 
                    result: ProcessingResult, 
                    output_dir: Path, 
                    config: ProcessingConfig) -> Dict[str, str]:
        """
        Guardar resultados genéricos.
        
        Args:
            result: Resultado de procesamiento
            output_dir: Directorio de salida
            config: Configuración de procesamiento
            
        Returns:
            Diccionario con rutas de archivos guardados
        """
        try:
            self.logger.info("Starting results saving")
            
            # Crear directorio de salida
            output_dir.mkdir(parents=True, exist_ok=True)
            
            saved_files = {}
            
            # Guardar predicciones como CSV
            if result.predictions is not None:
                predictions_file = output_dir / "predictions.csv"
                result.predictions.to_csv(predictions_file)
                saved_files['predictions'] = str(predictions_file)
            
            # Guardar configuración
            config_file = output_dir / "config.json"
            import json
            with open(config_file, 'w') as f:
                json.dump(config.__dict__, f, indent=2, default=str)
            saved_files['config'] = str(config_file)
            
            # Guardar métricas de calidad
            if result.prediction_metrics:
                metrics_file = output_dir / "quality_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(result.prediction_metrics, f, indent=2)
                saved_files['metrics'] = str(metrics_file)
            
            # Guardar metadatos del procesamiento
            metadata = {
                'processing_time': result.processing_time,
                'memory_usage_mb': result.memory_usage_mb,
                'quality_score': result.quality_score,
                'success': result.success,
                'error_message': result.error_message
            }
            
            metadata_file = output_dir / "processing_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            saved_files['metadata'] = str(metadata_file)
            
            self.logger.info(f"Results saving completed: {len(saved_files)} files")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Results saving error: {e}")
            raise
    
    def process_data(self, 
                    data: pd.DataFrame, 
                    config: Optional[ProcessingConfig] = None,
                    output_dir: Optional[Path] = None) -> ProcessingResult:
        """
        Procesar datos completos usando el sistema unificado.
        
        Args:
            data: DataFrame con datos de entrada
            config: Configuración de procesamiento (opcional)
            output_dir: Directorio de salida (opcional)
            
        Returns:
            Resultado del procesamiento
        """
        try:
            # Inicializar tiempo de procesamiento
            self._processing_start_time = time.time()
            
            # Configurar configuración
            if config is None:
                config = self.config_factory.create_adaptive_config(data)
            
            self._current_config = config
            self._current_data = data
            
            # Registrar inicio
            self.logger_service.log_processing_start(config)
            
            # Crear resultado
            result = ProcessingResult(
                input_data=data,
                config=config
            )
            
            try:
                # Paso 1: Validar datos
                if not self.validate_data(data):
                    raise ValueError("Data validation failed")
                
                # Paso 2: Preprocesar datos
                processed_data = self.preprocess_data(data, config)
                self.logger_service.log_processing_step("Data Preprocessing", {
                    'original_size': len(data),
                    'processed_size': len(processed_data)
                })
                
                # Paso 3: Descomponer serie
                target_series = processed_data[config.target_column]
                decomposition_result = self.decompose_series(target_series, config)
                result.eemd_result = decomposition_result
                
                self.logger_service.log_processing_step("Series Decomposition", {
                    'imfs_count': decomposition_result.imfs.shape[1] if hasattr(decomposition_result, 'imfs') else 0
                })
                
                # Paso 4: Clasificar componentes
                classifications = self.classify_components(decomposition_result, config)
                result.imf_classifications = classifications
                
                self.logger_service.log_processing_step("Component Classification", {
                    'sarimax_count': len(classifications.get('sarimax_imfs', [])),
                    'svr_count': len(classifications.get('svr_imfs', [])),
                    'extrapolation_count': len(classifications.get('extrapolation_imfs', []))
                })
                
                # Paso 5: Entrenar modelos
                trained_models, model_metrics = self.train_models(decomposition_result, classifications, config)
                result.trained_models = trained_models
                result.model_metrics = model_metrics
                
                self.logger_service.log_processing_step("Model Training", {
                    'models_count': len(trained_models),
                    'training_time': model_metrics.get('training_time', 0)
                })
                
                # Paso 6: Generar predicciones
                predictions, confidence_intervals = self.generate_predictions(trained_models, decomposition_result, config)
                result.predictions = predictions
                result.confidence_intervals = confidence_intervals
                
                self.logger_service.log_processing_step("Prediction Generation", {
                    'prediction_steps': len(predictions)
                })
                
                # Paso 7: Evaluar calidad
                quality_metrics = self.evaluate_quality(data, predictions, config)
                result.prediction_metrics = quality_metrics
                result.quality_score = quality_metrics.get('quality_score', 0.0)
                
                # Paso 8: Guardar resultados
                if output_dir:
                    saved_files = self.save_results(result, output_dir, config)
                    result.output_files = saved_files
                
                # Marcar como exitoso
                result.success = True
                
            except Exception as e:
                result.success = False
                result.error_message = str(e)
                self.logger_service.log_error(e, {'config': config.__dict__})
                raise
            
            finally:
                # Calcular tiempo de procesamiento
                if self._processing_start_time:
                    result.processing_time = time.time() - self._processing_start_time
                
                # Calcular uso de memoria
                memory_info = self.memory_manager.check_memory_usage()
                result.memory_usage_mb = memory_info.get('used_mb', 0.0)
                
                # Limpiar memoria
                self.memory_manager.cleanup_memory()
                
                # Registrar completación
                self.logger_service.log_processing_complete(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            raise
    
    def get_available_presets(self) -> Dict[str, str]:
        """
        Obtener presets disponibles.
        
        Returns:
            Diccionario con presets disponibles
        """
        return self.config_factory.get_available_presets()
    
    def create_config_from_preset(self, preset_name: str) -> ProcessingConfig:
        """
        Crear configuración desde preset.
        
        Args:
            preset_name: Nombre del preset
            
        Returns:
            Configuración del preset
        """
        return self.config_factory.create_config_from_preset(preset_name)
    
    def estimate_memory_requirements(self, data: pd.DataFrame, config: ProcessingConfig) -> float:
        """
        Estimar requerimientos de memoria.
        
        Args:
            data: DataFrame con datos
            config: Configuración de procesamiento
            
        Returns:
            Estimación de memoria en MB
        """
        return self.memory_manager.estimate_memory_requirements(data, config)
