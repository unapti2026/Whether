"""
Unified Configuration System

Este módulo implementa el sistema de configuración unificada que conecta
las interfaces genéricas con el sistema existente, proporcionando una
gestión centralizada de configuraciones agnósticas a la variable.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from ..interfaces.variable_agnostic_interfaces import (
    ProcessingConfig, ProcessingResult, IVariableAgnosticConfigFactory,
    IVariableAgnosticValidator, IVariableAgnosticLogger,
    IVariableAgnosticMemoryManager
)

from .variable_config_factory import VariableConfigFactory
from .preset_config_factory import PresetConfigFactory

logger = logging.getLogger(__name__)


@dataclass
class UnifiedConfigPreset:
    """
    Preset de configuración unificada.
    
    Define configuraciones predefinidas para diferentes tipos de procesamiento
    sin hacer referencia específica a variables meteorológicas.
    """
    name: str
    description: str
    config: ProcessingConfig
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class UnifiedConfigurationFactory(IVariableAgnosticConfigFactory):
    """
    Fábrica de configuración unificada.
    
    Implementa la interfaz IVariableAgnosticConfigFactory para crear
    configuraciones agnósticas a la variable.
    """
    
    def __init__(self):
        """Inicializar la fábrica de configuración."""
        self.logger = logger
        self._presets: Dict[str, UnifiedConfigPreset] = {}
        self._load_default_presets()
    
    def _load_default_presets(self) -> None:
        """Cargar presets por defecto."""
        # Preset para procesamiento rápido
        fast_config = ProcessingConfig(
            target_column="value",
            eemd_ensembles=3,
            eemd_noise_factor=0.05,
            svr_lags=5,
            prediction_steps_ratio=0.1,
            memory_limit_mb=1024,
            enable_downsampling=True,
            downsampling_threshold=10000
        )
        
        fast_preset = UnifiedConfigPreset(
            name="fast",
            description="Configuración optimizada para procesamiento rápido",
            config=fast_config,
            tags=["fast", "optimized", "memory-efficient"]
        )
        
        # Preset para alta precisión
        high_precision_config = ProcessingConfig(
            target_column="value",
            eemd_ensembles=10,
            eemd_noise_factor=0.15,
            svr_lags=14,
            prediction_steps_ratio=0.3,
            memory_limit_mb=4096,
            enable_downsampling=False,
            cross_validation_folds=10
        )
        
        high_precision_preset = UnifiedConfigPreset(
            name="high_precision",
            description="Configuración optimizada para máxima precisión",
            config=high_precision_config,
            tags=["high-precision", "accurate", "comprehensive"]
        )
        
        # Preset para memoria limitada
        memory_efficient_config = ProcessingConfig(
            target_column="value",
            eemd_ensembles=3,
            eemd_noise_factor=0.05,
            svr_lags=3,
            prediction_steps_ratio=0.1,
            memory_limit_mb=512,
            enable_downsampling=True,
            downsampling_threshold=5000,
            save_intermediate=False
        )
        
        memory_efficient_preset = UnifiedConfigPreset(
            name="memory_efficient",
            description="Configuración optimizada para uso eficiente de memoria",
            config=memory_efficient_config,
            tags=["memory-efficient", "lightweight", "fast"]
        )
        
        # Preset para producción
        production_config = ProcessingConfig(
            target_column="value",
            eemd_ensembles=5,
            eemd_noise_factor=0.1,
            svr_lags=7,
            prediction_steps_ratio=0.2,
            memory_limit_mb=2048,
            enable_downsampling=True,
            downsampling_threshold=15000,
            confidence_level=0.95,
            output_formats=['png', 'csv', 'json']
        )
        
        production_preset = UnifiedConfigPreset(
            name="production",
            description="Configuración optimizada para entorno de producción",
            config=production_config,
            tags=["production", "balanced", "reliable"]
        )
        
        # Registrar presets
        self._presets = {
            "fast": fast_preset,
            "high_precision": high_precision_preset,
            "memory_efficient": memory_efficient_preset,
            "production": production_preset
        }
        
        self.logger.info(f"Loaded {len(self._presets)} default configuration presets")
    
    def create_default_config(self) -> ProcessingConfig:
        """
        Crear configuración por defecto.
        
        Returns:
            Configuración por defecto
        """
        return ProcessingConfig(target_column="value")
    
    def create_config_from_preset(self, preset_name: str) -> ProcessingConfig:
        """
        Crear configuración desde preset.
        
        Args:
            preset_name: Nombre del preset
            
        Returns:
            Configuración del preset
            
        Raises:
            ValueError: Si el preset no existe
        """
        if preset_name not in self._presets:
            available_presets = list(self._presets.keys())
            raise ValueError(f"Preset '{preset_name}' not found. Available presets: {available_presets}")
        
        preset = self._presets[preset_name]
        self.logger.info(f"Created configuration from preset: {preset_name}")
        return preset.config
    
    def create_adaptive_config(self, data: pd.DataFrame) -> ProcessingConfig:
        """
        Crear configuración adaptativa basada en los datos.
        
        Args:
            data: DataFrame con datos de entrada
            
        Returns:
            Configuración adaptativa
        """
        data_size = len(data)
        
        # Determinar configuración base según el tamaño de datos
        if data_size < 1000:
            base_config = self.create_config_from_preset("fast")
        elif data_size < 10000:
            base_config = self.create_config_from_preset("production")
        elif data_size < 50000:
            base_config = self.create_config_from_preset("high_precision")
        else:
            base_config = self.create_config_from_preset("memory_efficient")
        
        # Adaptar parámetros específicos
        if data_size > 20000:
            base_config.enable_downsampling = True
            base_config.downsampling_threshold = min(15000, data_size // 2)
        
        if data_size < 5000:
            base_config.eemd_ensembles = max(3, base_config.eemd_ensembles // 2)
            base_config.svr_lags = max(3, base_config.svr_lags // 2)
        
        # Ajustar límite de memoria
        estimated_memory = data_size * 8 * 2  # Estimación básica en bytes
        base_config.memory_limit_mb = max(512, min(4096, estimated_memory // (1024 * 1024)))
        
        self.logger.info(f"Created adaptive configuration for {data_size} data points")
        return base_config
    
    def validate_config(self, config: ProcessingConfig) -> bool:
        """
        Validar configuración.
        
        Args:
            config: Configuración a validar
            
        Returns:
            True si la configuración es válida
            
        Raises:
            ValueError: Si la configuración es inválida
        """
        try:
            # Validar parámetros básicos
            if not config.target_column:
                raise ValueError("target_column cannot be empty")
            
            if config.min_data_points <= 0:
                raise ValueError("min_data_points must be positive")
            
            if config.max_data_points <= config.min_data_points:
                raise ValueError("max_data_points must be greater than min_data_points")
            
            # Validar parámetros EEMD
            if config.eemd_ensembles < 1:
                raise ValueError("eemd_ensembles must be at least 1")
            
            if not 0 < config.eemd_noise_factor < 1:
                raise ValueError("eemd_noise_factor must be between 0 and 1")
            
            if config.eemd_max_imfs < 1:
                raise ValueError("eemd_max_imfs must be at least 1")
            
            # Validar parámetros de modelos
            if config.svr_lags < 1:
                raise ValueError("svr_lags must be at least 1")
            
            if not 0 < config.svr_test_size < 1:
                raise ValueError("svr_test_size must be between 0 and 1")
            
            if config.sarimax_max_iter < 1:
                raise ValueError("sarimax_max_iter must be at least 1")
            
            # Validar parámetros de predicción
            if not 0 < config.prediction_steps_ratio < 1:
                raise ValueError("prediction_steps_ratio must be between 0 and 1")
            
            if not 0 < config.confidence_level < 1:
                raise ValueError("confidence_level must be between 0 and 1")
            
            # Validar parámetros de memoria
            if config.memory_limit_mb < 100:
                raise ValueError("memory_limit_mb must be at least 100")
            
            if config.downsampling_threshold < 100:
                raise ValueError("downsampling_threshold must be at least 100")
            
            # Validar parámetros de validación
            if not 0 < config.validation_split < 1:
                raise ValueError("validation_split must be between 0 and 1")
            
            if config.cross_validation_folds < 2:
                raise ValueError("cross_validation_folds must be at least 2")
            
            # Validar parámetros de salida
            if config.plot_dpi < 72:
                raise ValueError("plot_dpi must be at least 72")
            
            if not config.output_formats:
                raise ValueError("output_formats cannot be empty")
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")
    
    def get_available_presets(self) -> Dict[str, str]:
        """
        Obtener presets disponibles.
        
        Returns:
            Diccionario con nombres y descripciones de presets
        """
        return {name: preset.description for name, preset in self._presets.items()}
    
    def add_custom_preset(self, preset: UnifiedConfigPreset) -> None:
        """
        Agregar preset personalizado.
        
        Args:
            preset: Preset personalizado
        """
        self._presets[preset.name] = preset
        self.logger.info(f"Added custom preset: {preset.name}")
    
    def remove_preset(self, preset_name: str) -> None:
        """
        Remover preset.
        
        Args:
            preset_name: Nombre del preset a remover
        """
        if preset_name in self._presets:
            del self._presets[preset_name]
            self.logger.info(f"Removed preset: {preset_name}")
        else:
            self.logger.warning(f"Preset '{preset_name}' not found for removal")
    
    def export_presets(self, file_path: Path) -> None:
        """
        Exportar presets a archivo.
        
        Args:
            file_path: Ruta del archivo
        """
        try:
            presets_data = {}
            for name, preset in self._presets.items():
                presets_data[name] = {
                    'description': preset.description,
                    'tags': preset.tags,
                    'config': asdict(preset.config)
                }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(presets_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported {len(self._presets)} presets to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export presets: {e}")
            raise
    
    def import_presets(self, file_path: Path) -> None:
        """
        Importar presets desde archivo.
        
        Args:
            file_path: Ruta del archivo
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                presets_data = json.load(f)
            
            for name, data in presets_data.items():
                config = ProcessingConfig(**data['config'])
                preset = UnifiedConfigPreset(
                    name=name,
                    description=data['description'],
                    config=config,
                    tags=data.get('tags', [])
                )
                self._presets[name] = preset
            
            self.logger.info(f"Imported {len(presets_data)} presets from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to import presets: {e}")
            raise


class UnifiedConfigurationValidator(IVariableAgnosticValidator):
    """
    Validador de configuración unificada.
    
    Implementa la interfaz IVariableAgnosticValidator para validar
    configuraciones y datos de manera agnóstica a la variable.
    """
    
    def __init__(self):
        """Inicializar el validador."""
        self.logger = logger
    
    def validate_config(self, config: ProcessingConfig) -> bool:
        """
        Validar configuración de procesamiento.
        
        Args:
            config: Configuración a validar
            
        Returns:
            True si la configuración es válida
        """
        try:
            factory = UnifiedConfigurationFactory()
            return factory.validate_config(config)
        except Exception as e:
            self.logger.error(f"Config validation failed: {e}")
            return False
    
    def validate_data_structure(self, data: pd.DataFrame, config: ProcessingConfig) -> bool:
        """
        Validar estructura de datos.
        
        Args:
            data: DataFrame a validar
            config: Configuración de procesamiento
            
        Returns:
            True si la estructura es válida
        """
        try:
            # Verificar que el DataFrame no esté vacío
            if data.empty:
                raise ValueError("DataFrame is empty")
            
            # Verificar columnas requeridas
            required_columns = [config.date_column, config.target_column]
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Verificar número de filas
            if len(data) < config.min_data_points:
                raise ValueError(f"Data has {len(data)} rows, minimum required: {config.min_data_points}")
            
            if len(data) > config.max_data_points:
                raise ValueError(f"Data has {len(data)} rows, maximum allowed: {config.max_data_points}")
            
            # Verificar tipos de datos
            if not pd.api.types.is_datetime64_any_dtype(data[config.date_column]):
                raise ValueError(f"Column '{config.date_column}' must be datetime type")
            
            if not pd.api.types.is_numeric_dtype(data[config.target_column]):
                raise ValueError(f"Column '{config.target_column}' must be numeric type")
            
            self.logger.info("Data structure validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data structure validation failed: {e}")
            return False
    
    def validate_data_quality(self, data: pd.DataFrame, config: ProcessingConfig) -> bool:
        """
        Validar calidad de datos.
        
        Args:
            data: DataFrame a validar
            config: Configuración de procesamiento
            
        Returns:
            True si la calidad es aceptable
        """
        try:
            target_series = data[config.target_column]
            
            # Verificar valores faltantes
            missing_ratio = target_series.isnull().sum() / len(target_series)
            if missing_ratio > 0.5:
                raise ValueError(f"Too many missing values: {missing_ratio:.2%}")
            
            # Verificar valores infinitos
            infinite_count = np.isinf(target_series).sum()
            if infinite_count > 0:
                raise ValueError(f"Found {infinite_count} infinite values")
            
            # Verificar varianza
            variance = target_series.var()
            if variance == 0:
                raise ValueError("Target variable has zero variance")
            
            # Verificar orden temporal
            date_series = data[config.date_column]
            if not date_series.is_monotonic_increasing:
                raise ValueError("Date column is not in chronological order")
            
            self.logger.info("Data quality validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data quality validation failed: {e}")
            return False


class UnifiedConfigurationLogger(IVariableAgnosticLogger):
    """
    Logger de configuración unificada.
    
    Implementa la interfaz IVariableAgnosticLogger para logging
    agnóstico a la variable.
    """
    
    def __init__(self):
        """Inicializar el logger."""
        self.logger = logger
    
    def log_processing_start(self, config: ProcessingConfig) -> None:
        """
        Registrar inicio de procesamiento.
        
        Args:
            config: Configuración de procesamiento
        """
        self.logger.info("🚀 Starting unified processing")
        self.logger.info(f"📊 Configuration: {config.target_column} target")
        self.logger.info(f"⚙️ EEMD ensembles: {config.eemd_ensembles}")
        self.logger.info(f"🧠 Memory limit: {config.memory_limit_mb} MB")
        self.logger.info(f"📈 Prediction steps ratio: {config.prediction_steps_ratio}")
    
    def log_processing_step(self, step_name: str, step_data: Dict[str, Any]) -> None:
        """
        Registrar paso de procesamiento.
        
        Args:
            step_name: Nombre del paso
            step_data: Datos del paso
        """
        self.logger.info(f"🔧 Step: {step_name}")
        for key, value in step_data.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"   {key}: {value}")
            elif isinstance(value, str):
                self.logger.info(f"   {key}: {value}")
            else:
                self.logger.info(f"   {key}: {type(value).__name__}")
    
    def log_processing_complete(self, result: ProcessingResult) -> None:
        """
        Registrar completación de procesamiento.
        
        Args:
            result: Resultado del procesamiento
        """
        if result.success:
            self.logger.info("✅ Processing completed successfully")
            self.logger.info(f"⏱️ Processing time: {result.processing_time:.2f}s")
            self.logger.info(f"🧠 Memory usage: {result.memory_usage_mb:.2f} MB")
            self.logger.info(f"📊 Quality score: {result.quality_score:.4f}")
        else:
            self.logger.error("❌ Processing failed")
            if result.error_message:
                self.logger.error(f"Error: {result.error_message}")
    
    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """
        Registrar error.
        
        Args:
            error: Excepción ocurrida
            context: Contexto del error
        """
        self.logger.error(f"❌ Error: {type(error).__name__}: {error}")
        for key, value in context.items():
            self.logger.error(f"   {key}: {value}")


class UnifiedConfigurationMemoryManager(IVariableAgnosticMemoryManager):
    """
    Gestor de memoria de configuración unificada.
    
    Implementa la interfaz IVariableAgnosticMemoryManager para gestión
    de memoria agnóstica a la variable.
    """
    
    def __init__(self):
        """Inicializar el gestor de memoria."""
        self.logger = logger
    
    def check_memory_usage(self) -> Dict[str, float]:
        """
        Verificar uso de memoria.
        
        Returns:
            Diccionario con información de memoria
        """
        try:
            import psutil
            
            memory_info = psutil.virtual_memory()
            return {
                'total_mb': memory_info.total / (1024 * 1024),
                'available_mb': memory_info.available / (1024 * 1024),
                'used_mb': memory_info.used / (1024 * 1024),
                'percent_used': memory_info.percent
            }
        except ImportError:
            self.logger.warning("psutil not available, using basic memory estimation")
            return {
                'total_mb': 8192,  # Estimación por defecto
                'available_mb': 4096,
                'used_mb': 4096,
                'percent_used': 50.0
            }
    
    def optimize_memory_config(self, data_size: int, config: ProcessingConfig) -> ProcessingConfig:
        """
        Optimizar configuración para memoria.
        
        Args:
            data_size: Tamaño de los datos
            config: Configuración original
            
        Returns:
            Configuración optimizada
        """
        memory_info = self.check_memory_usage()
        available_memory = memory_info['available_mb']
        
        # Crear copia de la configuración
        optimized_config = ProcessingConfig(
            target_column=config.target_column,
            date_column=config.date_column,
            min_data_points=config.min_data_points,
            max_data_points=config.max_data_points,
            eemd_ensembles=config.eemd_ensembles,
            eemd_noise_factor=config.eemd_noise_factor,
            eemd_sd_thresh_range=config.eemd_sd_thresh_range,
            eemd_max_imfs=config.eemd_max_imfs,
            eemd_quality_threshold=config.eemd_quality_threshold,
            svr_lags=config.svr_lags,
            svr_test_size=config.svr_test_size,
            sarimax_max_iter=config.sarimax_max_iter,
            sarimax_data_limit_years=config.sarimax_data_limit_years,
            prediction_steps_ratio=config.prediction_steps_ratio,
            confidence_level=config.confidence_level,
            memory_limit_mb=config.memory_limit_mb,
            enable_downsampling=config.enable_downsampling,
            downsampling_threshold=config.downsampling_threshold,
            validation_split=config.validation_split,
            cross_validation_folds=config.cross_validation_folds,
            output_formats=config.output_formats,
            plot_dpi=config.plot_dpi,
            save_intermediate=config.save_intermediate
        )
        
        # Optimizar según memoria disponible
        if available_memory < 1024:  # Menos de 1GB
            optimized_config.eemd_ensembles = max(2, optimized_config.eemd_ensembles // 2)
            optimized_config.enable_downsampling = True
            optimized_config.downsampling_threshold = min(5000, data_size // 4)
            optimized_config.save_intermediate = False
            optimized_config.plot_dpi = 150
            self.logger.info("Applied low-memory optimizations")
        
        elif available_memory < 2048:  # Menos de 2GB
            optimized_config.eemd_ensembles = max(3, optimized_config.eemd_ensembles)
            optimized_config.enable_downsampling = True
            optimized_config.downsampling_threshold = min(10000, data_size // 3)
            self.logger.info("Applied medium-memory optimizations")
        
        else:  # 2GB o más
            optimized_config.enable_downsampling = data_size > 20000
            self.logger.info("Applied high-memory optimizations")
        
        return optimized_config
    
    def cleanup_memory(self) -> None:
        """Limpiar memoria."""
        try:
            import gc
            gc.collect()
            self.logger.info("Memory cleanup completed")
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")
    
    def estimate_memory_requirements(self, data: pd.DataFrame, config: ProcessingConfig) -> float:
        """
        Estimar requerimientos de memoria.
        
        Args:
            data: DataFrame con datos
            config: Configuración de procesamiento
            
        Returns:
            Estimación de memoria en MB
        """
        data_size = len(data)
        num_features = len(data.columns)
        
        # Estimación básica
        base_memory = data_size * num_features * 8  # bytes
        
        # Memoria para EEMD
        eemd_memory = data_size * config.eemd_ensembles * config.eemd_max_imfs * 8
        
        # Memoria para modelos
        model_memory = data_size * config.svr_lags * 8 * 2  # SVR features
        
        # Memoria para predicciones
        prediction_steps = int(data_size * config.prediction_steps_ratio)
        prediction_memory = prediction_steps * config.eemd_max_imfs * 8
        
        # Memoria total estimada
        total_memory_bytes = base_memory + eemd_memory + model_memory + prediction_memory
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        
        self.logger.info(f"Estimated memory requirements: {total_memory_mb:.2f} MB")
        return total_memory_mb


class UnifiedConfigurationManager:
    """
    Gestor unificado de configuración que integra todos los factories.
    
    Este gestor proporciona una interfaz unificada para acceder a todos
    los factories de configuración (variable, preset, etc.).
    """
    
    def __init__(self):
        """Inicializar el gestor de configuración unificado."""
        self.logger = logger
        self.variable_factory = VariableConfigFactory()
        self.preset_factory = PresetConfigFactory()
        self.unified_factory = UnifiedConfigurationFactory()
        self.logger.info("UnifiedConfigurationManager initialized")
    
    def create_config_for_variable_and_preset(self, 
                                            variable_name: str,
                                            preset_name: str = 'development',
                                            custom_params: Optional[Dict[str, Any]] = None) -> ProcessingConfig:
        """
        Crear configuración combinando variable y preset.
        
        Args:
            variable_name: Nombre de la variable meteorológica
            preset_name: Nombre del preset de configuración
            custom_params: Parámetros personalizados adicionales
            
        Returns:
            Configuración de procesamiento combinada
        """
        try:
            # Crear configuración específica para la variable
            var_config = self.variable_factory.create_config_for_variable(
                variable_name, preset_name, custom_params
            )
            
            self.logger.info(f"Created combined configuration for variable '{variable_name}' with preset '{preset_name}'")
            return var_config
            
        except Exception as e:
            self.logger.error(f"Error creating combined configuration: {e}")
            raise
    
    def get_available_variables(self) -> List[str]:
        """
        Obtener variables disponibles.
        
        Returns:
            Lista de variables soportadas
        """
        return self.variable_factory.get_supported_variables()
    
    def get_available_presets(self) -> List[str]:
        """
        Obtener presets disponibles.
        
        Returns:
            Lista de presets disponibles
        """
        return list(self.preset_factory.get_all_presets().keys())
    
    def get_preset_recommendations(self, 
                                 performance_profile: str = "balanced",
                                 memory_profile: str = "moderate",
                                 accuracy_profile: str = "standard") -> List[str]:
        """
        Obtener recomendaciones de presets.
        
        Args:
            performance_profile: Perfil de rendimiento
            memory_profile: Perfil de memoria
            accuracy_profile: Perfil de precisión
            
        Returns:
            Lista de presets recomendados
        """
        return self.preset_factory.get_preset_recommendations(
            performance_profile, memory_profile, accuracy_profile
        )
    
    def validate_configuration(self, 
                             variable_name: str, 
                             preset_name: str) -> Tuple[bool, List[str]]:
        """
        Validar configuración combinada.
        
        Args:
            variable_name: Nombre de la variable
            preset_name: Nombre del preset
            
        Returns:
            Tupla con (es_válido, lista_de_errores)
        """
        errors = []
        
        # Validar variable
        if variable_name not in self.get_available_variables():
            errors.append(f"Variable '{variable_name}' not supported")
        
        # Validar preset
        preset_valid, preset_errors = self.preset_factory.validate_preset(preset_name)
        if not preset_valid:
            errors.extend(preset_errors)
        
        # Validar combinación
        if not errors:
            try:
                config = self.create_config_for_variable_and_preset(variable_name, preset_name)
                var_valid = self.variable_factory.validate_variable_config(variable_name, config)
                if not var_valid:
                    errors.append(f"Configuration validation failed for variable '{variable_name}'")
            except Exception as e:
                errors.append(f"Configuration creation failed: {e}")
        
        return len(errors) == 0, errors
    
    def export_configuration(self, 
                           variable_name: str,
                           preset_name: str,
                           export_path: Path) -> bool:
        """
        Exportar configuración a archivo.
        
        Args:
            variable_name: Nombre de la variable
            preset_name: Nombre del preset
            export_path: Ruta de exportación
            
        Returns:
            True si se exportó exitosamente
        """
        try:
            config = self.create_config_for_variable_and_preset(variable_name, preset_name)
            
            # Crear directorio si no existe
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Preparar datos para exportación
            export_data = {
                'variable_name': variable_name,
                'preset_name': preset_name,
                'configuration': asdict(config),
                'metadata': {
                    'created_by': 'UnifiedConfigurationManager',
                    'variable_info': self.variable_factory.get_variable_info(variable_name),
                    'preset_info': self.preset_factory.get_preset_metadata(preset_name)
                }
            }
            
            # Guardar según extensión
            if export_path.suffix.lower() == '.json':
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            else:
                self.logger.error(f"Unsupported export format: {export_path.suffix}")
                return False
            
            self.logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            return False
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de configuraciones disponibles.
        
        Returns:
            Diccionario con resumen de configuraciones
        """
        return {
            'available_variables': self.get_available_variables(),
            'available_presets': self.get_available_presets(),
            'variable_count': len(self.get_available_variables()),
            'preset_count': len(self.get_available_presets()),
            'preset_categories': list(set(
                metadata.category for metadata in self.preset_factory._preset_metadata.values()
            ))
        }
