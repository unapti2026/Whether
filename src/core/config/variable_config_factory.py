"""
Variable Configuration Factory

Este módulo implementa un factory especializado para la configuración
de variables meteorológicas específicas, proporcionando configuraciones
optimizadas para diferentes tipos de variables (temperatura, precipitación, etc.).
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

from ..interfaces.variable_agnostic_interfaces import (
    ProcessingConfig, 
    IVariableAgnosticConfigFactory
)

logger = logging.getLogger(__name__)


@dataclass
class VariableConfig:
    """
    Configuración específica para una variable meteorológica.
    
    Attributes:
        variable_name: Nombre de la variable (temp_max, temp_min, precipitation, etc.)
        variable_type: Tipo de variable (temperature, precipitation, humidity, etc.)
        target_column: Nombre de la columna objetivo en los datos
        unit: Unidad de medida de la variable
        min_value: Valor mínimo esperado
        max_value: Valor máximo esperado
        seasonal_patterns: Patrones estacionales esperados
        processing_priority: Prioridad de procesamiento (1-10)
        quality_thresholds: Umbrales de calidad específicos para la variable
        model_preferences: Preferencias de modelos para la variable
    """
    variable_name: str
    variable_type: str
    target_column: str
    unit: str
    min_value: float
    max_value: float
    seasonal_patterns: List[str] = field(default_factory=list)
    processing_priority: int = 5
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    model_preferences: Dict[str, Any] = field(default_factory=dict)


class VariableConfigFactory:
    """
    Factory especializado para la configuración de variables meteorológicas.
    
    Este factory proporciona configuraciones optimizadas para diferentes
    tipos de variables meteorológicas, adaptando parámetros específicos
    según las características de cada variable.
    """
    
    def __init__(self):
        """Inicializar el factory de configuración de variables."""
        self.logger = logger
        self._variable_configs = self._initialize_variable_configs()
        self.logger.info("VariableConfigFactory initialized")
    
    def _initialize_variable_configs(self) -> Dict[str, VariableConfig]:
        """
        Inicializar configuraciones predefinidas para variables meteorológicas.
        
        Returns:
            Diccionario con configuraciones de variables
        """
        configs = {}
        
        # Configuración para temperatura máxima
        configs['temp_max'] = VariableConfig(
            variable_name='temp_max',
            variable_type='temperature',
            target_column='Temperatura',
            unit='°C',
            min_value=-50.0,
            max_value=60.0,
            seasonal_patterns=['annual', 'daily'],
            processing_priority=8,
            quality_thresholds={
                'reconstruction_error': 2.0,
                'orthogonality_score': 0.1,
                'seasonality_strength': 0.7
            },
            model_preferences={
                'svr_kernel': 'rbf',
                'sarimax_order': (1, 1, 0),
                'eemd_ensembles': 20,
                'eemd_sd_thresh': [0.05, 0.1, 0.15, 0.2]
            }
        )
        
        # Configuración para temperatura mínima
        configs['temp_min'] = VariableConfig(
            variable_name='temp_min',
            variable_type='temperature',
            target_column='Temperatura',
            unit='°C',
            min_value=-60.0,
            max_value=50.0,
            seasonal_patterns=['annual', 'daily'],
            processing_priority=8,
            quality_thresholds={
                'reconstruction_error': 2.0,
                'orthogonality_score': 0.1,
                'seasonality_strength': 0.7
            },
            model_preferences={
                'svr_kernel': 'rbf',
                'sarimax_order': (1, 1, 0),
                'eemd_ensembles': 20,
                'eemd_sd_thresh': [0.05, 0.1, 0.15, 0.2]
            }
        )
        
        # Configuración para precipitación
        configs['precipitation'] = VariableConfig(
            variable_name='precipitation',
            variable_type='precipitation',
            target_column='Precipitacion',
            unit='mm',
            min_value=0.0,
            max_value=500.0,
            seasonal_patterns=['annual', 'monthly'],
            processing_priority=7,
            quality_thresholds={
                'reconstruction_error': 5.0,
                'orthogonality_score': 0.15,
                'seasonality_strength': 0.6
            },
            model_preferences={
                'svr_kernel': 'poly',
                'sarimax_order': (2, 1, 1),
                'eemd_ensembles': 25,
                'eemd_sd_thresh': [0.1, 0.15, 0.2, 0.25]
            }
        )
        
        # Configuración para humedad relativa
        configs['humidity'] = VariableConfig(
            variable_name='humidity',
            variable_type='humidity',
            target_column='Humedad',
            unit='%',
            min_value=0.0,
            max_value=100.0,
            seasonal_patterns=['annual', 'daily'],
            processing_priority=6,
            quality_thresholds={
                'reconstruction_error': 8.0,
                'orthogonality_score': 0.2,
                'seasonality_strength': 0.5
            },
            model_preferences={
                'svr_kernel': 'rbf',
                'sarimax_order': (1, 1, 1),
                'eemd_ensembles': 15,
                'eemd_sd_thresh': [0.1, 0.15, 0.2]
            }
        )
        
        # Configuración para presión atmosférica
        configs['pressure'] = VariableConfig(
            variable_name='pressure',
            variable_type='pressure',
            target_column='Presion',
            unit='hPa',
            min_value=800.0,
            max_value=1100.0,
            seasonal_patterns=['annual', 'daily'],
            processing_priority=5,
            quality_thresholds={
                'reconstruction_error': 3.0,
                'orthogonality_score': 0.12,
                'seasonality_strength': 0.4
            },
            model_preferences={
                'svr_kernel': 'linear',
                'sarimax_order': (1, 1, 0),
                'eemd_ensembles': 18,
                'eemd_sd_thresh': [0.08, 0.12, 0.16]
            }
        )
        
        # Configuración para velocidad del viento
        configs['wind_speed'] = VariableConfig(
            variable_name='wind_speed',
            variable_type='wind',
            target_column='Velocidad_Viento',
            unit='m/s',
            min_value=0.0,
            max_value=50.0,
            seasonal_patterns=['annual', 'daily'],
            processing_priority=4,
            quality_thresholds={
                'reconstruction_error': 2.5,
                'orthogonality_score': 0.18,
                'seasonality_strength': 0.3
            },
            model_preferences={
                'svr_kernel': 'rbf',
                'sarimax_order': (2, 1, 1),
                'eemd_ensembles': 22,
                'eemd_sd_thresh': [0.1, 0.15, 0.2, 0.25]
            }
        )
        
        return configs
    
    def create_config_for_variable(self, 
                                 variable_name: str, 
                                 preset_name: str = 'development',
                                 custom_params: Optional[Dict[str, Any]] = None) -> ProcessingConfig:
        """
        Crear configuración específica para una variable meteorológica.
        
        Args:
            variable_name: Nombre de la variable (temp_max, temp_min, etc.)
            preset_name: Nombre del preset de configuración
            custom_params: Parámetros personalizados adicionales
            
        Returns:
            Configuración de procesamiento adaptada a la variable
            
        Raises:
            ValueError: Si la variable no está soportada
        """
        try:
            # Verificar que la variable esté soportada
            if variable_name not in self._variable_configs:
                supported_vars = list(self._variable_configs.keys())
                raise ValueError(f"Variable '{variable_name}' not supported. Supported variables: {supported_vars}")
            
            # Obtener configuración base de la variable
            var_config = self._variable_configs[variable_name]
            
            # Crear configuración base usando valores por defecto
            base_config = ProcessingConfig(
                target_column=var_config.target_column,
                prediction_steps=30,
                confidence_level=0.95,
                eemd_ensembles=20,
                eemd_sd_thresh_values=[0.05, 0.1, 0.15, 0.2],
                eemd_noise_factor=0.05,
                svr_kernel='rbf',
                svr_lags=7,
                svr_test_size=0.2,
                sarimax_order=(1, 1, 0),
                sarimax_max_iter=100,
                sarimax_data_limit_years=2,
                enable_downsampling=True,
                downsampling_threshold=15000,
                memory_cleanup=True,
                force_garbage_collection=True
            )
            
            # Adaptar configuración a la variable específica
            adapted_config = self._adapt_config_for_variable(base_config, var_config)
            
            # Aplicar parámetros personalizados si se proporcionan
            if custom_params:
                adapted_config = self._apply_custom_params(adapted_config, custom_params)
            
            self.logger.info(f"Created configuration for variable '{variable_name}' with preset '{preset_name}'")
            return adapted_config
            
        except Exception as e:
            self.logger.error(f"Error creating configuration for variable '{variable_name}': {e}")
            raise
    
    def _adapt_config_for_variable(self, 
                                 base_config: ProcessingConfig, 
                                 var_config: VariableConfig) -> ProcessingConfig:
        """
        Adaptar configuración base a una variable específica.
        
        Args:
            base_config: Configuración base
            var_config: Configuración específica de la variable
            
        Returns:
            Configuración adaptada
        """
        # Crear nueva configuración con valores adaptados
        adapted_config = ProcessingConfig(
            # Parámetros base
            target_column=var_config.target_column,
            prediction_steps=base_config.prediction_steps,
            confidence_level=base_config.confidence_level,
            
            # Parámetros de EEMD adaptados
            eemd_ensembles=var_config.model_preferences.get('eemd_ensembles', base_config.eemd_ensembles),
            eemd_sd_thresh_values=var_config.model_preferences.get('eemd_sd_thresh', base_config.eemd_sd_thresh_values),
            eemd_noise_factor=base_config.eemd_noise_factor,
            
            # Parámetros de SVR adaptados
            svr_kernel=var_config.model_preferences.get('svr_kernel', base_config.svr_kernel),
            svr_lags=base_config.svr_lags,
            svr_test_size=base_config.svr_test_size,
            
            # Parámetros de SARIMAX adaptados
            sarimax_order=var_config.model_preferences.get('sarimax_order', base_config.sarimax_order),
            sarimax_max_iter=base_config.sarimax_max_iter,
            sarimax_data_limit_years=base_config.sarimax_data_limit_years,
            
            # Parámetros de memoria
            enable_downsampling=base_config.enable_downsampling,
            downsampling_threshold=base_config.downsampling_threshold,
            memory_cleanup=base_config.memory_cleanup,
            force_garbage_collection=base_config.force_garbage_collection,
            
            # Umbrales de calidad adaptados
            quality_thresholds=var_config.quality_thresholds,
            
            # Metadatos de la variable
            variable_name=var_config.variable_name,
            variable_type=var_config.variable_type,
            unit=var_config.unit,
            min_value=var_config.min_value,
            max_value=var_config.max_value,
            seasonal_patterns=var_config.seasonal_patterns,
            processing_priority=var_config.processing_priority
        )
        
        return adapted_config
    
    def _apply_custom_params(self, 
                           config: ProcessingConfig, 
                           custom_params: Dict[str, Any]) -> ProcessingConfig:
        """
        Aplicar parámetros personalizados a la configuración.
        
        Args:
            config: Configuración base
            custom_params: Parámetros personalizados
            
        Returns:
            Configuración con parámetros personalizados aplicados
        """
        # Crear copia de la configuración
        updated_config = ProcessingConfig(
            target_column=custom_params.get('target_column', config.target_column),
            prediction_steps=custom_params.get('prediction_steps', config.prediction_steps),
            confidence_level=custom_params.get('confidence_level', config.confidence_level),
            
            # EEMD parameters
            eemd_ensembles=custom_params.get('eemd_ensembles', config.eemd_ensembles),
            eemd_sd_thresh_values=custom_params.get('eemd_sd_thresh_values', config.eemd_sd_thresh_values),
            eemd_noise_factor=custom_params.get('eemd_noise_factor', config.eemd_noise_factor),
            
            # SVR parameters
            svr_kernel=custom_params.get('svr_kernel', config.svr_kernel),
            svr_lags=custom_params.get('svr_lags', config.svr_lags),
            svr_test_size=custom_params.get('svr_test_size', config.svr_test_size),
            
            # SARIMAX parameters
            sarimax_order=custom_params.get('sarimax_order', config.sarimax_order),
            sarimax_max_iter=custom_params.get('sarimax_max_iter', config.sarimax_max_iter),
            sarimax_data_limit_years=custom_params.get('sarimax_data_limit_years', config.sarimax_data_limit_years),
            
            # Memory parameters
            enable_downsampling=custom_params.get('enable_downsampling', config.enable_downsampling),
            downsampling_threshold=custom_params.get('downsampling_threshold', config.downsampling_threshold),
            memory_cleanup=custom_params.get('memory_cleanup', config.memory_cleanup),
            force_garbage_collection=custom_params.get('force_garbage_collection', config.force_garbage_collection),
            
            # Quality thresholds
            quality_thresholds=custom_params.get('quality_thresholds', config.quality_thresholds),
            
            # Variable metadata
            variable_name=custom_params.get('variable_name', config.variable_name),
            variable_type=custom_params.get('variable_type', config.variable_type),
            unit=custom_params.get('unit', config.unit),
            min_value=custom_params.get('min_value', config.min_value),
            max_value=custom_params.get('max_value', config.max_value),
            seasonal_patterns=custom_params.get('seasonal_patterns', config.seasonal_patterns),
            processing_priority=custom_params.get('processing_priority', config.processing_priority)
        )
        
        return updated_config
    
    def get_supported_variables(self) -> List[str]:
        """
        Obtener lista de variables soportadas.
        
        Returns:
            Lista de nombres de variables soportadas
        """
        return list(self._variable_configs.keys())
    
    def get_variable_info(self, variable_name: str) -> Optional[VariableConfig]:
        """
        Obtener información detallada de una variable.
        
        Args:
            variable_name: Nombre de la variable
            
        Returns:
            Configuración de la variable o None si no existe
        """
        return self._variable_configs.get(variable_name)
    
    def validate_variable_config(self, variable_name: str, config: ProcessingConfig) -> bool:
        """
        Validar configuración para una variable específica.
        
        Args:
            variable_name: Nombre de la variable
            config: Configuración a validar
            
        Returns:
            True si la configuración es válida
        """
        try:
            var_config = self._variable_configs.get(variable_name)
            if not var_config:
                self.logger.error(f"Variable '{variable_name}' not found")
                return False
            
            # Validar valores mínimos y máximos
            if hasattr(config, 'min_value') and config.min_value < var_config.min_value:
                self.logger.warning(f"Min value {config.min_value} below recommended {var_config.min_value}")
            
            if hasattr(config, 'max_value') and config.max_value > var_config.max_value:
                self.logger.warning(f"Max value {config.max_value} above recommended {var_config.max_value}")
            
            # Validar columna objetivo
            if config.target_column != var_config.target_column:
                self.logger.warning(f"Target column mismatch: expected '{var_config.target_column}', got '{config.target_column}'")
            
            self.logger.info(f"Configuration validation passed for variable '{variable_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed for variable '{variable_name}': {e}")
            return False
