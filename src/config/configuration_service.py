"""
Configuration Service

Este módulo provee un servicio unificado para la gestión de todas las configuraciones
del sistema de predicción meteorológica.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .prediction_config_factory import PredictionConfigFactory
from .adaptive_config import create_adaptive_config, AdaptiveConfig
from src.core.interfaces.prediction_strategy import PredictionConfig

logger = logging.getLogger(__name__)


class ConfigurationService:
    """
    Servicio unificado para la gestión de configuraciones.
    
    Este servicio centraliza:
    - Configuraciones de predicción
    - Configuraciones adaptativas
    - Validación de configuraciones
    - Gestión de presets
    """
    
    def __init__(self):
        """Inicializar el servicio de configuración."""
        self.logger = logger
        self._config_cache: Dict[str, PredictionConfig] = {}
        self._adaptive_config_cache: Dict[str, AdaptiveConfig] = {}
        
    def get_prediction_config(self, 
                            preset: str, 
                            variable_type: str, 
                            **overrides) -> PredictionConfig:
        """
        Obtener configuración de predicción.
        
        Args:
            preset: Nombre del preset de configuración
            variable_type: Tipo de variable meteorológica
            **overrides: Parámetros para sobrescribir
            
        Returns:
            PredictionConfig configurada
            
        Raises:
            ValueError: Si el preset no existe
        """
        cache_key = f"{preset}_{variable_type}_{hash(str(overrides))}"
        
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        try:
            config = PredictionConfigFactory.create_from_preset(
                preset, variable_type, **overrides
            )
            
            # Validar configuración
            PredictionConfigFactory.validate_config(config)
            
            # Cachear configuración
            self._config_cache[cache_key] = config
            
            self.logger.info(f"Created prediction config: {preset} for {variable_type}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to create prediction config: {e}")
            raise ValueError(f"Invalid configuration preset '{preset}': {e}")
    
    def get_adaptive_config(self, 
                          series_length: int, 
                          variable_type: str) -> AdaptiveConfig:
        """
        Obtener configuración adaptativa basada en el tamaño de la serie.
        
        Args:
            series_length: Longitud de la serie temporal
            variable_type: Tipo de variable meteorológica
            
        Returns:
            AdaptiveConfig optimizada
        """
        cache_key = f"{series_length}_{variable_type}"
        
        if cache_key in self._adaptive_config_cache:
            return self._adaptive_config_cache[cache_key]
        
        try:
            adaptive_config = create_adaptive_config(series_length, variable_type)
            
            # Cachear configuración
            self._adaptive_config_cache[cache_key] = adaptive_config
            
            self.logger.info(f"Created adaptive config for {variable_type}: {series_length} points")
            return adaptive_config
            
        except Exception as e:
            self.logger.error(f"Failed to create adaptive config: {e}")
            raise ValueError(f"Failed to create adaptive config: {e}")
    
    def get_available_presets(self) -> Dict[str, str]:
        """
        Obtener presets disponibles.
        
        Returns:
            Diccionario con presets disponibles
        """
        return PredictionConfigFactory.get_available_presets()
    
    def validate_config(self, config: PredictionConfig) -> bool:
        """
        Validar configuración.
        
        Args:
            config: Configuración a validar
            
        Returns:
            True si la configuración es válida
            
        Raises:
            ValueError: Si la configuración es inválida
        """
        return PredictionConfigFactory.validate_config(config)
    
    def clear_cache(self) -> None:
        """Limpiar caché de configuraciones."""
        self._config_cache.clear()
        self._adaptive_config_cache.clear()
        self.logger.info("Configuration cache cleared")
    
    def get_cache_info(self) -> Dict[str, int]:
        """
        Obtener información del caché.
        
        Returns:
            Diccionario con información del caché
        """
        return {
            'prediction_configs': len(self._config_cache),
            'adaptive_configs': len(self._adaptive_config_cache)
        }
