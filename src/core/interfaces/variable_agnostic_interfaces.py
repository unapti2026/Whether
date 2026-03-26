"""
Variable-Agnostic Interfaces

Este módulo define las interfaces base que son completamente independientes
del tipo de variable meteorológica, permitiendo que el sistema sea genérico
y reutilizable para cualquier tipo de datos temporales.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    """
    Configuración genérica de procesamiento independiente de la variable.
    
    Esta clase contiene todos los parámetros necesarios para el procesamiento
    sin hacer referencia específica a ningún tipo de variable meteorológica.
    """
    # Configuración de datos
    target_column: str
    date_column: str = 'Fecha'
    min_data_points: int = 100
    max_data_points: int = 50000
    
    # Configuración EEMD
    eemd_ensembles: int = 5
    eemd_noise_factor: float = 0.1
    eemd_sd_thresh_range: Tuple[float, float] = (0.1, 0.15)
    eemd_max_imfs: int = 10
    eemd_quality_threshold: float = 0.5
    
    # Configuración de modelos
    svr_lags: int = 7
    svr_test_size: float = 0.2
    sarimax_max_iter: int = 30
    sarimax_data_limit_years: int = 2
    
    # Configuración de predicción
    prediction_steps_ratio: float = 0.2
    confidence_level: float = 0.95
    
    # Configuración de memoria
    memory_limit_mb: int = 2048
    enable_downsampling: bool = True
    downsampling_threshold: int = 15000
    
    # Configuración de validación
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    # Configuración de salida
    output_formats: List[str] = None
    plot_dpi: int = 300
    save_intermediate: bool = False
    
    def __post_init__(self):
        """Validar y establecer valores por defecto."""
        if self.output_formats is None:
            self.output_formats = ['png', 'csv', 'json']
        
        # Validar rangos
        if not 0 < self.eemd_noise_factor < 1:
            raise ValueError("eemd_noise_factor must be between 0 and 1")
        
        if not 0 < self.prediction_steps_ratio < 1:
            raise ValueError("prediction_steps_ratio must be between 0 and 1")
        
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")


@dataclass
class ProcessingResult:
    """
    Resultado genérico de procesamiento.
    
    Esta clase encapsula todos los resultados del procesamiento sin hacer
    referencia específica al tipo de variable meteorológica.
    """
    # Datos de entrada
    input_data: pd.DataFrame
    config: ProcessingConfig
    
    # Resultados EEMD
    eemd_result: Optional[Any] = None
    imf_classifications: Optional[Dict[str, List[int]]] = None
    
    # Resultados de modelos
    trained_models: Optional[Dict[str, Any]] = None
    model_metrics: Optional[Dict[str, float]] = None
    
    # Resultados de predicción
    predictions: Optional[pd.Series] = None
    confidence_intervals: Optional[Tuple[pd.Series, pd.Series]] = None
    prediction_metrics: Optional[Dict[str, float]] = None
    
    # Metadatos
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    quality_score: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    
    # Archivos generados
    output_files: Optional[Dict[str, str]] = None


class IVariableAgnosticProcessor(ABC):
    """
    Interfaz base para procesadores agnósticos a la variable.
    
    Esta interfaz define los métodos que cualquier procesador debe implementar
    sin hacer referencia específica al tipo de variable meteorológica.
    """
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validar datos de entrada genéricos.
        
        Args:
            data: DataFrame con datos temporales
            
        Returns:
            True si los datos son válidos
            
        Raises:
            ValueError: Si los datos no son válidos
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame, config: ProcessingConfig) -> pd.DataFrame:
        """
        Preprocesar datos genéricos.
        
        Args:
            data: DataFrame con datos originales
            config: Configuración de procesamiento
            
        Returns:
            DataFrame preprocesado
        """
        pass
    
    @abstractmethod
    def decompose_series(self, series: pd.Series, config: ProcessingConfig) -> Any:
        """
        Descomponer serie temporal genérica.
        
        Args:
            series: Serie temporal a descomponer
            config: Configuración de procesamiento
            
        Returns:
            Resultado de descomposición
        """
        pass
    
    @abstractmethod
    def classify_components(self, decomposition_result: Any, config: ProcessingConfig) -> Dict[str, List[int]]:
        """
        Clasificar componentes de la descomposición.
        
        Args:
            decomposition_result: Resultado de descomposición
            config: Configuración de procesamiento
            
        Returns:
            Diccionario con clasificación de componentes
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass


class IVariableAgnosticValidator(ABC):
    """
    Interfaz para validadores agnósticos a la variable.
    """
    
    @abstractmethod
    def validate_config(self, config: ProcessingConfig) -> bool:
        """Validar configuración de procesamiento."""
        pass
    
    @abstractmethod
    def validate_data_structure(self, data: pd.DataFrame, config: ProcessingConfig) -> bool:
        """Validar estructura de datos."""
        pass
    
    @abstractmethod
    def validate_data_quality(self, data: pd.DataFrame, config: ProcessingConfig) -> bool:
        """Validar calidad de datos."""
        pass


class IVariableAgnosticVisualizer(ABC):
    """
    Interfaz para visualizadores agnósticos a la variable.
    """
    
    @abstractmethod
    def create_decomposition_plots(self, 
                                 decomposition_result: Any, 
                                 output_dir: Path, 
                                 config: ProcessingConfig) -> Dict[str, str]:
        """Crear plots de descomposición."""
        pass
    
    @abstractmethod
    def create_prediction_plots(self, 
                              input_data: pd.DataFrame, 
                              predictions: pd.Series, 
                              output_dir: Path, 
                              config: ProcessingConfig) -> Dict[str, str]:
        """Crear plots de predicciones."""
        pass
    
    @abstractmethod
    def create_quality_plots(self, 
                           quality_metrics: Dict[str, float], 
                           output_dir: Path, 
                           config: ProcessingConfig) -> Dict[str, str]:
        """Crear plots de calidad."""
        pass


class IVariableAgnosticConfigFactory(ABC):
    """
    Interfaz para fábricas de configuración agnósticas a la variable.
    """
    
    @abstractmethod
    def create_default_config(self) -> ProcessingConfig:
        """Crear configuración por defecto."""
        pass
    
    @abstractmethod
    def create_config_from_preset(self, preset_name: str) -> ProcessingConfig:
        """Crear configuración desde preset."""
        pass
    
    @abstractmethod
    def create_adaptive_config(self, data: pd.DataFrame) -> ProcessingConfig:
        """Crear configuración adaptativa basada en los datos."""
        pass
    
    @abstractmethod
    def validate_config(self, config: ProcessingConfig) -> bool:
        """Validar configuración."""
        pass


class IVariableAgnosticLogger(ABC):
    """
    Interfaz para loggers agnósticos a la variable.
    """
    
    @abstractmethod
    def log_processing_start(self, config: ProcessingConfig) -> None:
        """Registrar inicio de procesamiento."""
        pass
    
    @abstractmethod
    def log_processing_step(self, step_name: str, step_data: Dict[str, Any]) -> None:
        """Registrar paso de procesamiento."""
        pass
    
    @abstractmethod
    def log_processing_complete(self, result: ProcessingResult) -> None:
        """Registrar completación de procesamiento."""
        pass
    
    @abstractmethod
    def log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Registrar error."""
        pass


class IVariableAgnosticMemoryManager(ABC):
    """
    Interfaz para gestores de memoria agnósticos a la variable.
    """
    
    @abstractmethod
    def check_memory_usage(self) -> Dict[str, float]:
        """Verificar uso de memoria."""
        pass
    
    @abstractmethod
    def optimize_memory_config(self, data_size: int, config: ProcessingConfig) -> ProcessingConfig:
        """Optimizar configuración para memoria."""
        pass
    
    @abstractmethod
    def cleanup_memory(self) -> None:
        """Limpiar memoria."""
        pass
    
    @abstractmethod
    def estimate_memory_requirements(self, data: pd.DataFrame, config: ProcessingConfig) -> float:
        """Estimar requerimientos de memoria."""
        pass
