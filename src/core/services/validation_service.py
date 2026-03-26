"""
Validation Service

Este módulo provee un servicio centralizado para todas las validaciones
del sistema de predicción meteorológica.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path

from ..validators.data_validator import DataValidator
from ..exceptions.processing_exceptions import ValidationError

logger = logging.getLogger(__name__)


class ValidationService:
    """
    Servicio centralizado para validaciones.
    
    Este servicio unifica:
    - Validación de series temporales
    - Validación de DataFrames
    - Validación de configuraciones
    - Validación de archivos
    """
    
    def __init__(self):
        """Inicializar el servicio de validación."""
        self.logger = logger
        self.data_validator = DataValidator()
        self._validation_history: List[Dict[str, Any]] = []
        
    def validate_time_series(self, 
                           time_series: pd.Series, 
                           min_length: int = 10,
                           require_variance: bool = True) -> bool:
        """
        Validar serie temporal.
        
        Args:
            time_series: Serie temporal a validar
            min_length: Longitud mínima requerida
            require_variance: Si requiere varianza no nula
            
        Returns:
            True si la validación pasa
            
        Raises:
            ValidationError: Si la validación falla
        """
        try:
            validation_result = self.data_validator.validate_time_series(time_series)
            
            # Validaciones adicionales
            if len(time_series) < min_length:
                raise ValidationError(f"Time series too short: {len(time_series)} < {min_length}")
            
            if require_variance and np.var(time_series.dropna()) == 0:
                raise ValidationError("Time series has zero variance")
            
            self._log_validation('time_series', True, len(time_series))
            return True
            
        except Exception as e:
            self._log_validation('time_series', False, len(time_series), str(e))
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Time series validation failed: {e}")
    
    def validate_dataframe(self, 
                          data: pd.DataFrame, 
                          required_columns: List[str],
                          min_rows: int = 1) -> bool:
        """
        Validar DataFrame.
        
        Args:
            data: DataFrame a validar
            required_columns: Columnas requeridas
            min_rows: Número mínimo de filas
            
        Returns:
            True si la validación pasa
            
        Raises:
            ValidationError: Si la validación falla
        """
        try:
            # Validar estructura
            self.data_validator.validate_dataframe_structure(data, required_columns)
            
            # Validar número de filas
            if len(data) < min_rows:
                raise ValidationError(f"DataFrame too small: {len(data)} < {min_rows}")
            
            self._log_validation('dataframe', True, len(data))
            return True
            
        except Exception as e:
            self._log_validation('dataframe', False, len(data), str(e))
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"DataFrame validation failed: {e}")
    
    def validate_configuration(self, 
                             config: Any, 
                             required_attributes: List[str]) -> bool:
        """
        Validar configuración.
        
        Args:
            config: Configuración a validar
            required_attributes: Atributos requeridos
            
        Returns:
            True si la validación pasa
            
        Raises:
            ValidationError: Si la validación falla
        """
        try:
            if config is None:
                raise ValidationError("Configuration is None")
            
            missing_attributes = []
            for attr in required_attributes:
                if not hasattr(config, attr):
                    missing_attributes.append(attr)
            
            if missing_attributes:
                raise ValidationError(f"Missing required attributes: {missing_attributes}")
            
            self._log_validation('configuration', True, len(required_attributes))
            return True
            
        except Exception as e:
            self._log_validation('configuration', False, len(required_attributes), str(e))
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Configuration validation failed: {e}")
    
    def validate_file_path(self, 
                          file_path: Union[str, Path], 
                          must_exist: bool = True,
                          must_be_file: bool = True) -> bool:
        """
        Validar ruta de archivo.
        
        Args:
            file_path: Ruta del archivo
            must_exist: Si el archivo debe existir
            must_be_file: Si debe ser un archivo (no directorio)
            
        Returns:
            True si la validación pasa
            
        Raises:
            ValidationError: Si la validación falla
        """
        try:
            path = Path(file_path)
            
            if must_exist and not path.exists():
                raise ValidationError(f"File does not exist: {file_path}")
            
            if must_be_file and path.exists() and not path.is_file():
                raise ValidationError(f"Path is not a file: {file_path}")
            
            self._log_validation('file_path', True, 1)
            return True
            
        except Exception as e:
            self._log_validation('file_path', False, 1, str(e))
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"File path validation failed: {e}")
    
    def validate_directory_path(self, 
                              dir_path: Union[str, Path], 
                              must_exist: bool = True,
                              create_if_missing: bool = False) -> bool:
        """
        Validar ruta de directorio.
        
        Args:
            dir_path: Ruta del directorio
            must_exist: Si el directorio debe existir
            create_if_missing: Si crear el directorio si no existe
            
        Returns:
            True si la validación pasa
            
        Raises:
            ValidationError: Si la validación falla
        """
        try:
            path = Path(dir_path)
            
            if not path.exists():
                if create_if_missing:
                    path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created directory: {dir_path}")
                elif must_exist:
                    raise ValidationError(f"Directory does not exist: {dir_path}")
            
            if path.exists() and not path.is_dir():
                raise ValidationError(f"Path is not a directory: {dir_path}")
            
            self._log_validation('directory_path', True, 1)
            return True
            
        except Exception as e:
            self._log_validation('directory_path', False, 1, str(e))
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Directory path validation failed: {e}")
    
    def _log_validation(self, 
                       validation_type: str, 
                       success: bool, 
                       data_size: int,
                       error_message: Optional[str] = None) -> None:
        """
        Registrar validación en el historial.
        
        Args:
            validation_type: Tipo de validación
            success: Si la validación fue exitosa
            data_size: Tamaño de los datos validados
            error_message: Mensaje de error (si aplica)
        """
        validation_record = {
            'type': validation_type,
            'success': success,
            'data_size': data_size,
            'timestamp': pd.Timestamp.now(),
            'error_message': error_message
        }
        
        self._validation_history.append(validation_record)
        
        if success:
            self.logger.debug(f"Validation passed: {validation_type} (size: {data_size})")
        else:
            self.logger.warning(f"Validation failed: {validation_type} - {error_message}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de validaciones.
        
        Returns:
            Diccionario con resumen de validaciones
        """
        if not self._validation_history:
            return {'total_validations': 0, 'success_rate': 0.0}
        
        total_validations = len(self._validation_history)
        successful_validations = sum(1 for v in self._validation_history if v['success'])
        success_rate = successful_validations / total_validations
        
        # Agrupar por tipo
        validation_types = {}
        for validation in self._validation_history:
            v_type = validation['type']
            if v_type not in validation_types:
                validation_types[v_type] = {'total': 0, 'successful': 0}
            
            validation_types[v_type]['total'] += 1
            if validation['success']:
                validation_types[v_type]['successful'] += 1
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'success_rate': success_rate,
            'validation_types': validation_types,
            'recent_validations': self._validation_history[-10:] if self._validation_history else []
        }
    
    def clear_validation_history(self) -> None:
        """Limpiar historial de validaciones."""
        self._validation_history.clear()
        self.logger.info("Validation history cleared")
