"""
Logging Service

Este módulo provee un servicio centralizado para la gestión de logs
del sistema de predicción meteorológica.
"""

import logging
import logging.config
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class LoggingService:
    """
    Servicio centralizado para logging.
    
    Este servicio unifica:
    - Configuración de logging
    - Gestión de loggers
    - Formato de logs
    - Historial de logs
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar el servicio de logging.
        
        Args:
            config: Configuración de logging
        """
        self.config = config or {}
        self._loggers: Dict[str, logging.Logger] = {}
        self._log_history: list = []
        
        # Configurar logging básico
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configurar logging básico."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    def get_logger(self, service_name: str) -> logging.Logger:
        """
        Obtener logger para un servicio.
        
        Args:
            service_name: Nombre del servicio
            
        Returns:
            Logger configurado
        """
        if service_name not in self._loggers:
            logger_instance = logging.getLogger(f"src.{service_name}")
            self._loggers[service_name] = logger_instance
            
        return self._loggers[service_name]
    
    def log_service_initialization(self, service_name: str, variable_type: str) -> None:
        """
        Registrar inicialización de servicio.
        
        Args:
            service_name: Nombre del servicio
            variable_type: Tipo de variable
        """
        logger_instance = self.get_logger(service_name)
        message = f"Initialized {service_name} for {variable_type}"
        logger_instance.info(message)
        self._log_to_history('service_initialization', service_name, message)
    
    def log_service_completion(self, service_name: str, processing_time: float) -> None:
        """
        Registrar completación de servicio.
        
        Args:
            service_name: Nombre del servicio
            processing_time: Tiempo de procesamiento
        """
        logger_instance = self.get_logger(service_name)
        message = f"{service_name} completed in {processing_time:.2f}s"
        logger_instance.info(message)
        self._log_to_history('service_completion', service_name, message)
    
    def log_validation_result(self, service_name: str, validation_type: str, success: bool, details: str = "") -> None:
        """
        Registrar resultado de validación.
        
        Args:
            service_name: Nombre del servicio
            validation_type: Tipo de validación
            success: Si la validación fue exitosa
            details: Detalles adicionales
        """
        logger_instance = self.get_logger(service_name)
        status = "[OK] PASSED" if success else "[ERROR] FAILED"
        message = f"{validation_type} validation {status}"
        if details:
            message += f": {details}"
        
        if success:
            logger_instance.info(message)
        else:
            logger_instance.warning(message)
        
        self._log_to_history('validation', service_name, message)
    
    def log_model_training(self, service_name: str, model_type: str, imf_index: int, metrics: Dict[str, float]) -> None:
        """
        Registrar entrenamiento de modelo.
        
        Args:
            service_name: Nombre del servicio
            model_type: Tipo de modelo (SVR, SARIMAX)
            imf_index: Índice del IMF
            metrics: Métricas del modelo
        """
        logger_instance = self.get_logger(service_name)
        
        # Formatear métricas
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        message = f"{model_type} IMF {imf_index + 1} - {metrics_str}"
        
        logger_instance.info(message)
        self._log_to_history('model_training', service_name, message)
    
    def log_prediction_generation(self, service_name: str, prediction_steps: int, processing_time: float) -> None:
        """
        Registrar generación de predicciones.
        
        Args:
            service_name: Nombre del servicio
            prediction_steps: Número de pasos de predicción
            processing_time: Tiempo de procesamiento
        """
        logger_instance = self.get_logger(service_name)
        message = f"Generated {prediction_steps} predictions in {processing_time:.2f}s"
        
        logger_instance.info(message)
        self._log_to_history('prediction_generation', service_name, message)
    
    def log_error(self, service_name: str, error_type: str, error_message: str, details: str = "") -> None:
        """
        Registrar error.
        
        Args:
            service_name: Nombre del servicio
            error_type: Tipo de error
            error_message: Mensaje de error
            details: Detalles adicionales
        """
        logger_instance = self.get_logger(service_name)
        message = f"[ERROR] {error_type}: {error_message}"
        if details:
            message += f" | Details: {details}"
        
        logger_instance.error(message)
        self._log_to_history('error', service_name, message)
    
    def log_warning(self, service_name: str, warning_type: str, warning_message: str) -> None:
        """
        Registrar advertencia.
        
        Args:
            service_name: Nombre del servicio
            warning_type: Tipo de advertencia
            warning_message: Mensaje de advertencia
        """
        logger_instance = self.get_logger(service_name)
        message = f"[WARNING] {warning_type}: {warning_message}"
        
        logger_instance.warning(message)
        self._log_to_history('warning', service_name, message)
    
    def log_memory_usage(self, service_name: str, memory_info: Dict[str, Any]) -> None:
        """
        Registrar uso de memoria.
        
        Args:
            service_name: Nombre del servicio
            memory_info: Información de memoria
        """
        logger_instance = self.get_logger(service_name)
        
        if 'memory_usage_mb' in memory_info:
            message = f"Memory usage: {memory_info['memory_usage_mb']:.2f} MB"
            if 'memory_cleanup' in memory_info and memory_info['memory_cleanup']:
                message += " (cleanup performed)"
            
            logger_instance.info(message)
            self._log_to_history('memory_usage', service_name, message)
    
    def _log_to_history(self, log_type: str, service_name: str, message: str) -> None:
        """
        Registrar en historial interno.
        
        Args:
            log_type: Tipo de log
            service_name: Nombre del servicio
            message: Mensaje
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': log_type,
            'service': service_name,
            'message': message
        }
        
        self._log_history.append(log_entry)
        
        # Mantener solo los últimos 1000 logs
        if len(self._log_history) > 1000:
            self._log_history = self._log_history[-1000:]
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de logs.
        
        Returns:
            Diccionario con resumen de logs
        """
        if not self._log_history:
            return {'total_logs': 0, 'log_types': {}}
        
        # Contar por tipo
        log_types = {}
        for log_entry in self._log_history:
            log_type = log_entry['type']
            if log_type not in log_types:
                log_types[log_type] = 0
            log_types[log_type] += 1
        
        # Contar por servicio
        service_logs = {}
        for log_entry in self._log_history:
            service = log_entry['service']
            if service not in service_logs:
                service_logs[service] = 0
            service_logs[service] += 1
        
        return {
            'total_logs': len(self._log_history),
            'log_types': log_types,
            'service_logs': service_logs,
            'recent_logs': self._log_history[-10:] if self._log_history else []
        }
    
    def save_logs_to_file(self, file_path: str) -> None:
        """
        Guardar logs en archivo.
        
        Args:
            file_path: Ruta del archivo
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self._log_history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Logs saved to: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save logs: {e}")
    
    def clear_log_history(self) -> None:
        """Limpiar historial de logs."""
        self._log_history.clear()
        logger.info("Log history cleared")
    
    def set_log_level(self, service_name: str, level: str) -> None:
        """
        Establecer nivel de log para un servicio.
        
        Args:
            service_name: Nombre del servicio
            level: Nivel de log (DEBUG, INFO, WARNING, ERROR)
        """
        logger_instance = self.get_logger(service_name)
        
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        
        if level.upper() in level_map:
            logger_instance.setLevel(level_map[level.upper()])
            logger.info(f"Log level set to {level} for {service_name}")
        else:
            logger.warning(f"Invalid log level: {level}")
