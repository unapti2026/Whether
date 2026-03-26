"""
User Messages Configuration

Centralized configuration for all user-facing messages in Spanish.
All technical code, variables, functions, and classes remain in English.
"""

from typing import Dict, Any

MESSAGES: Dict[str, Any] = {
    "pipeline": {
        "initialization": "Inicializando Pipeline Meteorológico",
        "variable": "Variable",
        "max_stations": "Estaciones máximas",
        "all_stations": "Todas",
        "stages": "Etapas",
        "starting": "Iniciando Pipeline Meteorológico",
        "completed": "Pipeline completado exitosamente",
        "failed": "Pipeline falló",
        "total_time": "Tiempo total",
        "results": "Resultados",
    },
    "stages": {
        "preprocess": "PREPROCESAMIENTO",
        "imputation": "IMPUTACIÓN",
        "prediction": "PREDICCIÓN",
        "executing": "Ejecutando etapa",
        "completed": "completada en",
        "failed": "falló",
    },
    "errors": {
        "file_not_found": "Archivo no encontrado",
        "unsupported_variable": "Variable no soportada",
        "unsupported_stage": "Etapa no soportada",
        "invalid_config": "Configuración inválida",
        "processing_failed": "Procesamiento falló",
    },
    "summary": {
        "title": "RESUMEN DEL PIPELINE",
        "variable": "Variable",
        "stages_executed": "Etapas ejecutadas",
        "success_rate": "Tasa de éxito",
        "total_time": "Tiempo total",
        "output_directories": "Directorios de salida",
    },
    "validation": {
        "data_files_validated": "Archivos de datos validados",
        "excel": "Excel",
        "csv": "CSV",
    },
}

def get_message(category: str, key: str, default: str = "") -> str:
    """
    Get a user-facing message.
    
    Args:
        category: Message category (e.g., 'pipeline', 'errors')
        key: Message key within category
        default: Default value if message not found
        
    Returns:
        User-facing message in Spanish
    """
    return MESSAGES.get(category, {}).get(key, default)

