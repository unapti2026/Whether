"""
Model Persistence Service

Este módulo provee funcionalidades para guardar y cargar modelos entrenados,
permitiendo reutilizar modelos sin necesidad de reentrenamiento.
"""

import logging
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import json
import pandas as pd
import numpy as np

from src.core.interfaces.prediction_strategy import ModelTrainingResult
from src.core.exceptions.processing_exceptions import ModelPersistenceError


class ModelPersistenceService:
    """
    Servicio para persistencia de modelos entrenados.
    
    Este servicio maneja el guardado y carga de:
    - Modelos SVR entrenados
    - Modelos SARIMAX entrenados
    - Scalers de preprocesamiento
    - Metadatos de entrenamiento
    - Configuraciones de modelos
    """
    
    def __init__(self, variable_type: str):
        """
        Inicializar el servicio de persistencia de modelos.
        
        Args:
            variable_type: Tipo de variable meteorológica
        """
        self.variable_type = variable_type
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized ModelPersistenceService for {variable_type}")
        
        # Formatos soportados
        self.supported_formats = ['pickle', 'joblib', 'json']
        
    def save_models(self, 
                   model_result: ModelTrainingResult,
                   station_name: str,
                   output_dir: Path,
                   format: str = 'joblib',
                   include_metadata: bool = True) -> Dict[str, str]:
        """
        Guardar modelos entrenados y metadatos.
        
        Args:
            model_result: Resultado del entrenamiento de modelos
            station_name: Nombre de la estación
            output_dir: Directorio de salida
            format: Formato de guardado ('pickle', 'joblib', 'json')
            include_metadata: Si incluir metadatos de entrenamiento
            
        Returns:
            Diccionario con las rutas de los archivos guardados
        """
        try:
            self.logger.info(f"Saving trained models for station: {station_name}")
            
            # Crear directorio de salida
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Normalizar nombre de estación para archivos
            safe_station_name = self._normalize_filename(station_name)
            
            saved_files = {}
            
            # Guardar modelos SVR
            if model_result.svr_models:
                svr_models_file = output_dir / f"{safe_station_name}_svr_models.{format}"
                self._save_svr_models(model_result.svr_models, svr_models_file, format)
                saved_files['svr_models'] = str(svr_models_file)
                self.logger.info(f"  ✅ SVR models saved: {len(model_result.svr_models)} models")
            
            # Guardar modelos SARIMAX
            if model_result.sarimax_model:
                sarimax_model_file = output_dir / f"{safe_station_name}_sarimax_model.{format}"
                self._save_sarimax_model(model_result.sarimax_model, sarimax_model_file, format)
                saved_files['sarimax_model'] = str(sarimax_model_file)
                self.logger.info(f"  ✅ SARIMAX model saved")
            
            # Guardar metadatos de entrenamiento
            if include_metadata:
                metadata_file = output_dir / f"{safe_station_name}_training_metadata.json"
                self._save_training_metadata(model_result, metadata_file, station_name)
                saved_files['training_metadata'] = str(metadata_file)
                self.logger.info(f"  ✅ Training metadata saved")
            
            # Guardar configuración de modelos
            config_file = output_dir / f"{safe_station_name}_model_config.json"
            self._save_model_config(model_result, config_file)
            saved_files['model_config'] = str(config_file)
            self.logger.info(f"  ✅ Model configuration saved")
            
            # Guardar resumen de modelos
            summary_file = output_dir / f"{safe_station_name}_models_summary.json"
            self._save_models_summary(model_result, summary_file, station_name)
            saved_files['models_summary'] = str(summary_file)
            self.logger.info(f"  ✅ Models summary saved")
            
            self.logger.info(f"Successfully saved {len(saved_files)} model files for station {station_name}")
            return saved_files
            
        except Exception as e:
            error_msg = f"Failed to save models for station {station_name}: {e}"
            self.logger.error(error_msg)
            raise ModelPersistenceError(error_msg) from e
    
    def load_models(self, 
                   station_name: str,
                   models_dir: Path,
                   format: str = 'joblib') -> Tuple[ModelTrainingResult, Dict[str, Any]]:
        """
        Cargar modelos guardados.
        
        Args:
            station_name: Nombre de la estación
            models_dir: Directorio donde están guardados los modelos
            format: Formato de los archivos ('pickle', 'joblib', 'json')
            
        Returns:
            Tupla con (ModelTrainingResult, metadatos)
        """
        try:
            self.logger.info(f"Loading trained models for station: {station_name}")
            
            # Normalizar nombre de estación
            safe_station_name = self._normalize_filename(station_name)
            
            # Cargar modelos SVR
            svr_models = {}
            svr_models_file = models_dir / f"{safe_station_name}_svr_models.{format}"
            if svr_models_file.exists():
                svr_models = self._load_svr_models(svr_models_file, format)
                self.logger.info(f"  ✅ Loaded {len(svr_models)} SVR models")
            
            # Cargar modelo SARIMAX
            sarimax_model = None
            sarimax_model_file = models_dir / f"{safe_station_name}_sarimax_model.{format}"
            if sarimax_model_file.exists():
                sarimax_model = self._load_sarimax_model(sarimax_model_file, format)
                self.logger.info(f"  ✅ Loaded SARIMAX model")
            
            # Cargar metadatos
            metadata = {}
            metadata_file = models_dir / f"{safe_station_name}_training_metadata.json"
            if metadata_file.exists():
                metadata = self._load_training_metadata(metadata_file)
                self.logger.info(f"  ✅ Loaded training metadata")
            
            # Crear ModelTrainingResult
            model_result = ModelTrainingResult(
                svr_models=svr_models,
                sarimax_model=sarimax_model,
                selected_imf_for_sarimax=metadata.get('selected_imf_for_sarimax', 0),
                training_time=metadata.get('training_time', 0.0),
                success=True
            )
            
            self.logger.info(f"Successfully loaded models for station {station_name}")
            return model_result, metadata
            
        except Exception as e:
            error_msg = f"Failed to load models for station {station_name}: {e}"
            self.logger.error(error_msg)
            raise ModelPersistenceError(error_msg) from e
    
    def list_saved_models(self, models_dir: Path) -> List[Dict[str, Any]]:
        """
        Listar modelos guardados disponibles.
        
        Args:
            models_dir: Directorio de modelos
            
        Returns:
            Lista de información de modelos guardados
        """
        try:
            if not models_dir.exists():
                return []
            
            models_info = []
            
            # Buscar archivos de modelos
            for file_path in models_dir.glob("*_svr_models.*"):
                station_name = file_path.stem.replace('_svr_models', '')
                
                # Buscar archivos relacionados
                metadata_file = models_dir / f"{station_name}_training_metadata.json"
                config_file = models_dir / f"{station_name}_model_config.json"
                summary_file = models_dir / f"{station_name}_models_summary.json"
                
                model_info = {
                    'station_name': station_name,
                    'svr_models_file': str(file_path),
                    'has_metadata': metadata_file.exists(),
                    'has_config': config_file.exists(),
                    'has_summary': summary_file.exists(),
                    'file_size': file_path.stat().st_size if file_path.exists() else 0,
                    'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
                }
                
                # Cargar información adicional si está disponible
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                            model_info.update({
                                'num_svr_models': summary.get('num_svr_models', 0),
                                'has_sarimax_model': summary.get('has_sarimax_model', False),
                                'training_time': summary.get('training_time', 0.0),
                                'training_date': summary.get('training_date')
                            })
                    except Exception as e:
                        self.logger.warning(f"Could not load summary for {station_name}: {e}")
                
                models_info.append(model_info)
            
            return models_info
            
        except Exception as e:
            self.logger.error(f"Failed to list saved models: {e}")
            return []
    
    def delete_models(self, station_name: str, models_dir: Path) -> bool:
        """
        Eliminar modelos guardados de una estación.
        
        Args:
            station_name: Nombre de la estación
            models_dir: Directorio de modelos
            
        Returns:
            True si se eliminaron correctamente
        """
        try:
            self.logger.info(f"Deleting saved models for station: {station_name}")
            
            safe_station_name = self._normalize_filename(station_name)
            
            # Patrones de archivos a eliminar
            file_patterns = [
                f"{safe_station_name}_svr_models.*",
                f"{safe_station_name}_sarimax_model.*",
                f"{safe_station_name}_training_metadata.json",
                f"{safe_station_name}_model_config.json",
                f"{safe_station_name}_models_summary.json"
            ]
            
            deleted_count = 0
            for pattern in file_patterns:
                for file_path in models_dir.glob(pattern):
                    file_path.unlink()
                    deleted_count += 1
                    self.logger.info(f"  ✅ Deleted: {file_path.name}")
            
            self.logger.info(f"Successfully deleted {deleted_count} files for station {station_name}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to delete models for station {station_name}: {e}"
            self.logger.error(error_msg)
            return False
    
    def _save_svr_models(self, svr_models: Dict[int, Any], file_path: Path, format: str) -> None:
        """Guardar modelos SVR."""
        if format == 'joblib':
            joblib.dump(svr_models, file_path)
        elif format == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(svr_models, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _load_svr_models(self, file_path: Path, format: str) -> Dict[int, Any]:
        """Cargar modelos SVR."""
        if format == 'joblib':
            return joblib.load(file_path)
        elif format == 'pickle':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_sarimax_model(self, sarimax_model: Any, file_path: Path, format: str) -> None:
        """Guardar modelo SARIMAX."""
        if format == 'joblib':
            joblib.dump(sarimax_model, file_path)
        elif format == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(sarimax_model, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _load_sarimax_model(self, file_path: Path, format: str) -> Any:
        """Cargar modelo SARIMAX."""
        if format == 'joblib':
            return joblib.load(file_path)
        elif format == 'pickle':
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_training_metadata(self, model_result: ModelTrainingResult, file_path: Path, station_name: str) -> None:
        """Guardar metadatos de entrenamiento."""
        metadata = {
            'station_name': station_name,
            'variable_type': self.variable_type,
            'training_date': datetime.now().isoformat(),
            'training_time': model_result.training_time,
            'num_svr_models': len(model_result.svr_models),
            'has_sarimax_model': model_result.sarimax_model is not None,
            'selected_imf_for_sarimax': model_result.selected_imf_for_sarimax,
            'success': model_result.success,
            'error_message': model_result.error_message
        }
        
        # Add IMF classifications if available (ensure JSON serializable)
        if hasattr(model_result, 'imf_classifications') and model_result.imf_classifications:
            # Convert any numpy types to regular Python types for JSON serialization
            imf_classifications = {}
            for key, value in model_result.imf_classifications.items():
                if isinstance(value, list):
                    imf_classifications[key] = [int(idx) for idx in value]
                else:
                    imf_classifications[key] = value
            metadata['imf_classifications'] = imf_classifications
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _load_training_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Cargar metadatos de entrenamiento."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_model_config(self, model_result: ModelTrainingResult, file_path: Path) -> None:
        """Guardar configuración de modelos."""
        config = {
            'variable_type': self.variable_type,
            'svr_model_indices': list(model_result.svr_models.keys()),
            'has_sarimax_model': model_result.sarimax_model is not None,
            'selected_imf_for_sarimax': model_result.selected_imf_for_sarimax
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _save_models_summary(self, model_result: ModelTrainingResult, file_path: Path, station_name: str) -> None:
        """Guardar resumen de modelos."""
        summary = {
            'station_name': station_name,
            'variable_type': self.variable_type,
            'training_date': datetime.now().isoformat(),
            'training_time': model_result.training_time,
            'num_svr_models': len(model_result.svr_models),
            'has_sarimax_model': model_result.sarimax_model is not None,
            'selected_imf_for_sarimax': model_result.selected_imf_for_sarimax,
            'success': model_result.success,
            'model_types': {
                'svr_models': [f'IMF_{idx+1}' for idx in model_result.svr_models.keys()],
                'sarimax_model': 'Available' if model_result.sarimax_model else 'Not available'
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def _normalize_filename(self, filename: str) -> str:
        """Normalizar nombre de archivo para evitar caracteres problemáticos."""
        # Reemplazar caracteres problemáticos
        normalized = filename.replace(' ', '_').replace(',', '').replace('.', '')
        normalized = normalized.replace('á', 'a').replace('é', 'e').replace('í', 'i')
        normalized = normalized.replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
        normalized = normalized.replace('Á', 'A').replace('É', 'E').replace('Í', 'I')
        normalized = normalized.replace('Ó', 'O').replace('Ú', 'U').replace('Ñ', 'N')
        
        # Limitar longitud
        if len(normalized) > 50:
            normalized = normalized[:50]
        
        return normalized 