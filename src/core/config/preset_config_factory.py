"""
Preset Configuration Factory

Este módulo implementa un factory especializado para la gestión de presets
de configuración, proporcionando configuraciones predefinidas para diferentes
escenarios de uso (desarrollo, producción, alta calidad, etc.).
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml

from ..interfaces.variable_agnostic_interfaces import (
    ProcessingConfig, 
    IVariableAgnosticConfigFactory
)

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
    category: str
    config_params: Dict[str, Any]
    
    def __post_init__(self):
        if not hasattr(self, 'config_params'):
            self.config_params = {}


@dataclass
class PresetMetadata:
    """
    Metadatos de un preset de configuración.
    
    Attributes:
        name: Nombre del preset
        description: Descripción del preset
        category: Categoría del preset (development, production, research, etc.)
        version: Versión del preset
        author: Autor del preset
        created_date: Fecha de creación
        last_modified: Última modificación
        tags: Etiquetas para categorización
        performance_profile: Perfil de rendimiento esperado
        memory_profile: Perfil de uso de memoria
        accuracy_profile: Perfil de precisión esperada
    """
    name: str
    description: str
    category: str
    version: str = "1.0.0"
    author: str = "System"
    created_date: str = ""
    last_modified: str = ""
    tags: List[str] = field(default_factory=list)
    performance_profile: str = "balanced"  # fast, balanced, accurate
    memory_profile: str = "moderate"       # low, moderate, high
    accuracy_profile: str = "standard"     # basic, standard, high


class PresetConfigFactory:
    """
    Factory especializado para la gestión de presets de configuración.
    
    Este factory proporciona configuraciones predefinidas para diferentes
    escenarios de uso, permitiendo la carga desde archivos y la gestión
    dinámica de presets.
    """
    
    def __init__(self, preset_dir: Optional[Path] = None):
        """
        Inicializar el factory de presets de configuración.
        
        Args:
            preset_dir: Directorio opcional para cargar presets desde archivos
        """
        self.logger = logger
        self.preset_dir = preset_dir or Path("config/presets")
        self._custom_presets = {}
        self._preset_metadata = {}
        
        # Inicializar presets predefinidos
        self._initialize_builtin_presets()
        
        # Cargar presets desde archivos si el directorio existe
        if self.preset_dir.exists():
            self._load_presets_from_files()
        
        self.logger.info("PresetConfigFactory initialized")
    
    def _initialize_builtin_presets(self):
        """Inicializar presets predefinidos del sistema."""
        
        # Preset de desarrollo
        dev_preset = UnifiedConfigPreset(
            name="development",
            description="Configuration for development and debugging",
            category="development",
            config_params={
                'prediction_steps': 30,
                'eemd_ensembles': 5,
                'eemd_sd_thresh_values': [0.1, 0.15],
                'svr_kernel': 'rbf',
                'sarimax_order': (1, 1, 0),
                'sarimax_max_iter': 50,
                'enable_downsampling': True,
                'downsampling_threshold': 10000,
                'memory_cleanup': True,
                'force_garbage_collection': True
            }
        )
        
        # Preset de producción
        prod_preset = UnifiedConfigPreset(
            name="production",
            description="Production-ready configuration for all stations",
            category="production",
            config_params={
                'prediction_steps': 90,
                'eemd_ensembles': 20,
                'eemd_sd_thresh_values': [0.05, 0.1, 0.15, 0.2],
                'svr_kernel': 'rbf',
                'sarimax_order': (1, 1, 0),
                'sarimax_max_iter': 100,
                'enable_downsampling': True,
                'downsampling_threshold': 15000,
                'memory_cleanup': True,
                'force_garbage_collection': True
            }
        )
        
        # Preset de alta calidad
        hq_preset = UnifiedConfigPreset(
            name="high_quality",
            description="High-quality configuration with extensive parameter search",
            category="research",
            config_params={
                'prediction_steps': 180,
                'eemd_ensembles': 50,
                'eemd_sd_thresh_values': [0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2],
                'svr_kernel': 'rbf',
                'sarimax_order': (2, 1, 1),
                'sarimax_max_iter': 200,
                'enable_downsampling': False,
                'memory_cleanup': True,
                'force_garbage_collection': True
            }
        )
        
        # Preset de memoria optimizada
        mem_preset = UnifiedConfigPreset(
            name="memory_optimized",
            description="Memory-optimized configuration for large datasets",
            category="optimization",
            config_params={
                'prediction_steps': 60,
                'eemd_ensembles': 10,
                'eemd_sd_thresh_values': [0.1, 0.15],
                'svr_kernel': 'linear',
                'sarimax_order': (1, 1, 0),
                'sarimax_max_iter': 50,
                'sarimax_data_limit_years': 1,
                'enable_downsampling': True,
                'downsampling_threshold': 5000,
                'memory_cleanup': True,
                'force_garbage_collection': True
            }
        )
        
        # Preset de velocidad
        speed_preset = UnifiedConfigPreset(
            name="fast",
            description="Fast processing configuration for quick results",
            category="performance",
            config_params={
                'prediction_steps': 15,
                'eemd_ensembles': 3,
                'eemd_sd_thresh_values': [0.15],
                'svr_kernel': 'linear',
                'sarimax_order': (1, 1, 0),
                'sarimax_max_iter': 20,
                'enable_downsampling': True,
                'downsampling_threshold': 3000,
                'memory_cleanup': False,
                'force_garbage_collection': False
            }
        )
        
        # Preset de investigación
        research_preset = UnifiedConfigPreset(
            name="research",
            description="Research configuration for detailed analysis",
            category="research",
            config_params={
                'prediction_steps': 365,
                'eemd_ensembles': 100,
                'eemd_sd_thresh_values': [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25],
                'svr_kernel': 'rbf',
                'sarimax_order': (3, 1, 2),
                'sarimax_max_iter': 500,
                'enable_downsampling': False,
                'memory_cleanup': True,
                'force_garbage_collection': True
            }
        )
        
        # Agregar presets al factory
        self.add_preset(dev_preset)
        self.add_preset(prod_preset)
        self.add_preset(hq_preset)
        self.add_preset(mem_preset)
        self.add_preset(speed_preset)
        self.add_preset(research_preset)
        
        # Agregar metadatos
        self._add_preset_metadata("development", PresetMetadata(
            name="development",
            description="Configuration for development and debugging",
            category="development",
            performance_profile="fast",
            memory_profile="low",
            accuracy_profile="basic",
            tags=["dev", "debug", "fast"]
        ))
        
        self._add_preset_metadata("production", PresetMetadata(
            name="production",
            description="Production-ready configuration for all stations",
            category="production",
            performance_profile="balanced",
            memory_profile="moderate",
            accuracy_profile="standard",
            tags=["prod", "stable", "balanced"]
        ))
        
        self._add_preset_metadata("high_quality", PresetMetadata(
            name="high_quality",
            description="High-quality configuration with extensive parameter search",
            category="research",
            performance_profile="slow",
            memory_profile="high",
            accuracy_profile="high",
            tags=["quality", "accurate", "research"]
        ))
        
        self._add_preset_metadata("memory_optimized", PresetMetadata(
            name="memory_optimized",
            description="Memory-optimized configuration for large datasets",
            category="optimization",
            performance_profile="balanced",
            memory_profile="low",
            accuracy_profile="standard",
            tags=["memory", "optimized", "large-data"]
        ))
        
        self._add_preset_metadata("fast", PresetMetadata(
            name="fast",
            description="Fast processing configuration for quick results",
            category="performance",
            performance_profile="fast",
            memory_profile="low",
            accuracy_profile="basic",
            tags=["fast", "quick", "prototype"]
        ))
        
        self._add_preset_metadata("research", PresetMetadata(
            name="research",
            description="Research configuration for detailed analysis",
            category="research",
            performance_profile="slow",
            memory_profile="high",
            accuracy_profile="high",
            tags=["research", "detailed", "analysis"]
        ))
    
    def _add_preset_metadata(self, preset_name: str, metadata: PresetMetadata):
        """Agregar metadatos de un preset."""
        self._preset_metadata[preset_name] = metadata
    
    def add_preset(self, preset: UnifiedConfigPreset) -> bool:
        """
        Agregar un nuevo preset al factory.
        
        Args:
            preset: Preset de configuración a agregar
            
        Returns:
            True si se agregó exitosamente
        """
        try:
            self._custom_presets[preset.name] = preset
            self.logger.info(f"Added custom preset: {preset.name}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding preset '{preset.name}': {e}")
            return False
    
    def remove_preset(self, preset_name: str) -> bool:
        """
        Remover un preset del factory.
        
        Args:
            preset_name: Nombre del preset a remover
            
        Returns:
            True si se removió exitosamente
        """
        try:
            if preset_name in self._custom_presets:
                del self._custom_presets[preset_name]
                self.logger.info(f"Removed custom preset: {preset_name}")
                return True
            else:
                self.logger.warning(f"Preset '{preset_name}' not found for removal")
                return False
        except Exception as e:
            self.logger.error(f"Error removing preset '{preset_name}': {e}")
            return False
    
    def get_preset(self, preset_name: str) -> Optional[UnifiedConfigPreset]:
        """
        Obtener un preset por nombre.
        
        Args:
            preset_name: Nombre del preset
            
        Returns:
            Preset de configuración o None si no existe
        """
        # Buscar en presets personalizados primero
        if preset_name in self._custom_presets:
            return self._custom_presets[preset_name]
        
        # Buscar en presets del sistema
        return super().get_preset(preset_name)
    
    def get_all_presets(self) -> Dict[str, UnifiedConfigPreset]:
        """
        Obtener todos los presets disponibles.
        
        Returns:
            Diccionario con todos los presets
        """
        all_presets = {}
        
        # Agregar presets del sistema
        system_presets = super().get_all_presets()
        all_presets.update(system_presets)
        
        # Agregar presets personalizados
        all_presets.update(self._custom_presets)
        
        return all_presets
    
    def get_presets_by_category(self, category: str) -> Dict[str, UnifiedConfigPreset]:
        """
        Obtener presets por categoría.
        
        Args:
            category: Categoría de presets
            
        Returns:
            Diccionario con presets de la categoría
        """
        category_presets = {}
        
        for name, preset in self.get_all_presets().items():
            metadata = self._preset_metadata.get(name)
            if metadata and metadata.category == category:
                category_presets[name] = preset
        
        return category_presets
    
    def get_preset_metadata(self, preset_name: str) -> Optional[PresetMetadata]:
        """
        Obtener metadatos de un preset.
        
        Args:
            preset_name: Nombre del preset
            
        Returns:
            Metadatos del preset o None si no existe
        """
        return self._preset_metadata.get(preset_name)
    
    def get_preset(self, preset_name: str) -> Optional[UnifiedConfigPreset]:
        """
        Obtener un preset por nombre.
        
        Args:
            preset_name: Nombre del preset
            
        Returns:
            Preset de configuración o None si no existe
        """
        # Buscar en presets personalizados primero
        if preset_name in self._custom_presets:
            return self._custom_presets[preset_name]
        
        # Buscar en presets del sistema
        return None
    
    def get_all_presets(self) -> Dict[str, UnifiedConfigPreset]:
        """
        Obtener todos los presets disponibles.
        
        Returns:
            Diccionario con todos los presets
        """
        all_presets = {}
        
        # Agregar presets personalizados
        all_presets.update(self._custom_presets)
        
        return all_presets
    
    def create_config_from_preset(self, preset_name: str) -> ProcessingConfig:
        """
        Crear configuración desde un preset.
        
        Args:
            preset_name: Nombre del preset
            
        Returns:
            Configuración de procesamiento
            
        Raises:
            ValueError: Si el preset no existe
        """
        preset = self.get_preset(preset_name)
        if not preset:
            available_presets = list(self.get_all_presets().keys())
            raise ValueError(f"Preset '{preset_name}' not found. Available presets: {available_presets}")
        
        return super().create_config_from_preset(preset_name)
    
    def save_preset_to_file(self, preset_name: str, file_path: Path) -> bool:
        """
        Guardar un preset en un archivo.
        
        Args:
            preset_name: Nombre del preset
            file_path: Ruta del archivo
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            preset = self.get_preset(preset_name)
            if not preset:
                self.logger.error(f"Preset '{preset_name}' not found")
                return False
            
            # Crear directorio si no existe
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Preparar datos para guardar
            preset_data = {
                'name': preset.name,
                'description': preset.description,
                'category': preset.category,
                'config_params': preset.config_params
            }
            
            # Agregar metadatos si existen
            metadata = self.get_preset_metadata(preset_name)
            if metadata:
                preset_data['metadata'] = {
                    'version': metadata.version,
                    'author': metadata.author,
                    'created_date': metadata.created_date,
                    'last_modified': metadata.last_modified,
                    'tags': metadata.tags,
                    'performance_profile': metadata.performance_profile,
                    'memory_profile': metadata.memory_profile,
                    'accuracy_profile': metadata.accuracy_profile
                }
            
            # Guardar según la extensión del archivo
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(preset_data, f, indent=2, ensure_ascii=False)
            elif file_path.suffix.lower() in ['.yml', '.yaml']:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(preset_data, f, default_flow_style=False, allow_unicode=True)
            else:
                self.logger.error(f"Unsupported file format: {file_path.suffix}")
                return False
            
            self.logger.info(f"Saved preset '{preset_name}' to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving preset '{preset_name}' to {file_path}: {e}")
            return False
    
    def load_preset_from_file(self, file_path: Path) -> bool:
        """
        Cargar un preset desde un archivo.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            True si se cargó exitosamente
        """
        try:
            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return False
            
            # Cargar según la extensión del archivo
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)
            elif file_path.suffix.lower() in ['.yml', '.yaml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    preset_data = yaml.safe_load(f)
            else:
                self.logger.error(f"Unsupported file format: {file_path.suffix}")
                return False
            
            # Crear preset
            preset = UnifiedConfigPreset(
                name=preset_data['name'],
                description=preset_data['description'],
                category=preset_data['category'],
                config_params=preset_data['config_params']
            )
            
            # Agregar preset
            success = self.add_preset(preset)
            
            # Agregar metadatos si existen
            if success and 'metadata' in preset_data:
                metadata_dict = preset_data['metadata']
                metadata = PresetMetadata(
                    name=preset_data['name'],
                    description=preset_data['description'],
                    category=preset_data['category'],
                    version=metadata_dict.get('version', '1.0.0'),
                    author=metadata_dict.get('author', 'System'),
                    created_date=metadata_dict.get('created_date', ''),
                    last_modified=metadata_dict.get('last_modified', ''),
                    tags=metadata_dict.get('tags', []),
                    performance_profile=metadata_dict.get('performance_profile', 'balanced'),
                    memory_profile=metadata_dict.get('memory_profile', 'moderate'),
                    accuracy_profile=metadata_dict.get('accuracy_profile', 'standard')
                )
                self._add_preset_metadata(preset.name, metadata)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error loading preset from {file_path}: {e}")
            return False
    
    def _load_presets_from_files(self):
        """Cargar presets desde archivos en el directorio de presets."""
        try:
            if not self.preset_dir.exists():
                self.logger.info(f"Preset directory not found: {self.preset_dir}")
                return
            
            # Buscar archivos de presets
            preset_files = []
            for ext in ['*.json', '*.yml', '*.yaml']:
                preset_files.extend(self.preset_dir.glob(ext))
            
            if not preset_files:
                self.logger.info(f"No preset files found in {self.preset_dir}")
                return
            
            # Cargar cada archivo
            loaded_count = 0
            for file_path in preset_files:
                if self.load_preset_from_file(file_path):
                    loaded_count += 1
            
            self.logger.info(f"Loaded {loaded_count}/{len(preset_files)} preset files from {self.preset_dir}")
            
        except Exception as e:
            self.logger.error(f"Error loading presets from files: {e}")
    
    def export_presets(self, export_dir: Path) -> bool:
        """
        Exportar todos los presets a archivos.
        
        Args:
            export_dir: Directorio de exportación
            
        Returns:
            True si se exportaron exitosamente
        """
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            
            exported_count = 0
            for preset_name in self.get_all_presets().keys():
                file_path = export_dir / f"{preset_name}.json"
                if self.save_preset_to_file(preset_name, file_path):
                    exported_count += 1
            
            self.logger.info(f"Exported {exported_count} presets to {export_dir}")
            return exported_count > 0
            
        except Exception as e:
            self.logger.error(f"Error exporting presets to {export_dir}: {e}")
            return False
    
    def get_preset_recommendations(self, 
                                 performance_profile: str = "balanced",
                                 memory_profile: str = "moderate",
                                 accuracy_profile: str = "standard") -> List[str]:
        """
        Obtener recomendaciones de presets basadas en perfiles.
        
        Args:
            performance_profile: Perfil de rendimiento deseado
            memory_profile: Perfil de memoria deseado
            accuracy_profile: Perfil de precisión deseado
            
        Returns:
            Lista de nombres de presets recomendados
        """
        recommendations = []
        
        for name, metadata in self._preset_metadata.items():
            if (metadata.performance_profile == performance_profile and
                metadata.memory_profile == memory_profile and
                metadata.accuracy_profile == accuracy_profile):
                recommendations.append(name)
        
        return recommendations
    
    def validate_preset(self, preset_name: str) -> Tuple[bool, List[str]]:
        """
        Validar un preset.
        
        Args:
            preset_name: Nombre del preset
            
        Returns:
            Tupla con (es_válido, lista_de_errores)
        """
        errors = []
        
        try:
            preset = self.get_preset(preset_name)
            if not preset:
                errors.append(f"Preset '{preset_name}' not found")
                return False, errors
            
            # Validar parámetros requeridos
            required_params = ['prediction_steps', 'eemd_ensembles', 'svr_kernel']
            for param in required_params:
                if param not in preset.config_params:
                    errors.append(f"Missing required parameter: {param}")
            
            # Validar valores de parámetros
            if 'prediction_steps' in preset.config_params:
                steps = preset.config_params['prediction_steps']
                if not isinstance(steps, int) or steps <= 0:
                    errors.append("prediction_steps must be a positive integer")
            
            if 'eemd_ensembles' in preset.config_params:
                ensembles = preset.config_params['eemd_ensembles']
                if not isinstance(ensembles, int) or ensembles <= 0:
                    errors.append("eemd_ensembles must be a positive integer")
            
            # Crear configuración para validación completa
            try:
                config = self.create_config_from_preset(preset_name)
                # Aquí se podrían agregar validaciones adicionales de la configuración
            except Exception as e:
                errors.append(f"Configuration creation failed: {e}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors
