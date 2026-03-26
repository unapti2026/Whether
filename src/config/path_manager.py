"""
Path Manager

Centralized path management with environment variable support.
Eliminates relative paths (../) and uses project-relative or absolute paths.
"""

import os
from pathlib import Path
from typing import Optional, Dict
from functools import lru_cache


class PathManager:
    """
    Manages all project paths with environment variable support.
    
    Paths can be configured via environment variables:
    - WEATHER_DATA_DIR: Data input directory
    - WEATHER_OUTPUT_DIR: Output directory
    - WEATHER_THRESHOLDS_FILE: Thresholds file path
    - WEATHER_CONFIG_DIR: Configuration directory
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize path manager.
        
        Args:
            project_root: Project root directory. If None, auto-detects.
        """
        if project_root is None:
            # Auto-detect: go up from src/config/path_manager.py to project root
            self._project_root = Path(__file__).parent.parent.parent.resolve()
        else:
            self._project_root = Path(project_root).resolve()
        
        self._validate_project_root()
    
    def _validate_project_root(self) -> None:
        """Validate that project root exists and contains expected structure."""
        if not self._project_root.exists():
            raise ValueError(f"Project root does not exist: {self._project_root}")
        
        expected_dirs = ['src', 'config']
        for dir_name in expected_dirs:
            expected_path = self._project_root / dir_name
            if not expected_path.exists():
                # Create if missing (for config/)
                if dir_name == 'config':
                    expected_path.mkdir(exist_ok=True)
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return self._project_root
    
    @property
    def data_dir(self) -> Path:
        """Get data directory (configurable via WEATHER_DATA_DIR)."""
        env_path = os.getenv('WEATHER_DATA_DIR')
        if env_path:
            return Path(env_path).resolve()
        return self._project_root / "data"
    
    @property
    def input_dir(self) -> Path:
        """Get input data directory."""
        return self.data_dir / "input"
    
    @property
    def thresholds_file(self) -> Path:
        """Get thresholds file path (configurable via WEATHER_THRESHOLDS_FILE)."""
        env_path = os.getenv('WEATHER_THRESHOLDS_FILE')
        if env_path:
            return Path(env_path).resolve()
        return self.data_dir / "thresholds" / "Umbrales_Olas de Frío y Calor.xlsx"
    
    @property
    def output_dir(self) -> Path:
        """Get output directory (configurable via WEATHER_OUTPUT_DIR)."""
        env_path = os.getenv('WEATHER_OUTPUT_DIR')
        if env_path:
            return Path(env_path).resolve()
        return self._project_root / "output"
    
    @property
    def config_dir(self) -> Path:
        """Get configuration directory (configurable via WEATHER_CONFIG_DIR)."""
        env_path = os.getenv('WEATHER_CONFIG_DIR')
        if env_path:
            return Path(env_path).resolve()
        return self._project_root / "config"
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        return self._project_root / "logs"
    
    def get_variable_data_file(self, variable_name: str, extension: str = "xlsx") -> Path:
        """
        Get data file path for a variable.
        
        Args:
            variable_name: Variable name (e.g., 'temp_max')
            extension: File extension ('xlsx' or 'csv')
            
        Returns:
            Path to variable data file
        """
        return self.input_dir / f"{variable_name}.{extension}"
    
    def get_output_subdir(self, stage: str, variable_name: str) -> Path:
        """
        Get output subdirectory for a stage and variable.
        
        Args:
            stage: Processing stage ('preprocessing', 'imputation', 'prediction')
            variable_name: Variable name
            
        Returns:
            Path to output subdirectory
        """
        return self.output_dir / stage / variable_name
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.data_dir,
            self.input_dir,
            self.data_dir / "thresholds",
            self.output_dir,
            self.config_dir,
            self.logs_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_all_paths(self) -> Dict[str, Path]:
        """
        Get all configured paths.
        
        Returns:
            Dictionary with all path configurations
        """
        return {
            'project_root': self.project_root,
            'data_dir': self.data_dir,
            'input_dir': self.input_dir,
            'thresholds_file': self.thresholds_file,
            'output_dir': self.output_dir,
            'config_dir': self.config_dir,
            'logs_dir': self.logs_dir,
        }


@lru_cache(maxsize=1)
def get_path_manager() -> PathManager:
    """
    Get singleton PathManager instance.
    
    Returns:
        PathManager instance
    """
    return PathManager()

