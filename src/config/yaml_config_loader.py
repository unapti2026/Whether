"""YAML Configuration Loader for Weather Prediction System."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from src.config.path_manager import get_path_manager


class YAMLConfigLoader:
    """Load and merge configuration with priority: YAML -> ENV -> CLI."""
    
    _env_mapping = {
        'WEATHER_MAX_STATIONS': ('processing', 'max_stations', int),
        'WEATHER_ENABLE_VALIDATION': ('processing', 'enable_validation', lambda x: x.lower() == 'true'),
        'WEATHER_ENABLE_LOGGING': ('processing', 'enable_logging', lambda x: x.lower() == 'true'),
        'WEATHER_ENABLE_PLOTS': ('output', 'enable_plots', lambda x: x.lower() == 'true'),
        'WEATHER_HORIZON_WEEKS': ('prediction', 'default_horizon_weeks', int),
        'WEATHER_HORIZON_DAYS': ('prediction', 'default_horizon_days', int),
        'WEATHER_OUTPUT_DIR': ('output', 'base_dir', str),
    }
    
    def __init__(self, config_file: Optional[str] = None):
        path_manager = get_path_manager()
        if config_file is None:
            config_file = path_manager.config_dir / "default.yaml"
        
        self.config_file = Path(config_file)
        self._config: Optional[Dict[str, Any]] = None
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self._config is not None:
            return self._config
        
        if not self.config_file.exists():
            self._config = {}
            return self._config
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            return self._config
        except Exception as e:
            raise ValueError(f"Failed to load YAML config from {self.config_file}: {e}")
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        for env_var, (section, key, converter) in self._env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    if section not in env_config:
                        env_config[section] = {}
                    env_config[section][key] = converted_value
                except (ValueError, TypeError):
                    pass
        
        return env_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        config = self.load()
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_merged_config(self, cli_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get merged configuration with priority: YAML -> ENV -> CLI.
        
        Args:
            cli_args: Command line arguments (highest priority)
            
        Returns:
            Merged configuration dictionary
        """
        yaml_config = self.load()
        env_config = self._load_from_env()
        
        merged = self._deep_merge(yaml_config, env_config)
        
        if cli_args:
            cli_normalized = self._normalize_cli_args(cli_args)
            merged = self._deep_merge(merged, cli_normalized)
        
        return merged
    
    def _normalize_cli_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize CLI arguments to match YAML structure."""
        normalized = {}
        
        if 'max_stations' in args and args['max_stations'] is not None:
            normalized.setdefault('processing', {})['max_stations'] = args['max_stations']
        
        if 'enable_validation' in args:
            normalized.setdefault('processing', {})['enable_validation'] = args['enable_validation']
        
        if 'enable_logging' in args:
            normalized.setdefault('processing', {})['enable_logging'] = args['enable_logging']
        
        if 'enable_plots' in args:
            normalized.setdefault('output', {})['enable_plots'] = args['enable_plots']
        
        if 'prediction_horizon_weeks' in args and args['prediction_horizon_weeks'] is not None:
            normalized.setdefault('prediction', {})['default_horizon_weeks'] = args['prediction_horizon_weeks']
        
        if 'prediction_horizon_days' in args and args['prediction_horizon_days'] is not None:
            normalized.setdefault('prediction', {})['default_horizon_days'] = args['prediction_horizon_days']
        
        if 'output_dir' in args and args['output_dir'] is not None:
            normalized.setdefault('output', {})['base_dir'] = args['output_dir']
        
        return normalized
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

