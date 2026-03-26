"""
Prediction Configuration Factory

This module provides factory methods for creating prediction configurations
with different presets and validation.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.core.interfaces.prediction_strategy import PredictionConfig


@dataclass
class PredictionPreset:
    """Preset configuration for prediction."""
    name: str
    description: str
    config: Dict[str, Any]


class PredictionConfigFactory:
    """
    Factory for creating prediction configurations.
    
    This factory provides predefined configurations and validation
    for different prediction scenarios.
    """
    
    # Predefined presets
    PRESETS = {
        'development': PredictionPreset(
            name="Development",
            description="Configuration for development and debugging",
            config={
                'max_stations': None,  # Process all stations
                'prediction_steps': 30,
                'use_fixed_horizon': True,
                'prediction_horizon_weeks': 3,  # 3 weeks = 21 days
                'prediction_horizon_days': None,
                'eemd_nensembles': 20,
                'eemd_sd_thresh_values': [0.05, 0.1, 0.15, 0.2],
                # Fase 2: Temporal Weighting
                'use_temporal_weighting': True,
                'temporal_weighting_method': 'exponential',
                'temporal_decay_factor': 0.1,
                'temporal_recent_window_days': 30,
                'temporal_weighting_strength': 0.5
            }
        ),
        'production': PredictionPreset(
            name="Production",
            description="Production-ready configuration for all stations",
            config={
                'max_stations': None,  # Process all stations
                'prediction_steps': 90,
                'use_fixed_horizon': True,
                'prediction_horizon_weeks': 4,  # 4 weeks = 28 days
                'prediction_horizon_days': None,
                'eemd_nensembles': 50,
                'eemd_sd_thresh_values': [0.05, 0.1, 0.15, 0.2, 0.25],
                # Fase 2: Temporal Weighting
                'use_temporal_weighting': True,
                'temporal_weighting_method': 'exponential',
                'temporal_decay_factor': 0.15,  # Slightly stronger for production
                'temporal_recent_window_days': 30,
                'temporal_weighting_strength': 0.6
            }
        ),
        'high_quality': PredictionPreset(
            name="High Quality",
            description="High-quality configuration with extensive parameter search",
            config={
                'max_stations': None,  # Process all stations
                'prediction_steps': 60,
                'use_fixed_horizon': True,
                'prediction_horizon_weeks': 3,  # 3 weeks = 21 days
                'prediction_horizon_days': None,
                'eemd_nensembles': 100,
                'eemd_sd_thresh_values': [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                # Fase 2: Temporal Weighting
                'use_temporal_weighting': True,
                'temporal_weighting_method': 'exponential',
                'temporal_decay_factor': 0.12,
                'temporal_recent_window_days': 30,
                'temporal_weighting_strength': 0.7  # Stronger weighting for high quality
            }
        )
    }
    
    @classmethod
    def create_from_preset(cls, preset_name: str, variable_type: str, **overrides) -> PredictionConfig:
        """
        Create configuration from a predefined preset.
        
        Args:
            preset_name: Name of the preset to use
            variable_type: Type of meteorological variable
            **overrides: Additional parameters to override
            
        Returns:
            PredictionConfig instance
            
        Raises:
            ValueError: If preset name is invalid
        """
        if preset_name not in cls.PRESETS:
            available_presets = list(cls.PRESETS.keys())
            raise ValueError(f"Invalid preset '{preset_name}'. Available: {available_presets}")
        
        preset = cls.PRESETS[preset_name]
        config_dict = preset.config.copy()
        config_dict.update(overrides)
        config_dict['variable_type'] = variable_type
        
        return cls._create_config(config_dict)
    
    @classmethod
    def create_custom(cls, variable_type: str, **kwargs) -> PredictionConfig:
        """
        Create custom configuration with provided parameters.
        
        Args:
            variable_type: Type of meteorological variable
            **kwargs: Configuration parameters
            
        Returns:
            PredictionConfig instance
        """
        config_dict = {'variable_type': variable_type}
        config_dict.update(kwargs)
        
        return cls._create_config(config_dict)
    
    @classmethod
    def _create_config(cls, config_dict: Dict[str, Any]) -> PredictionConfig:
        """
        Create PredictionConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            PredictionConfig instance
        """
        # Set default values if not provided
        defaults = {
            'prediction_steps': 30,
            'num_lags': 7,
            'train_test_split': 0.8,
            'max_stations': None,  # Process all stations by default
            
            # Prediction Horizon Configuration (NEW)
            'use_fixed_horizon': True,
            'prediction_horizon_weeks': 3,  # Default: 3 weeks
            'prediction_horizon_days': None,
            'legacy_horizon_ratio': 0.2,
            'max_horizon_days': 60,
            
            # EEMD Configuration
            'eemd_sd_thresh_values': [0.05, 0.1, 0.15, 0.2],
            'eemd_nensembles': 20,
            'eemd_noise_factor': 0.1,
            'eemd_max_imfs': 10,
            'eemd_orthogonality_threshold': 0.1,
            'eemd_correlation_threshold': 0.1,
            
            # Model Configuration
            'svr_kernel': 'rbf',
            'svr_c': 1.0,
            'svr_gamma': 'scale',
            'sarimax_order': (1, 1, 0),
            'sarimax_seasonal_order': (1, 0, 0, 365),
            
            # Fase 2: Temporal Weighting Configuration
            'use_temporal_weighting': True,
            'temporal_weighting_method': 'exponential',
            'temporal_decay_factor': 0.1,
            'temporal_increment_factor': 1.0,
            'temporal_recent_window_days': 30,
            'temporal_weighting_strength': 0.5
        }
        
        # Merge defaults with provided config
        final_config = defaults.copy()
        final_config.update(config_dict)
        
        return PredictionConfig(**final_config)
    
    @classmethod
    def get_available_presets(cls) -> Dict[str, str]:
        """
        Get available preset names and descriptions.
        
        Returns:
            Dictionary mapping preset names to descriptions
        """
        return {name: preset.description for name, preset in cls.PRESETS.items()}
    
    @classmethod
    def validate_config(cls, config: PredictionConfig) -> bool:
        """
        Validate prediction configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate basic parameters
        if config.prediction_steps <= 0:
            raise ValueError("prediction_steps must be positive")
        
        if config.num_lags <= 0:
            raise ValueError("num_lags must be positive")
        
        if not 0 < config.train_test_split < 1:
            raise ValueError("train_test_split must be between 0 and 1")
        
        # Validate EEMD parameters
        if not config.eemd_sd_thresh_values:
            raise ValueError("eemd_sd_thresh_values cannot be empty")
        
        if config.eemd_nensembles <= 0:
            raise ValueError("eemd_nensembles must be positive")
        
        if config.eemd_noise_factor <= 0:
            raise ValueError("eemd_noise_factor must be positive")
        
        if config.eemd_max_imfs <= 0:
            raise ValueError("eemd_max_imfs must be positive")
        
        return True 