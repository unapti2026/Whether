"""
Base Plotter Module

This module provides the abstract base class for all data visualizers
in the weather prediction system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from pathlib import Path
import logging


class BasePlotter(ABC):
    """
    Abstract base class for data visualizers.
    
    This class defines the interface that all plotters must implement.
    It provides common functionality for plot styling, data preparation,
    and file management.
    
    Attributes:
        config (Dict[str, Any]): Configuration parameters for plotting
        logger (logging.Logger): Logger instance for debugging and monitoring
        style_config (Dict[str, Any]): Matplotlib style configuration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base plotter.
        
        Args:
            config: Optional configuration dictionary with plotting parameters
        """
        self.config = config or {}
        self.logger = self._setup_logger()
        self.style_config = self._get_style_config()
        self._apply_style()
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logger for the plotter instance.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(f"{self.__class__.__name__}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _get_style_config(self) -> Dict[str, Any]:
        """
        Get the default style configuration.
        
        Returns:
            Dictionary with matplotlib style parameters
        """
        return {
            'figure_size': (12, 5),
            'font_family': 'serif',
            'font_size': 10,
            'title_size': 14,
            'label_size': 12,
            'line_width': 0.5,
            'original_data_color': 'k',
            'forecasted_data_color': 'r',
            'grid_enabled': False
        }
    
    def _apply_style(self) -> None:
        """
        Apply the configured style to matplotlib.
        """
        try:
            rcParams['figure.figsize'] = self.style_config['figure_size']
            rcParams['font.family'] = self.style_config['font_family']
            rcParams['font.size'] = self.style_config['font_size']
            rcParams['axes.titlesize'] = self.style_config['title_size']
            rcParams['axes.labelsize'] = self.style_config['label_size']
            rcParams['axes.linewidth'] = 1
            rcParams['xtick.labelsize'] = self.style_config['font_size']
            rcParams['ytick.labelsize'] = self.style_config['font_size']
            rcParams['legend.fontsize'] = self.style_config['font_size']
            rcParams['legend.title_fontsize'] = self.style_config['title_size']
            rcParams['figure.titlesize'] = self.style_config['title_size']
            rcParams['pdf.fonttype'] = 3
            
            self.logger.info("Matplotlib style applied successfully")
        except Exception as e:
            self.logger.error(f"Error applying matplotlib style: {e}")
    
    def split_into_continuous_segments(self, data: pd.DataFrame, date_column: str) -> List[pd.DataFrame]:
        """
        Split the data into continuous segments based on the date column.
        
        This method identifies gaps in the time series and splits the data
        into continuous segments for better visualization.
        
        Args:
            data: DataFrame containing the time series data
            date_column: Name of the date column
            
        Returns:
            List of DataFrames, each representing a continuous segment
        """
        try:
            # Ensure data is sorted by date
            data = data.sort_values(date_column).copy()
            
            # Calculate the difference between consecutive dates
            data['date_diff'] = data[date_column].diff().dt.days
            
            # Create groups for discontinuous segments (gap > 1 day)
            data['group'] = (data['date_diff'] > 1).cumsum()
            
            # Split into segments
            segments = [group for _, group in data.groupby('group')]
            
            self.logger.info(f"Split data into {len(segments)} continuous segments")
            return segments
            
        except Exception as e:
            self.logger.error(f"Error splitting data into segments: {e}")
            return [data]
    
    def prepare_output_directory(self, output_path: Union[str, Path]) -> Path:
        """
        Prepare the output directory for saving plots.
        
        Args:
            output_path: Path where to save the plot
            
        Returns:
            Path object for the output directory
            
        Raises:
            IOError: If the directory cannot be created
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directory prepared: {output_path.parent}")
        return output_path
    
    @abstractmethod
    def plot_data(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Plot the data. Must be implemented by subclasses.
        
        Args:
            data: DataFrame containing the data to plot
            **kwargs: Additional plotting parameters
        """
        pass
    
    @abstractmethod
    def save_plot(self, data: pd.DataFrame, output_path: Union[str, Path], **kwargs) -> None:
        """
        Save the plot to file. Must be implemented by subclasses.
        
        Args:
            data: DataFrame containing the data to plot
            output_path: Path where to save the plot
            **kwargs: Additional plotting parameters
        """
        pass
    
    def get_plotting_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the plotting configuration.
        
        Returns:
            Dictionary with plotting configuration and statistics
        """
        summary = {
            'plotter_class': self.__class__.__name__,
            'style_config': self.style_config.copy(),
            'config_parameters': len(self.config)
        }
        
        return summary 