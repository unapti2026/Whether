"""
Meteorological Plotter Module

This module provides the specialized plotter for meteorological data visualization,
including time series plots, forecast comparisons, and data analysis plots.
"""

from typing import Dict, Any, Optional, Union, List
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging

from .base_plotter import BasePlotter
from src.config.settings import get_plot_config


class MeteorologicalPlotter(BasePlotter):
    """
    Specialized plotter for meteorological data visualization.
    
    This plotter handles the visualization of meteorological time series data,
    including historical data, forecasts, and comparative analysis.
    
    Attributes:
        date_column (str): Name of the date column
        value_column (str): Name of the value column
        station_column (str): Name of the station column
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 plot_type: str = 'default'):
        """
        Initialize the meteorological plotter.
        
        Args:
            config: Custom configuration dictionary (uses default if None)
            plot_type: Type of plot ('default', 'forecast', 'analysis')
        """
        # Get default configuration
        default_config = get_plot_config(plot_type)
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        super().__init__(default_config)
        
        # Set default column names
        self.date_column = 'Fecha'
        self.value_column = 'Temperatura'
        self.station_column = 'Estación'
        
        self.logger.info(f"MeteorologicalPlotter initialized for {plot_type} plots")
    
    def plot_data(self, 
                  data: pd.DataFrame,
                  date_column: Optional[str] = None,
                  value_column: Optional[str] = None,
                  title: str = 'Meteorological Data',
                  x_label: str = 'Date',
                  y_label: str = 'Temperature (°C)',
                  show_grid: Optional[bool] = None,
                  **kwargs) -> None:
        """
        Plot meteorological data with continuous segments.
        
        Args:
            data: DataFrame containing the meteorological data
            date_column: Name of the date column (uses default if None)
            value_column: Name of the value column (uses default if None)
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            show_grid: Whether to show grid (uses config if None)
            **kwargs: Additional plotting parameters
        """
        try:
            date_col = date_column or self.date_column
            value_col = value_column or self.value_column
            show_grid = show_grid if show_grid is not None else self.style_config['grid_enabled']
            
            # Split data into continuous segments
            segments = self.split_into_continuous_segments(data, date_col)
            
            # Create the plot
            plt.figure(figsize=self.style_config['figure_size'])
            
            for segment in segments:
                plt.plot(segment[date_col], 
                        segment[value_col], 
                        linewidth=self.style_config['line_width'], 
                        color=self.style_config['original_data_color'])
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.grid(show_grid)
            plt.xticks(rotation=45)
            
            # Set y-axis limits with some padding
            y_min = data[value_col].min()
            y_max = data[value_col].max()
            y_padding = (y_max - y_min) * 0.05
            plt.ylim(y_min - y_padding, y_max + y_padding)
            
            plt.tight_layout()
            plt.show()
            
            self.logger.info(f"Data plotted successfully: {len(data)} points, {len(segments)} segments")
            
        except Exception as e:
            self.logger.error(f"Error plotting data: {e}")
            raise
    
    def save_plot(self, 
                  data: pd.DataFrame,
                  output_path: Union[str, Path],
                  date_column: Optional[str] = None,
                  value_column: Optional[str] = None,
                  title: str = 'Meteorological Data',
                  x_label: str = 'Date',
                  y_label: str = 'Temperature (°C)',
                  show_grid: Optional[bool] = None,
                  **kwargs) -> None:
        """
        Save meteorological data plot to file.
        
        Args:
            data: DataFrame containing the meteorological data
            output_path: Path where to save the plot
            date_column: Name of the date column (uses default if None)
            value_column: Name of the value column (uses default if None)
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            show_grid: Whether to show grid (uses config if None)
            **kwargs: Additional plotting parameters
        """
        try:
            date_col = date_column or self.date_column
            value_col = value_column or self.value_column
            show_grid = show_grid if show_grid is not None else self.style_config['grid_enabled']
            
            # Prepare output directory
            output_path = self.prepare_output_directory(output_path)
            
            # Split data into continuous segments
            segments = self.split_into_continuous_segments(data, date_col)
            
            # Create the plot
            plt.figure(figsize=self.style_config['figure_size'])
            
            for segment in segments:
                plt.plot(segment[date_col], 
                        segment[value_col], 
                        linewidth=self.style_config['line_width'], 
                        color=self.style_config['original_data_color'])
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.grid(show_grid)
            plt.xticks(rotation=45)
            
            # Set y-axis limits with some padding
            y_min = data[value_col].min()
            y_max = data[value_col].max()
            y_padding = (y_max - y_min) * 0.05
            plt.ylim(y_min - y_padding, y_max + y_padding)
            
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            self.logger.info(f"Plot saved successfully: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving plot: {e}")
            raise
    
    def plot_forecast(self, 
                     historical_data: pd.DataFrame,
                     forecast_data: pd.DataFrame,
                     date_column: Optional[str] = None,
                     value_column: Optional[str] = None,
                     title: str = 'Meteorological Forecast',
                     x_label: str = 'Date',
                     y_label: str = 'Temperature (°C)',
                     historical_label: str = 'Historical Data',
                     forecast_label: str = 'Forecast',
                     show_grid: Optional[bool] = None,
                     **kwargs) -> None:
        """
        Plot historical data with forecast comparison.
        
        Args:
            historical_data: DataFrame containing historical data
            forecast_data: DataFrame containing forecast data
            date_column: Name of the date column (uses default if None)
            value_column: Name of the value column (uses default if None)
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            historical_label: Label for historical data
            forecast_label: Label for forecast data
            show_grid: Whether to show grid (uses config if None)
            **kwargs: Additional plotting parameters
        """
        try:
            date_col = date_column or self.date_column
            value_col = value_column or self.value_column
            show_grid = show_grid if show_grid is not None else self.style_config['grid_enabled']
            
            # Create the plot
            plt.figure(figsize=self.style_config['figure_size'])
            
            # Plot historical data
            plt.plot(historical_data[date_col], 
                    historical_data[value_col], 
                    label=historical_label,
                    linewidth=self.style_config['line_width'], 
                    color=self.style_config['original_data_color'])
            
            # Plot forecast data
            plt.plot(forecast_data[date_col], 
                    forecast_data[value_col], 
                    label=forecast_label,
                    linewidth=self.style_config['line_width'], 
                    color=self.style_config['forecasted_data_color'])
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.legend()
            plt.grid(show_grid)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            self.logger.info(f"Forecast plot created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating forecast plot: {e}")
            raise
    
    def save_forecast_plot(self, 
                          historical_data: pd.DataFrame,
                          forecast_data: pd.DataFrame,
                          output_path: Union[str, Path],
                          date_column: Optional[str] = None,
                          value_column: Optional[str] = None,
                          title: str = 'Meteorological Forecast',
                          x_label: str = 'Date',
                          y_label: str = 'Temperature (°C)',
                          historical_label: str = 'Historical Data',
                          forecast_label: str = 'Forecast',
                          show_grid: Optional[bool] = None,
                          **kwargs) -> None:
        """
        Save forecast plot to file.
        
        Args:
            historical_data: DataFrame containing historical data
            forecast_data: DataFrame containing forecast data
            output_path: Path where to save the plot
            date_column: Name of the date column (uses default if None)
            value_column: Name of the value column (uses default if None)
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            historical_label: Label for historical data
            forecast_label: Label for forecast data
            show_grid: Whether to show grid (uses config if None)
            **kwargs: Additional plotting parameters
        """
        try:
            date_col = date_column or self.date_column
            value_col = value_column or self.value_column
            show_grid = show_grid if show_grid is not None else self.style_config['grid_enabled']
            
            # Prepare output directory
            output_path = self.prepare_output_directory(output_path)
            
            # Create the plot
            plt.figure(figsize=self.style_config['figure_size'])
            
            # Plot historical data
            plt.plot(historical_data[date_col], 
                    historical_data[value_col], 
                    label=historical_label,
                    linewidth=self.style_config['line_width'], 
                    color=self.style_config['original_data_color'])
            
            # Plot forecast data
            plt.plot(forecast_data[date_col], 
                    forecast_data[value_col], 
                    label=forecast_label,
                    linewidth=self.style_config['line_width'], 
                    color=self.style_config['forecasted_data_color'])
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.legend()
            plt.grid(show_grid)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            self.logger.info(f"Forecast plot saved successfully: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving forecast plot: {e}")
            raise
    
    def plot_station_comparison(self, 
                               data: pd.DataFrame,
                               stations: List[str],
                               date_column: Optional[str] = None,
                               value_column: Optional[str] = None,
                               title: str = 'Station Comparison',
                               x_label: str = 'Date',
                               y_label: str = 'Temperature (°C)',
                               show_grid: Optional[bool] = None,
                               **kwargs) -> None:
        """
        Plot comparison of multiple meteorological stations.
        
        Args:
            data: DataFrame containing data from multiple stations
            stations: List of station names to compare
            date_column: Name of the date column (uses default if None)
            value_column: Name of the value column (uses default if None)
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            show_grid: Whether to show grid (uses config if None)
            **kwargs: Additional plotting parameters
        """
        try:
            date_col = date_column or self.date_column
            value_col = value_column or self.value_column
            station_col = self.station_column
            show_grid = show_grid if show_grid is not None else self.style_config['grid_enabled']
            
            # Create the plot
            plt.figure(figsize=self.style_config['figure_size'])
            
            # Plot each station
            for station in stations:
                station_data = data[data[station_col] == station]
                if not station_data.empty:
                    plt.plot(station_data[date_col], 
                            station_data[value_col], 
                            label=station,
                            linewidth=self.style_config['line_width'])
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.legend()
            plt.grid(show_grid)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            self.logger.info(f"Station comparison plot created for {len(stations)} stations")
            
        except Exception as e:
            self.logger.error(f"Error creating station comparison plot: {e}")
            raise 