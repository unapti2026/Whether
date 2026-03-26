"""
Time Series Plotter

Este módulo provee herramientas para graficar series temporales meteorológicas.
"""

import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
import pandas as pd
from ....core.interfaces.visualization_strategy import VisualizationStrategyInterface

class TimeSeriesPlotter(VisualizationStrategyInterface):
    """
    Plotter especializado en series temporales.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def plot(self, data: pd.DataFrame, x: str, y: str, title: str = '', xlabel: str = '', ylabel: str = '', show: bool = True, grid: bool = True, save_path: Optional[str] = None, **kwargs):
        plt.figure(figsize=self.config.get('figsize', (12, 5)))
        plt.plot(data[x], data[y], label=kwargs.get('label', y), color=kwargs.get('color', 'b'), linewidth=kwargs.get('linewidth', 1.5))
        plt.title(title or f"Serie temporal de {y}")
        plt.xlabel(xlabel or x)
        plt.ylabel(ylabel or y)
        plt.grid(grid)
        plt.legend()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_multiple(self, data: pd.DataFrame, x: str, y_columns: list, title: str = '', xlabel: str = '', ylabel: str = '', show: bool = True, grid: bool = True, save_path: Optional[str] = None, **kwargs):
        plt.figure(figsize=self.config.get('figsize', (12, 5)))
        for y in y_columns:
            plt.plot(data[x], data[y], label=y, linewidth=kwargs.get('linewidth', 1.5))
        plt.title(title or "Series temporales")
        plt.xlabel(xlabel or x)
        plt.ylabel(ylabel or ', '.join(y_columns))
        plt.grid(grid)
        plt.legend()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close() 