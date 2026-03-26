"""
Statistics Plotter

Este módulo provee herramientas para visualización estadística de datos meteorológicos.
"""

import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
import pandas as pd
from ....core.interfaces.visualization_strategy import VisualizationStrategyInterface

class StatisticsPlotter(VisualizationStrategyInterface):
    """
    Plotter especializado en visualización estadística (histogramas, boxplots, etc).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def plot_histogram(self, data: pd.DataFrame, column: str, bins: int = 30, title: str = '', xlabel: str = '', ylabel: str = 'Frecuencia', show: bool = True, save_path: Optional[str] = None, **kwargs):
        plt.figure(figsize=self.config.get('figsize', (8, 5)))
        plt.hist(data[column].dropna(), bins=bins, color=kwargs.get('color', 'skyblue'), edgecolor='black', alpha=0.7)
        plt.title(title or f"Histograma de {column}")
        plt.xlabel(xlabel or column)
        plt.ylabel(ylabel)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    def plot_boxplot(self, data: pd.DataFrame, column: str, by: Optional[str] = None, title: str = '', xlabel: str = '', ylabel: str = '', show: bool = True, save_path: Optional[str] = None, **kwargs):
        plt.figure(figsize=self.config.get('figsize', (8, 5)))
        if by:
            data.boxplot(column=column, by=by, grid=False)
            plt.title(title or f"Boxplot de {column} por {by}")
            plt.suptitle('')
            plt.xlabel(xlabel or by)
        else:
            data.boxplot(column=column, grid=False)
            plt.title(title or f"Boxplot de {column}")
            plt.xlabel(xlabel or column)
        plt.ylabel(ylabel or column)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close() 