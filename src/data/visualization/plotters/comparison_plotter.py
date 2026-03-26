"""
Comparison Plotter

Este módulo provee herramientas para comparar series temporales o conjuntos de datos.
"""

import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
import pandas as pd
from ....core.interfaces.visualization_strategy import VisualizationStrategyInterface

class ComparisonPlotter(VisualizationStrategyInterface):
    """
    Plotter especializado en comparación de series o conjuntos de datos.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def plot_comparison(self, data_list: List[pd.DataFrame], x: str, y: str, labels: List[str], title: str = '', xlabel: str = '', ylabel: str = '', show: bool = True, grid: bool = True, save_path: Optional[str] = None, **kwargs):
        plt.figure(figsize=self.config.get('figsize', (12, 5)))
        for data, label in zip(data_list, labels):
            plt.plot(data[x], data[y], label=label, linewidth=kwargs.get('linewidth', 1.5))
        plt.title(title or "Comparación de series")
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

    def plot_before_after(self, before: pd.DataFrame, after: pd.DataFrame, x: str, y: str, title: str = '', xlabel: str = '', ylabel: str = '', show: bool = True, grid: bool = True, save_path: Optional[str] = None, **kwargs):
        plt.figure(figsize=self.config.get('figsize', (12, 5)))
        plt.plot(before[x], before[y], label='Antes', color='gray', linestyle='--', linewidth=kwargs.get('linewidth', 1.5))
        plt.plot(after[x], after[y], label='Después', color='blue', linewidth=kwargs.get('linewidth', 1.5))
        plt.title(title or "Comparación Antes/Después")
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