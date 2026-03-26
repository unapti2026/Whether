"""
Visualization Strategy Interface

This module defines the abstract interface for visualization strategies.
All visualization strategies must implement this interface to ensure consistency.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class VisualizationStrategyInterface(ABC):
    """
    Abstract interface for visualization strategies.
    
    This interface defines the contract that all visualization strategies must implement.
    It ensures consistency across different types of data visualization methods.
    """
    
    @abstractmethod
    def create_visualization(self, data: pd.DataFrame, **kwargs) -> Figure:
        """
        Create a visualization from the input data.
        
        Args:
            data: DataFrame containing the data to visualize
            **kwargs: Additional visualization parameters
            
        Returns:
            Matplotlib Figure object
            
        Raises:
            VisualizationError: If visualization creation fails
        """
        pass
    
    @abstractmethod
    def save_visualization(self, figure: Figure, filepath: str, **kwargs) -> None:
        """
        Save the visualization to a file.
        
        Args:
            figure: Matplotlib Figure object to save
            filepath: Path where to save the visualization
            **kwargs: Additional saving parameters (dpi, format, etc.)
            
        Raises:
            VisualizationError: If saving fails
        """
        pass
    
    @abstractmethod
    def get_visualization_info(self) -> Dict[str, Any]:
        """
        Get information about the visualization.
        
        Returns:
            Dictionary containing visualization metadata
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data is suitable for visualization.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid for visualization, False otherwise
            
        Raises:
            ValidationError: If validation fails
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported output formats for saving.
        
        Returns:
            List of supported file formats (e.g., ['png', 'jpg', 'pdf'])
        """
        pass
    
    @abstractmethod
    def set_style(self, style_name: str) -> None:
        """
        Set the visualization style.
        
        Args:
            style_name: Name of the style to apply
            
        Raises:
            ValidationError: If style is not supported
        """
        pass 