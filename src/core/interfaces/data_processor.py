"""
Data Processor Interface

This module defines the abstract interface for data processors.
All data processors must implement this interface to ensure consistency.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import pandas as pd


class DataProcessorInterface(ABC):
    """
    Abstract interface for data processors.
    
    This interface defines the contract that all data processors must implement.
    It ensures consistency across different types of data processing operations.
    """
    
    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Process the input data according to the processor's logic.
        
        Args:
            data: Input DataFrame to be processed
            **kwargs: Additional processing parameters
            
        Returns:
            Processed DataFrame
            
        Raises:
            DataProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate the input data before processing.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
            
        Raises:
            ValidationError: If validation fails
        """
        pass
    
    @abstractmethod
    def get_processing_info(self) -> Dict[str, Any]:
        """
        Get information about the processing operation.
        
        Returns:
            Dictionary containing processing metadata
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the processor to its initial state.
        Useful for reusing the same processor instance.
        """
        pass 