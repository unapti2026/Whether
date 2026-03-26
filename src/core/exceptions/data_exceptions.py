"""
Data Exceptions

This module contains custom exceptions related to data processing operations.
"""

from typing import Optional, Dict, Any


class DataProcessingError(Exception):
    """
    Exception raised when data processing operations fail.
    
    This exception is used to indicate errors that occur during data processing,
    such as invalid data formats, missing required columns, or processing failures.
    """
    
    def __init__(self, message: str, operation: Optional[str] = None, data_info: Optional[Dict[str, Any]] = None):
        """
        Initialize the DataProcessingError.
        
        Args:
            message: Error message describing the failure
            operation: Name of the operation that failed
            data_info: Additional information about the data that caused the error
        """
        self.message = message
        self.operation = operation
        self.data_info = data_info or {}
        
        # Build the full error message
        full_message = f"Data processing error: {message}"
        if operation:
            full_message += f" (Operation: {operation})"
        if data_info:
            full_message += f" (Data info: {data_info})"
            
        super().__init__(full_message)
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        return self.message
    
    def get_details(self) -> Dict[str, Any]:
        """
        Get detailed information about the error.
        
        Returns:
            Dictionary containing error details
        """
        return {
            'message': self.message,
            'operation': self.operation,
            'data_info': self.data_info
        } 