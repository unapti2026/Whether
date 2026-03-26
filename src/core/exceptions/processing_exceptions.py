"""
Processing Exceptions

This module contains custom exceptions related to data processing operations
such as imputation, validation, and prediction.
"""

from typing import Optional, Dict, Any


class ImputationError(Exception):
    """
    Exception raised when imputation operations fail.
    
    This exception is used to indicate errors that occur during missing value
    imputation, such as invalid imputation methods or insufficient data.
    """
    
    def __init__(self, message: str, column: Optional[str] = None, method: Optional[str] = None, 
                 data_info: Optional[Dict[str, Any]] = None):
        """
        Initialize the ImputationError.
        
        Args:
            message: Error message describing the failure
            column: Name of the column where imputation failed
            method: Imputation method that failed
            data_info: Additional information about the data
        """
        self.message = message
        self.column = column
        self.method = method
        self.data_info = data_info or {}
        
        # Build the full error message
        full_message = f"Imputation error: {message}"
        if column:
            full_message += f" (Column: {column})"
        if method:
            full_message += f" (Method: {method})"
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
            'column': self.column,
            'method': self.method,
            'data_info': self.data_info
        }


class DecompositionError(Exception):
    """
    Exception raised when EEMD decomposition operations fail.
    
    This exception is used to indicate errors that occur during EEMD decomposition,
    such as invalid parameters, insufficient data, or decomposition failures.
    """
    
    def __init__(self, message: str, parameters: Optional[Dict[str, Any]] = None, 
                 data_info: Optional[Dict[str, Any]] = None):
        """
        Initialize the DecompositionError.
        
        Args:
            message: Error message describing the failure
            parameters: EEMD parameters that were used
            data_info: Additional information about the data
        """
        self.message = message
        self.parameters = parameters or {}
        self.data_info = data_info or {}
        
        # Build the full error message
        full_message = f"Decomposition error: {message}"
        if parameters:
            full_message += f" (Parameters: {parameters})"
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
            'parameters': self.parameters,
            'data_info': self.data_info
        }


class ModelTrainingError(Exception):
    """
    Exception raised when model training operations fail.
    
    This exception is used to indicate errors that occur during model training,
    such as invalid model parameters, insufficient training data, or training failures.
    """
    
    def __init__(self, message: str, model_type: Optional[str] = None, 
                 parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the ModelTrainingError.
        
        Args:
            message: Error message describing the failure
            model_type: Type of model that failed to train
            parameters: Model parameters that were used
        """
        self.message = message
        self.model_type = model_type
        self.parameters = parameters or {}
        
        # Build the full error message
        full_message = f"Model training error: {message}"
        if model_type:
            full_message += f" (Model: {model_type})"
        if parameters:
            full_message += f" (Parameters: {parameters})"
            
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
            'model_type': self.model_type,
            'parameters': self.parameters
        }


class PredictionError(Exception):
    """
    Exception raised when prediction operations fail.
    
    This exception is used to indicate errors that occur during prediction generation,
    such as invalid prediction parameters, model failures, or prediction generation failures.
    """
    
    def __init__(self, message: str, prediction_steps: Optional[int] = None, 
                 model_info: Optional[Dict[str, Any]] = None):
        """
        Initialize the PredictionError.
        
        Args:
            message: Error message describing the failure
            prediction_steps: Number of prediction steps that were attempted
            model_info: Information about the models used
        """
        self.message = message
        self.prediction_steps = prediction_steps
        self.model_info = model_info or {}
        
        # Build the full error message
        full_message = f"Prediction error: {message}"
        if prediction_steps is not None:
            full_message += f" (Steps: {prediction_steps})"
        if model_info:
            full_message += f" (Model info: {model_info})"
            
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
            'prediction_steps': self.prediction_steps,
            'model_info': self.model_info
        }


class ProcessingError(Exception):
    """
    Exception raised when general processing operations fail.
    
    This exception is used to indicate errors that occur during general data processing,
    such as data loading, file operations, or workflow orchestration failures.
    """
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 data_info: Optional[Dict[str, Any]] = None):
        """
        Initialize the ProcessingError.
        
        Args:
            message: Error message describing the failure
            operation: Type of operation that failed
            data_info: Additional information about the data or operation
        """
        self.message = message
        self.operation = operation
        self.data_info = data_info or {}
        
        # Build the full error message
        full_message = f"Processing error: {message}"
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


class ValidationError(Exception):
    """
    Exception raised when validation operations fail.
    
    This exception is used to indicate errors that occur during data validation,
    such as invalid data types, missing required fields, or constraint violations.
    """
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None,
                 constraint: Optional[str] = None):
        """
        Initialize the ValidationError.
        
        Args:
            message: Error message describing the validation failure
            field: Name of the field that failed validation
            value: Value that failed validation
            constraint: Constraint that was violated
        """
        self.message = message
        self.field = field
        self.value = value
        self.constraint = constraint
        
        # Build the full error message
        full_message = f"Validation error: {message}"
        if field:
            full_message += f" (Field: {field})"
        if value is not None:
            full_message += f" (Value: {value})"
        if constraint:
            full_message += f" (Constraint: {constraint})"
            
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
            'field': self.field,
            'value': self.value,
            'constraint': self.constraint
        }


class ModelPersistenceError(Exception):
    """
    Exception raised when model persistence operations fail.
    
    This exception is used to indicate errors that occur during saving or loading
    of trained models, such as file system errors, serialization issues, or
    incompatible model formats.
    """
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 model_info: Optional[Dict[str, Any]] = None):
        """
        Initialize the ModelPersistenceError.
        
        Args:
            message: Error message describing the persistence failure
            operation: Type of operation that failed ('save', 'load', 'delete')
            model_info: Additional information about the model
        """
        self.message = message
        self.operation = operation
        self.model_info = model_info or {}
        
        # Build the full error message
        full_message = f"Model persistence error: {message}"
        if operation:
            full_message += f" (Operation: {operation})"
        if model_info:
            full_message += f" (Model info: {model_info})"
            
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
            'model_info': self.model_info
        } 