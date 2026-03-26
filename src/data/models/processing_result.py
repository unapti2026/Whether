"""
Processing Result Model

This module defines the ProcessingResult model for representing
the results of data processing operations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd


@dataclass
class ProcessingResult:
    """
    Model for representing the results of data processing operations.
    
    This model encapsulates processing results with metadata, statistics,
    and status information.
    """
    
    # Core information
    status: str  # 'success', 'failed', 'partial'
    variable_type: str
    input_file: str
    output_file: str
    
    # Processing statistics
    input_rows: int = 0
    output_rows: int = 0
    processing_time_seconds: float = 0.0
    
    # Data characteristics
    date_range: Dict[str, str] = field(default_factory=dict)
    stations: List[str] = field(default_factory=list)
    value_statistics: Dict[str, float] = field(default_factory=dict)
    
    # Processing details
    processing_steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Configuration used
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Processing date
    processing_date: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate the processing result after creation."""
        self._validate_status()
    
    def _validate_status(self):
        """Validate the processing status."""
        valid_statuses = ['success', 'failed', 'partial']
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}. Valid statuses: {valid_statuses}")
    
    def is_successful(self) -> bool:
        """
        Check if processing was successful.
        
        Returns:
            True if processing was successful, False otherwise
        """
        return self.status == 'success'
    
    def has_warnings(self) -> bool:
        """
        Check if processing had warnings.
        
        Returns:
            True if there are warnings, False otherwise
        """
        return len(self.warnings) > 0
    
    def has_errors(self) -> bool:
        """
        Check if processing had errors.
        
        Returns:
            True if there are errors, False otherwise
        """
        return len(self.errors) > 0
    
    def get_processing_efficiency(self) -> float:
        """
        Calculate processing efficiency (output rows / input rows).
        
        Returns:
            Processing efficiency as a percentage
        """
        if self.input_rows == 0:
            return 0.0
        return (self.output_rows / self.input_rows) * 100
    
    def get_processing_rate(self) -> float:
        """
        Calculate processing rate (rows per second).
        
        Returns:
            Processing rate in rows per second
        """
        if self.processing_time_seconds == 0:
            return 0.0
        return self.output_rows / self.processing_time_seconds
    
    def add_warning(self, warning: str):
        """
        Add a warning message.
        
        Args:
            warning: Warning message to add
        """
        self.warnings.append(f"{datetime.now().isoformat()}: {warning}")
    
    def add_error(self, error: str):
        """
        Add an error message.
        
        Args:
            error: Error message to add
        """
        self.errors.append(f"{datetime.now().isoformat()}: {error}")
        if self.status == 'success':
            self.status = 'partial'
    
    def add_processing_step(self, step: str):
        """
        Add a processing step.
        
        Args:
            step: Processing step description
        """
        self.processing_steps.append(f"{datetime.now().isoformat()}: {step}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the processing result.
        
        Returns:
            Dictionary with processing summary
        """
        return {
            'status': self.status,
            'variable_type': self.variable_type,
            'processing_date': self.processing_date.isoformat(),
            'input_file': self.input_file,
            'output_file': self.output_file,
            'input_rows': self.input_rows,
            'output_rows': self.output_rows,
            'processing_efficiency': self.get_processing_efficiency(),
            'processing_rate': self.get_processing_rate(),
            'processing_time_seconds': self.processing_time_seconds,
            'stations_count': len(self.stations),
            'date_range': self.date_range,
            'value_statistics': self.value_statistics,
            'processing_steps_count': len(self.processing_steps),
            'warnings_count': len(self.warnings),
            'errors_count': len(self.errors)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary representation.
        
        Returns:
            Dictionary representation of the result
        """
        return {
            'status': self.status,
            'variable_type': self.variable_type,
            'processing_date': self.processing_date.isoformat(),
            'input_file': self.input_file,
            'output_file': self.output_file,
            'input_rows': self.input_rows,
            'output_rows': self.output_rows,
            'processing_time_seconds': self.processing_time_seconds,
            'date_range': self.date_range,
            'stations': self.stations,
            'value_statistics': self.value_statistics,
            'processing_steps': self.processing_steps,
            'warnings': self.warnings,
            'errors': self.errors,
            'config': self.config
        }
    
    @classmethod
    def create_success(cls, 
                      variable_type: str, 
                      input_file: str, 
                      output_file: str,
                      input_rows: int,
                      output_rows: int,
                      processing_time: float,
                      **kwargs) -> 'ProcessingResult':
        """
        Create a successful processing result.
        
        Args:
            variable_type: Type of meteorological variable
            input_file: Input file path
            output_file: Output file path
            input_rows: Number of input rows
            output_rows: Number of output rows
            processing_time: Processing time in seconds
            **kwargs: Additional parameters
            
        Returns:
            ProcessingResult instance with success status
        """
        return cls(
            status='success',
            variable_type=variable_type,
            input_file=input_file,
            output_file=output_file,
            input_rows=input_rows,
            output_rows=output_rows,
            processing_time_seconds=processing_time,
            **kwargs
        )
    
    @classmethod
    def create_failure(cls, 
                      variable_type: str, 
                      input_file: str, 
                      error_message: str,
                      **kwargs) -> 'ProcessingResult':
        """
        Create a failed processing result.
        
        Args:
            variable_type: Type of meteorological variable
            input_file: Input file path
            error_message: Error message
            **kwargs: Additional parameters
            
        Returns:
            ProcessingResult instance with failed status
        """
        result = cls(
            status='failed',
            variable_type=variable_type,
            input_file=input_file,
            output_file='',
            **kwargs
        )
        result.add_error(error_message)
        return result
    
    def __str__(self) -> str:
        """String representation of the result."""
        return (f"ProcessingResult(status='{self.status}', "
                f"variable_type='{self.variable_type}', "
                f"input_rows={self.input_rows}, output_rows={self.output_rows}, "
                f"processing_time={self.processing_time_seconds:.2f}s)") 