"""
Configuration Validator

This module contains the ConfigValidator class for validating configuration
settings in the weather prediction system.
"""

from typing import Any, Dict, List, Optional, Union
import os
from pathlib import Path
from ..exceptions import ValidationError


class ConfigValidator:
    """
    Validator for configuration settings.
    
    This class provides methods to validate various aspects of configuration,
    including file paths, parameter ranges, and required settings.
    """
    
    def __init__(self):
        """Initialize the ConfigValidator."""
        self.validation_rules = {}
        self.validation_history = []
    
    def validate_file_path(self, file_path: str, must_exist: bool = True, 
                          file_type: Optional[str] = None) -> bool:
        """
        Validate that a file path is valid and optionally exists.
        
        Args:
            file_path: Path to validate
            must_exist: Whether the file must exist
            file_type: Expected file extension (e.g., '.csv', '.xlsx')
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValidationError: If path is invalid or file doesn't exist when required
        """
        if not isinstance(file_path, str):
            raise ValidationError(
                "File path must be a string",
                field="file_path",
                value=type(file_path),
                constraint="string_type"
            )
        
        if not file_path.strip():
            raise ValidationError(
                "File path cannot be empty",
                field="file_path",
                value=file_path,
                constraint="non_empty"
            )
        
        # Check file extension if specified
        if file_type:
            if not file_path.lower().endswith(file_type.lower()):
                raise ValidationError(
                    f"File path must end with '{file_type}'",
                    field="file_path",
                    value=file_path,
                    constraint=f"extension_{file_type}"
                )
        
        # Check if file exists if required
        if must_exist:
            if not os.path.exists(file_path):
                raise ValidationError(
                    f"File does not exist: {file_path}",
                    field="file_path",
                    value=file_path,
                    constraint="file_exists"
                )
        
        return True
    
    def validate_directory_path(self, dir_path: str, must_exist: bool = True, 
                              create_if_missing: bool = False) -> bool:
        """
        Validate that a directory path is valid and optionally exists.
        
        Args:
            dir_path: Directory path to validate
            must_exist: Whether the directory must exist
            create_if_missing: Whether to create the directory if it doesn't exist
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValidationError: If path is invalid or directory doesn't exist when required
        """
        if not isinstance(dir_path, str):
            raise ValidationError(
                "Directory path must be a string",
                field="dir_path",
                value=type(dir_path),
                constraint="string_type"
            )
        
        if not dir_path.strip():
            raise ValidationError(
                "Directory path cannot be empty",
                field="dir_path",
                value=dir_path,
                constraint="non_empty"
            )
        
        # Check if directory exists
        if os.path.exists(dir_path):
            if not os.path.isdir(dir_path):
                raise ValidationError(
                    f"Path exists but is not a directory: {dir_path}",
                    field="dir_path",
                    value=dir_path,
                    constraint="is_directory"
                )
        else:
            if must_exist:
                if create_if_missing:
                    try:
                        os.makedirs(dir_path, exist_ok=True)
                    except OSError as e:
                        raise ValidationError(
                            f"Cannot create directory: {dir_path}",
                            field="dir_path",
                            value=dir_path,
                            constraint="creatable"
                        ) from e
                else:
                    raise ValidationError(
                        f"Directory does not exist: {dir_path}",
                        field="dir_path",
                        value=dir_path,
                        constraint="directory_exists"
                    )
        
        return True
    
    def validate_parameter_range(self, value: Union[int, float], param_name: str,
                               min_value: Optional[Union[int, float]] = None,
                               max_value: Optional[Union[int, float]] = None) -> bool:
        """
        Validate that a parameter value is within specified range.
        
        Args:
            value: Parameter value to validate
            param_name: Name of the parameter
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValidationError: If value is outside the specified range
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(
                f"Parameter '{param_name}' must be numeric",
                field=param_name,
                value=type(value),
                constraint="numeric_type"
            )
        
        # Check minimum value
        if min_value is not None and value < min_value:
            raise ValidationError(
                f"Parameter '{param_name}' value {value} is below minimum {min_value}",
                field=param_name,
                value=value,
                constraint=f"min_{min_value}"
            )
        
        # Check maximum value
        if max_value is not None and value > max_value:
            raise ValidationError(
                f"Parameter '{param_name}' value {value} is above maximum {max_value}",
                field=param_name,
                value=value,
                constraint=f"max_{max_value}"
            )
        
        return True
    
    def validate_required_keys(self, config: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that a configuration dictionary contains all required keys.
        
        Args:
            config: Configuration dictionary to validate
            required_keys: List of keys that must be present
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValidationError: If required keys are missing
        """
        if not isinstance(config, dict):
            raise ValidationError(
                "Configuration must be a dictionary",
                field="config",
                value=type(config),
                constraint="dict_type"
            )
        
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            raise ValidationError(
                f"Missing required configuration keys: {missing_keys}",
                field="config",
                value=missing_keys,
                constraint="required_keys"
            )
        
        return True
    
    def validate_string_length(self, value: str, param_name: str,
                             min_length: Optional[int] = None,
                             max_length: Optional[int] = None) -> bool:
        """
        Validate that a string parameter has the expected length.
        
        Args:
            value: String value to validate
            param_name: Name of the parameter
            min_length: Minimum allowed length (inclusive)
            max_length: Maximum allowed length (inclusive)
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValidationError: If string length is outside the specified range
        """
        if not isinstance(value, str):
            raise ValidationError(
                f"Parameter '{param_name}' must be a string",
                field=param_name,
                value=type(value),
                constraint="string_type"
            )
        
        length = len(value)
        
        # Check minimum length
        if min_length is not None and length < min_length:
            raise ValidationError(
                f"Parameter '{param_name}' length {length} is below minimum {min_length}",
                field=param_name,
                value=length,
                constraint=f"min_length_{min_length}"
            )
        
        # Check maximum length
        if max_length is not None and length > max_length:
            raise ValidationError(
                f"Parameter '{param_name}' length {length} is above maximum {max_length}",
                field=param_name,
                value=length,
                constraint=f"max_length_{max_length}"
            )
        
        return True
    
    def validate_enum_value(self, value: Any, param_name: str, 
                          allowed_values: List[Any]) -> bool:
        """
        Validate that a parameter value is one of the allowed values.
        
        Args:
            value: Parameter value to validate
            param_name: Name of the parameter
            allowed_values: List of allowed values
            
        Returns:
            True if validation passes, False otherwise
            
        Raises:
            ValidationError: If value is not in the allowed values list
        """
        if value not in allowed_values:
            raise ValidationError(
                f"Parameter '{param_name}' value '{value}' is not allowed. "
                f"Allowed values: {allowed_values}",
                field=param_name,
                value=value,
                constraint=f"enum_{allowed_values}"
            )
        
        return True
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of validation operations performed.
        
        Returns:
            Dictionary containing validation summary
        """
        return {
            'total_validations': len(self.validation_history),
            'validation_rules': self.validation_rules,
            'recent_validations': self.validation_history[-10:] if self.validation_history else []
        } 