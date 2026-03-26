"""
File Repository

This module defines the FileRepository class for handling
general file operations in the weather prediction system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, BinaryIO
from pathlib import Path
import json
import pickle
import yaml
import logging
import shutil
import os

from ...core.exceptions import DataProcessingError, ValidationError


class FileRepository(ABC):
    """
    Abstract repository for file operations.
    
    This repository provides an abstraction layer for file operations
    including reading, writing, and managing files.
    """
    
    def __init__(self):
        """Initialize the file repository."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def read_file(self, file_path: str, **kwargs) -> Any:
        """
        Read content from a file.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional reading parameters
            
        Returns:
            File content
            
        Raises:
            DataProcessingError: If reading fails
        """
        pass
    
    @abstractmethod
    def write_file(self, file_path: str, content: Any, **kwargs) -> bool:
        """
        Write content to a file.
        
        Args:
            file_path: Path to the file
            content: Content to write
            **kwargs: Additional writing parameters
            
        Returns:
            True if writing was successful, False otherwise
            
        Raises:
            DataProcessingError: If writing fails
        """
        pass
    
    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file exists, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if deletion was successful, False otherwise
        """
        pass


class LocalFileRepository(FileRepository):
    """
    Local file system implementation of FileRepository.
    
    This repository handles file operations on the local file system.
    """
    
    def __init__(self, base_path: Union[str, Path] = "."):
        """
        Initialize the local file repository.
        
        Args:
            base_path: Base path for file operations
        """
        super().__init__()
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized LocalFileRepository with base path: {self.base_path}")
    
    def read_file(self, file_path: str, **kwargs) -> Any:
        """
        Read content from a file.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional reading parameters
            
        Returns:
            File content
            
        Raises:
            DataProcessingError: If reading fails
        """
        try:
            resolved_path = self._resolve_path(file_path)
            
            if not resolved_path.exists():
                raise DataProcessingError(
                    f"File not found: {resolved_path}",
                    operation="read_file",
                    data_info={"file_path": file_path, "resolved_path": str(resolved_path)}
                )
            
            # Determine file type and read accordingly
            file_extension = resolved_path.suffix.lower()
            
            if file_extension == '.json':
                return self._read_json(resolved_path, **kwargs)
            elif file_extension in ['.yaml', '.yml']:
                return self._read_yaml(resolved_path, **kwargs)
            elif file_extension == '.pickle':
                return self._read_pickle(resolved_path, **kwargs)
            elif file_extension == '.txt':
                return self._read_text(resolved_path, **kwargs)
            else:
                # Default to binary read
                return self._read_binary(resolved_path, **kwargs)
                
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(
                f"Error reading file {file_path}: {str(e)}",
                operation="read_file",
                data_info={"file_path": file_path, "error": str(e)}
            ) from e
    
    def write_file(self, file_path: str, content: Any, **kwargs) -> bool:
        """
        Write content to a file.
        
        Args:
            file_path: Path to the file
            content: Content to write
            **kwargs: Additional writing parameters
            
        Returns:
            True if writing was successful, False otherwise
            
        Raises:
            DataProcessingError: If writing fails
        """
        try:
            resolved_path = self._resolve_path(file_path)
            
            # Ensure directory exists
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine file type and write accordingly
            file_extension = resolved_path.suffix.lower()
            
            if file_extension == '.json':
                return self._write_json(resolved_path, content, **kwargs)
            elif file_extension in ['.yaml', '.yml']:
                return self._write_yaml(resolved_path, content, **kwargs)
            elif file_extension == '.pickle':
                return self._write_pickle(resolved_path, content, **kwargs)
            elif file_extension == '.txt':
                return self._write_text(resolved_path, content, **kwargs)
            else:
                # Default to binary write
                return self._write_binary(resolved_path, content, **kwargs)
                
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(
                f"Error writing file {file_path}: {str(e)}",
                operation="write_file",
                data_info={"file_path": file_path, "error": str(e)}
            ) from e
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            resolved_path = self._resolve_path(file_path)
            return resolved_path.exists() and resolved_path.is_file()
        except Exception:
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            resolved_path = self._resolve_path(file_path)
            
            if not resolved_path.exists():
                self.logger.warning(f"File does not exist: {resolved_path}")
                return False
            
            resolved_path.unlink()
            self.logger.info(f"Successfully deleted file: {resolved_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting file {file_path}: {e}")
            return False
    
    def copy_file(self, source_path: str, destination_path: str) -> bool:
        """
        Copy a file from source to destination.
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            
        Returns:
            True if copy was successful, False otherwise
        """
        try:
            source_resolved = self._resolve_path(source_path)
            dest_resolved = self._resolve_path(destination_path)
            
            if not source_resolved.exists():
                raise DataProcessingError(
                    f"Source file not found: {source_resolved}",
                    operation="copy_file",
                    data_info={"source_path": source_path, "destination_path": destination_path}
                )
            
            # Ensure destination directory exists
            dest_resolved.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source_resolved, dest_resolved)
            self.logger.info(f"Successfully copied file from {source_resolved} to {dest_resolved}")
            return True
            
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(
                f"Error copying file from {source_path} to {destination_path}: {str(e)}",
                operation="copy_file",
                data_info={"source_path": source_path, "destination_path": destination_path, "error": str(e)}
            ) from e
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
            
        Raises:
            DataProcessingError: If information retrieval fails
        """
        try:
            resolved_path = self._resolve_path(file_path)
            
            if not resolved_path.exists():
                raise DataProcessingError(
                    f"File not found: {resolved_path}",
                    operation="get_file_info",
                    data_info={"file_path": file_path}
                )
            
            stat = resolved_path.stat()
            
            info = {
                'file_path': str(resolved_path),
                'file_name': resolved_path.name,
                'file_extension': resolved_path.suffix,
                'file_size': stat.st_size,
                'created_time': stat.st_ctime,
                'modified_time': stat.st_mtime,
                'is_file': resolved_path.is_file(),
                'is_directory': resolved_path.is_dir(),
                'is_readable': resolved_path.is_file() and os.access(resolved_path, os.R_OK),
                'is_writable': resolved_path.is_file() and os.access(resolved_path, os.W_OK)
            }
            
            return info
            
        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(
                f"Error getting file info for {file_path}: {str(e)}",
                operation="get_file_info",
                data_info={"file_path": file_path, "error": str(e)}
            ) from e
    
    def list_files(self, directory: str = "", pattern: str = "*") -> List[str]:
        """
        List files in a directory matching a pattern.
        
        Args:
            directory: Directory to list (relative to base path)
            pattern: File pattern to match (default: all files)
            
        Returns:
            List of file paths
        """
        try:
            dir_path = self._resolve_path(directory)
            
            if not dir_path.exists() or not dir_path.is_dir():
                return []
            
            files = list(dir_path.glob(pattern))
            return [str(f.relative_to(self.base_path)) for f in files if f.is_file()]
            
        except Exception as e:
            self.logger.error(f"Error listing files in {directory}: {e}")
            return []
    
    def _resolve_path(self, file_path: str) -> Path:
        """
        Resolve a file path relative to the base path.
        
        Args:
            file_path: File path (relative or absolute)
            
        Returns:
            Resolved path
        """
        path = Path(file_path)
        
        # If it's already an absolute path, return as is
        if path.is_absolute():
            return path
        
        # If it's a relative path, resolve against base path
        return self.base_path / path
    
    def _read_json(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Read JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f, **kwargs)
    
    def _write_json(self, file_path: Path, content: Dict[str, Any], **kwargs) -> bool:
        """Write JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False, **kwargs)
        return True
    
    def _read_yaml(self, file_path: Path, **kwargs) -> Dict[str, Any]:
        """Read YAML file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f, **kwargs)
    
    def _write_yaml(self, file_path: Path, content: Dict[str, Any], **kwargs) -> bool:
        """Write YAML file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(content, f, default_flow_style=False, allow_unicode=True, **kwargs)
        return True
    
    def _read_pickle(self, file_path: Path, **kwargs) -> Any:
        """Read pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f, **kwargs)
    
    def _write_pickle(self, file_path: Path, content: Any, **kwargs) -> bool:
        """Write pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(content, f, **kwargs)
        return True
    
    def _read_text(self, file_path: Path, **kwargs) -> str:
        """Read text file."""
        encoding = kwargs.get('encoding', 'utf-8')
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    
    def _write_text(self, file_path: Path, content: str, **kwargs) -> bool:
        """Write text file."""
        encoding = kwargs.get('encoding', 'utf-8')
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    
    def _read_binary(self, file_path: Path, **kwargs) -> bytes:
        """Read binary file."""
        with open(file_path, 'rb') as f:
            return f.read()
    
    def _write_binary(self, file_path: Path, content: Union[bytes, BinaryIO], **kwargs) -> bool:
        """Write binary file."""
        with open(file_path, 'wb') as f:
            if isinstance(content, bytes):
                f.write(content)
            else:
                shutil.copyfileobj(content, f)
        return True 