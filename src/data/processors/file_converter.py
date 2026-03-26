"""
File Converter Processor

This module defines the FileConverter class for handling file format conversions
(e.g., Excel to CSV) in the weather prediction system.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd
import logging
from ...core.interfaces.data_processor import DataProcessorInterface
from ...core.exceptions import DataProcessingError


class FileConverter(DataProcessorInterface):
    """
    Processor for converting files between formats (e.g., Excel to CSV).
    Implements DataProcessorInterface for consistency.
    """
    def __init__(self, input_path: str, output_path: Optional[str] = None):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else self.input_path.with_suffix('.csv')
        self.logger = logging.getLogger(self.__class__.__name__)
        self.info: Dict[str, Any] = {}

    def process(self, data=None, **kwargs) -> pd.DataFrame:
        """
        Convert the input file to the desired output format (currently Excel to CSV).
        Returns the loaded DataFrame.
        """
        try:
            if not self.input_path.exists():
                raise DataProcessingError(f"Input file not found: {self.input_path}")
            if self.input_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(self.input_path)
                df.to_csv(self.output_path, index=False, encoding='utf-8')
                self.logger.info(f"Converted {self.input_path} to {self.output_path}")
                self.info['conversion'] = 'excel_to_csv'
                self.info['input'] = str(self.input_path)
                self.info['output'] = str(self.output_path)
                return df
            else:
                raise DataProcessingError(f"Unsupported input file type: {self.input_path.suffix}")
        except Exception as e:
            self.logger.error(f"Error converting file: {e}")
            raise DataProcessingError(f"Error converting file: {e}") from e

    def validate_input(self, data=None) -> bool:
        """Validate that the input file exists and is a supported format."""
        if not self.input_path.exists():
            raise DataProcessingError(f"Input file not found: {self.input_path}")
        if self.input_path.suffix.lower() not in ['.xlsx', '.xls']:
            raise DataProcessingError(f"Unsupported input file type: {self.input_path.suffix}")
        return True

    def get_processing_info(self) -> Dict[str, Any]:
        """Return information about the conversion process."""
        return self.info

    def reset(self) -> None:
        """Reset the converter state."""
        self.info = {} 