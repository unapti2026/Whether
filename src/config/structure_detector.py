"""
Structure Detection Module

This module provides utilities for automatically detecting the structure
of meteorological data files, including column prefixes, day ranges,
and data organization patterns.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import re
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataStructureConfig:
    """Configuration class for detected data structure."""
    day_prefix: str
    max_days: int
    day_columns: List[str]
    has_day_columns: bool
    structure_type: str
    confidence: float
    detected_patterns: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'day_prefix': self.day_prefix,
            'max_days': self.max_days,
            'day_columns': self.day_columns,
            'has_day_columns': self.has_day_columns,
            'structure_type': self.structure_type,
            'confidence': self.confidence,
            'detected_patterns': self.detected_patterns
        }


class StructureDetector:
    """
    Utility class for detecting data structure from meteorological data files.
    
    This class analyzes the structure of input data to automatically determine:
    - Day column prefix (e.g., 'Día', 'Day', 'D')
    - Maximum number of days per month
    - Data organization pattern
    - Column naming conventions
    """
    
    # Common day prefixes in different languages
    DAY_PREFIXES = {
        'es': ['Día', 'Dia', 'D'],
        'en': ['Day', 'D'],
        'fr': ['Jour', 'J'],
        'de': ['Tag', 'T'],
        'pt': ['Dia', 'D'],
        'it': ['Giorno', 'G']
    }
    
    # Common patterns for day columns
    DAY_PATTERNS = [
        r'^Día(\d{1,2})$',  # Día01, Día02, etc.
        r'^Dia(\d{1,2})$',  # Dia01, Dia02, etc.
        r'^Day(\d{1,2})$',  # Day01, Day02, etc.
        r'^D(\d{1,2})$',    # D01, D02, etc.
        r'^Jour(\d{1,2})$', # Jour01, Jour02, etc.
        r'^Tag(\d{1,2})$',  # Tag01, Tag02, etc.
        r'^Dia(\d{1,2})$',  # Dia01, Dia02, etc. (Portuguese)
        r'^G(\d{1,2})$',    # G01, G02, etc. (Italian)
    ]
    
    @classmethod
    def detect_structure(cls, data: pd.DataFrame, sample_size: Optional[int] = None) -> DataStructureConfig:
        """
        Detect the structure of meteorological data.
        
        Args:
            data: DataFrame to analyze
            sample_size: Number of rows to sample for analysis (None for all)
            
        Returns:
            DataStructureConfig with detected structure information
        """
        logger.info("Starting data structure detection...")
        
        # Sample data if specified
        if sample_size and len(data) > sample_size:
            sample_data = data.sample(n=sample_size, random_state=42)
        else:
            sample_data = data
        
        # Detect day columns and prefix
        day_info = cls._detect_day_columns(sample_data.columns)
        
        # Analyze data patterns
        patterns = cls._analyze_data_patterns(sample_data, day_info)
        
        # Determine structure type
        structure_type = cls._determine_structure_type(day_info, patterns)
        
        # Calculate confidence
        confidence = cls._calculate_confidence(day_info, patterns)
        
        config = DataStructureConfig(
            day_prefix=day_info['prefix'],
            max_days=day_info['max_days'],
            day_columns=day_info['columns'],
            has_day_columns=day_info['has_day_columns'],
            structure_type=structure_type,
            confidence=confidence,
            detected_patterns=patterns
        )
        
        logger.info(f"Structure detection completed:")
        logger.info(f"  Day prefix: {config.day_prefix}")
        logger.info(f"  Max days: {config.max_days}")
        logger.info(f"  Structure type: {config.structure_type}")
        logger.info(f"  Confidence: {config.confidence:.2f}")
        
        return config
    
    @classmethod
    def _detect_day_columns(cls, columns: pd.Index) -> Dict[str, Any]:
        """
        Detect day columns and their prefix.
        
        Args:
            columns: DataFrame columns
            
        Returns:
            Dictionary with day column information
        """
        day_columns = []
        detected_prefix = None
        max_day_number = 0
        
        # Try each pattern
        for pattern in cls.DAY_PATTERNS:
            matches = []
            for col in columns:
                match = re.match(pattern, col)
                if match:
                    day_num = int(match.group(1))
                    matches.append((col, day_num))
                    max_day_number = max(max_day_number, day_num)
            
            if matches:
                # Sort by day number to find the prefix
                matches.sort(key=lambda x: x[1])
                first_col = matches[0][0]
                
                # Extract prefix
                if first_col.startswith('Día'):
                    detected_prefix = 'Día'
                elif first_col.startswith('Dia'):
                    detected_prefix = 'Dia'
                elif first_col.startswith('Day'):
                    detected_prefix = 'Day'
                elif first_col.startswith('D') and len(first_col) > 1:
                    detected_prefix = 'D'
                elif first_col.startswith('Jour'):
                    detected_prefix = 'Jour'
                elif first_col.startswith('Tag'):
                    detected_prefix = 'Tag'
                elif first_col.startswith('G'):
                    detected_prefix = 'G'
                
                day_columns = [col for col, _ in matches]
                break
        
        return {
            'prefix': detected_prefix or 'Día',  # Default fallback
            'max_days': max_day_number,
            'columns': day_columns,
            'has_day_columns': len(day_columns) > 0
        }
    
    @classmethod
    def _analyze_data_patterns(cls, data: pd.DataFrame, day_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data patterns to understand structure.
        
        Args:
            data: DataFrame to analyze
            day_info: Day column information
            
        Returns:
            Dictionary with pattern analysis
        """
        patterns = {
            'has_year_column': False,
            'has_month_column': False,
            'has_station_column': False,
            'has_code_column': False,
            'year_column': None,
            'month_column': None,
            'station_column': None,
            'code_column': None,
            'day_column_distribution': {},
            'data_completeness': 0.0
        }
        
        # Check for common column names
        columns_lower = [col.lower() for col in data.columns]
        
        # Year column detection
        year_candidates = ['año', 'year', 'ano', 'annee', 'jahr']
        for candidate in year_candidates:
            if candidate in columns_lower:
                patterns['has_year_column'] = True
                patterns['year_column'] = data.columns[columns_lower.index(candidate)]
                break
        
        # Month column detection
        month_candidates = ['mes', 'month', 'mois', 'monat']
        for candidate in month_candidates:
            if candidate in columns_lower:
                patterns['has_month_column'] = True
                patterns['month_column'] = data.columns[columns_lower.index(candidate)]
                break
        
        # Station column detection
        station_candidates = ['estación', 'estacion', 'station', 'gare', 'bahnhof']
        for candidate in station_candidates:
            if candidate in columns_lower:
                patterns['has_station_column'] = True
                patterns['station_column'] = data.columns[columns_lower.index(candidate)]
                break
        
        # Code column detection
        code_candidates = ['código', 'codigo', 'code', 'code']
        for candidate in code_candidates:
            if candidate in columns_lower:
                patterns['has_code_column'] = True
                patterns['code_column'] = data.columns[columns_lower.index(candidate)]
                break
        
        # Analyze day column distribution
        if day_info['has_day_columns']:
            for col in day_info['columns']:
                non_null_count = data[col].notna().sum()
                total_count = len(data)
                patterns['day_column_distribution'][col] = {
                    'non_null_count': non_null_count,
                    'total_count': total_count,
                    'completeness': non_null_count / total_count if total_count > 0 else 0
                }
        
        # Calculate overall data completeness
        if day_info['has_day_columns']:
            total_values = sum(info['total_count'] for info in patterns['day_column_distribution'].values())
            total_non_null = sum(info['non_null_count'] for info in patterns['day_column_distribution'].values())
            patterns['data_completeness'] = total_non_null / total_values if total_values > 0 else 0
        
        return patterns
    
    @classmethod
    def _determine_structure_type(cls, day_info: Dict[str, Any], patterns: Dict[str, Any]) -> str:
        """
        Determine the type of data structure.
        
        Args:
            day_info: Day column information
            patterns: Pattern analysis
            
        Returns:
            Structure type string
        """
        if not day_info['has_day_columns']:
            return 'unknown'
        
        if patterns['has_year_column'] and patterns['has_month_column']:
            if patterns['has_station_column'] and patterns['has_code_column']:
                return 'monthly_station_data'
            elif patterns['has_station_column']:
                return 'monthly_station_data_no_code'
            else:
                return 'monthly_data'
        elif patterns['has_station_column']:
            return 'station_data'
        else:
            return 'basic_data'
    
    @classmethod
    def _calculate_confidence(cls, day_info: Dict[str, Any], patterns: Dict[str, Any]) -> float:
        """
        Calculate confidence in the detected structure.
        
        Args:
            day_info: Day column information
            patterns: Pattern analysis
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.0
        
        # Base confidence from day column detection
        if day_info['has_day_columns']:
            confidence += 0.3
            if day_info['max_days'] >= 28:  # Reasonable day range
                confidence += 0.1
        
        # Confidence from required columns
        if patterns['has_year_column']:
            confidence += 0.2
        if patterns['has_month_column']:
            confidence += 0.2
        if patterns['has_station_column']:
            confidence += 0.1
        if patterns['has_code_column']:
            confidence += 0.1
        
        # Confidence from data completeness
        confidence += patterns['data_completeness'] * 0.1
        
        return min(confidence, 1.0)
    
    @classmethod
    def validate_structure(cls, data: pd.DataFrame, config: DataStructureConfig) -> bool:
        """
        Validate that the detected structure is consistent with the data.
        
        Args:
            data: DataFrame to validate
            config: Detected structure configuration
            
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # Check if day columns exist
            if config.has_day_columns:
                missing_columns = [col for col in config.day_columns if col not in data.columns]
                if missing_columns:
                    logger.warning(f"Missing day columns: {missing_columns}")
                    return False
            
            # Check if required columns exist
            required_columns = []
            if config.detected_patterns.get('year_column'):
                required_columns.append(config.detected_patterns['year_column'])
            if config.detected_patterns.get('month_column'):
                required_columns.append(config.detected_patterns['month_column'])
            if config.detected_patterns.get('station_column'):
                required_columns.append(config.detected_patterns['station_column'])
            
            missing_required = [col for col in required_columns if col not in data.columns]
            if missing_required:
                logger.warning(f"Missing required columns: {missing_required}")
                return False
            
            # Check data types
            if config.detected_patterns.get('year_column'):
                year_col = config.detected_patterns['year_column']
                if not pd.api.types.is_numeric_dtype(data[year_col]):
                    logger.warning(f"Year column {year_col} is not numeric")
                    return False
            
            if config.detected_patterns.get('month_column'):
                month_col = config.detected_patterns['month_column']
                if not pd.api.types.is_numeric_dtype(data[month_col]):
                    logger.warning(f"Month column {month_col} is not numeric")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Structure validation failed: {e}")
            return False


def detect_data_structure(data: pd.DataFrame, sample_size: Optional[int] = None) -> DataStructureConfig:
    """
    Convenience function to detect data structure.
    
    Args:
        data: DataFrame to analyze
        sample_size: Number of rows to sample for analysis
        
    Returns:
        DataStructureConfig with detected structure
    """
    return StructureDetector.detect_structure(data, sample_size)


def validate_data_structure(data: pd.DataFrame, config: DataStructureConfig) -> bool:
    """
    Convenience function to validate data structure.
    
    Args:
        data: DataFrame to validate
        config: Detected structure configuration
        
    Returns:
        True if structure is valid, False otherwise
    """
    return StructureDetector.validate_structure(data, config)


def get_default_structure_config() -> DataStructureConfig:
    """
    Get default structure configuration as fallback.
    
    Returns:
        Default DataStructureConfig
    """
    return DataStructureConfig(
        day_prefix='Día',
        max_days=31,
        day_columns=[],
        has_day_columns=False,
        structure_type='unknown',
        confidence=0.0,
        detected_patterns={}
    ) 