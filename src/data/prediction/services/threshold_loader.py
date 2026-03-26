"""
Threshold Loader Module

This module loads temperature thresholds from Excel file for extreme temperature alert detection.
The thresholds are based on historical percentiles for cold waves (OF) and heat waves (OC).
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TemperatureThresholds:
    """
    Temperature thresholds for a station.
    
    Based on percentiles:
    - Cold Wave (OF): Percentiles 5 (extreme) and 10 (moderate)
    - Heat Wave (OC): Percentiles 95 (extreme) and 90 (moderate)
    """
    station_code: str
    station_name: str
    
    # Cold Wave (OF) - Ola de Frío
    cold_wave_extreme_tmax: float  # Percentil 5 - Extremo
    cold_wave_extreme_tmin: float  # Percentil 5 - Extremo
    cold_wave_tmax: float  # Percentil 10 - Moderado
    cold_wave_tmin: float  # Percentil 10 - Moderado
    
    # Heat Wave (OC) - Ola de Calor
    heat_wave_extreme_tmax: float  # Percentil 95 - Extremo
    heat_wave_extreme_tmin: float  # Percentil 95 - Extremo
    heat_wave_tmax: float  # Percentil 90 - Moderado
    heat_wave_tmin: float  # Percentil 90 - Moderado
    
    def get_cold_wave_threshold(self, variable_type: str, extreme: bool = False) -> Optional[float]:
        """
        Get cold wave threshold for a variable type.
        
        Args:
            variable_type: "temp_max" or "temp_min"
            extreme: If True, return extreme threshold (percentil 5), else moderate (percentil 10)
            
        Returns:
            Threshold value or None
        """
        if variable_type == "temp_max":
            return self.cold_wave_extreme_tmax if extreme else self.cold_wave_tmax
        elif variable_type == "temp_min":
            return self.cold_wave_extreme_tmin if extreme else self.cold_wave_tmin
        return None
    
    def get_heat_wave_threshold(self, variable_type: str, extreme: bool = False) -> Optional[float]:
        """
        Get heat wave threshold for a variable type.
        
        Args:
            variable_type: "temp_max" or "temp_min"
            extreme: If True, return extreme threshold (percentil 95), else moderate (percentil 90)
            
        Returns:
            Threshold value or None
        """
        if variable_type == "temp_max":
            return self.heat_wave_extreme_tmax if extreme else self.heat_wave_tmax
        elif variable_type == "temp_min":
            return self.heat_wave_extreme_tmin if extreme else self.heat_wave_tmin
        return None


class ThresholdLoader:
    """
    Loads temperature thresholds from Excel file.
    
    The Excel file has a specific structure:
    - Row 0: Main headers ("OLA DE FRÍO (OF)" and "OLA DE CALOR (OC)")
    - Row 1: Secondary headers ("Código", "Estación", "Percentil 5", etc.)
    - Row 2: Variable headers ("Tmax", "Tmin")
    - Row 3+: Data rows with station codes and thresholds
    """
    
    def __init__(self, threshold_file: Optional[str] = None):
        """
        Initialize threshold loader.
        
        Args:
            threshold_file: Path to Excel file. If None, searches in current directory.
        """
        if threshold_file is None:
            # Search for the file in multiple locations
            # Strategy: Find project root first, then search from there
            import os
            
            # Use PathManager for consistent path resolution
            from src.config.path_manager import get_path_manager
            path_manager = get_path_manager()
            
            # Define search paths (from most specific to least specific)
            possible_paths = [
                # Primary location: data/thresholds/
                path_manager.thresholds_file,
                # Project root (fallback)
                path_manager.project_root / "Umbrales_Olas de Frío y Calor.xlsx",
                # Current working directory (fallback)
                Path("Umbrales_Olas de Frío y Calor.xlsx"),
            ]
            
            # Try each path
            for path in possible_paths:
                if path.exists():
                    threshold_file = str(path.resolve())
                    logger.info(f"Found threshold file at: {threshold_file}")
                    break
            
            # If still not found, do a recursive search from project root
            if threshold_file is None:
                logger.warning("Threshold file not found in standard locations, searching recursively...")
                for root, dirs, files in os.walk(project_root):
                    # Skip virtual environments and other unnecessary directories
                    if 'venv' in root or '__pycache__' in root or '.git' in root:
                        continue
                    for file in files:
                        if 'Umbrales' in file and file.endswith('.xlsx'):
                            threshold_file = os.path.join(root, file)
                            logger.info(f"Found threshold file via recursive search: {threshold_file}")
                            break
                    if threshold_file:
                        break
        
        if threshold_file is None:
            raise FileNotFoundError(
                "Threshold file 'Umbrales_Olas de Frío y Calor.xlsx' not found. "
                "Please ensure the file exists in the project directory."
            )
        
        self.threshold_file = Path(threshold_file)
        if not self.threshold_file.exists():
            raise FileNotFoundError(f"Threshold file not found: {self.threshold_file}")
        
        self.thresholds: Dict[str, TemperatureThresholds] = {}
        self._load_thresholds()
    
    def _load_thresholds(self) -> None:
        """
        Load thresholds from Excel file.
        
        Excel structure:
        - Columns 0-5: Cold Wave (OF) data
          - Col 0: Código
          - Col 1: Estación
          - Col 2: Tmax Percentil 5 (Extremo)
          - Col 3: Tmin Percentil 5 (Extremo)
          - Col 4: Tmax Percentil 10
          - Col 5: Tmin Percentil 10
        - Column 6: Separator (NaN)
        - Columns 7-12: Heat Wave (OC) data
          - Col 7: Código
          - Col 8: Estación
          - Col 9: Tmax Percentil 95 (Extremo)
          - Col 10: Tmin Percentil 95 (Extremo)
          - Col 11: Tmax Percentil 90
          - Col 12: Tmin Percentil 90
        """
        try:
            logger.info(f"Loading thresholds from: {self.threshold_file}")
            
            # Read Excel file, skip first 3 rows (headers)
            df = pd.read_excel(self.threshold_file, header=None, skiprows=3)
            
            # Reset column names for clarity
            df.columns = range(len(df.columns))
            
            loaded_count = 0
            error_count = 0
            
            for idx, row in df.iterrows():
                try:
                    # Extract Cold Wave (OF) data from columns 0-5
                    code_of = row.iloc[0] if pd.notna(row.iloc[0]) else None
                    station_name_of = row.iloc[1] if pd.notna(row.iloc[1]) else None
                    
                    # Extract Heat Wave (OC) data from columns 7-12
                    code_oc = row.iloc[7] if len(row) > 7 and pd.notna(row.iloc[7]) else None
                    station_name_oc = row.iloc[8] if len(row) > 8 and pd.notna(row.iloc[8]) else None
                    
                    # Validate that codes match
                    if code_of is None or code_oc is None:
                        continue
                    
                    # Convert codes to string for consistency
                    code_of_str = str(int(float(code_of))) if pd.notna(code_of) else None
                    code_oc_str = str(int(float(code_oc))) if pd.notna(code_oc) else None
                    
                    if code_of_str != code_oc_str:
                        logger.warning(f"Code mismatch in row {idx}: OF={code_of_str}, OC={code_oc_str}")
                        continue
                    
                    station_code = code_of_str
                    
                    # Use station name from OF (should be same in both)
                    station_name = str(station_name_of) if pd.notna(station_name_of) else f"Estación_{station_code}"
                    
                    # Extract Cold Wave thresholds
                    of_p5_tmax = float(row.iloc[2]) if pd.notna(row.iloc[2]) else None
                    of_p5_tmin = float(row.iloc[3]) if pd.notna(row.iloc[3]) else None
                    of_p10_tmax = float(row.iloc[4]) if pd.notna(row.iloc[4]) else None
                    of_p10_tmin = float(row.iloc[5]) if pd.notna(row.iloc[5]) else None
                    
                    # Extract Heat Wave thresholds
                    oc_p95_tmax = float(row.iloc[9]) if len(row) > 9 and pd.notna(row.iloc[9]) else None
                    oc_p95_tmin = float(row.iloc[10]) if len(row) > 10 and pd.notna(row.iloc[10]) else None
                    oc_p90_tmax = float(row.iloc[11]) if len(row) > 11 and pd.notna(row.iloc[11]) else None
                    oc_p90_tmin = float(row.iloc[12]) if len(row) > 12 and pd.notna(row.iloc[12]) else None
                    
                    # Validate all thresholds are present
                    if all(x is not None for x in [
                        of_p5_tmax, of_p5_tmin, of_p10_tmax, of_p10_tmin,
                        oc_p95_tmax, oc_p95_tmin, oc_p90_tmax, oc_p90_tmin
                    ]):
                        threshold = TemperatureThresholds(
                            station_code=station_code,
                            station_name=station_name,
                            cold_wave_extreme_tmax=of_p5_tmax,
                            cold_wave_extreme_tmin=of_p5_tmin,
                            cold_wave_tmax=of_p10_tmax,
                            cold_wave_tmin=of_p10_tmin,
                            heat_wave_extreme_tmax=oc_p95_tmax,
                            heat_wave_extreme_tmin=oc_p95_tmin,
                            heat_wave_tmax=oc_p90_tmax,
                            heat_wave_tmin=oc_p90_tmin
                        )
                        self.thresholds[station_code] = threshold
                        loaded_count += 1
                    else:
                        logger.warning(f"Missing thresholds for station {station_code} in row {idx}")
                        error_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    error_count += 1
                    continue
            
            logger.info(f"Loaded {loaded_count} station thresholds (errors: {error_count})")
            
            if loaded_count == 0:
                raise ValueError("No valid thresholds loaded from file")
                
        except Exception as e:
            logger.error(f"Failed to load thresholds: {e}")
            raise
    
    def get_thresholds(self, station_code: str) -> Optional[TemperatureThresholds]:
        """
        Get thresholds for a station.
        
        Args:
            station_code: Station code as string
            
        Returns:
            TemperatureThresholds object or None if not found
        """
        # Try exact match first
        code_str = str(station_code)
        if code_str in self.thresholds:
            return self.thresholds[code_str]
        
        # Try to extract code from station name if it contains code
        import re
        code_match = re.search(r'^(\d+)_', code_str)
        if code_match:
            extracted_code = code_match.group(1)
            if extracted_code in self.thresholds:
                return self.thresholds[extracted_code]
        
        return None
    
    def has_thresholds(self, station_code: str) -> bool:
        """
        Check if thresholds exist for a station.
        
        Args:
            station_code: Station code as string
            
        Returns:
            True if thresholds exist, False otherwise
        """
        return self.get_thresholds(station_code) is not None
    
    def get_all_station_codes(self) -> list:
        """
        Get list of all station codes with thresholds.
        
        Returns:
            List of station codes
        """
        return list(self.thresholds.keys())

