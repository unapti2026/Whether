"""
Alert Detector Module

This module detects extreme temperature alerts in predictions based on thresholds.
It classifies alerts by type (cold wave, heat wave) and severity (extreme, moderate).
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .threshold_loader import ThresholdLoader, TemperatureThresholds

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of temperature alerts."""
    COLD_WAVE_EXTREME = "cold_wave_extreme"  # Percentil 5 - Extremo
    COLD_WAVE = "cold_wave"  # Percentil 10 - Moderado
    HEAT_WAVE_EXTREME = "heat_wave_extreme"  # Percentil 95 - Extremo
    HEAT_WAVE = "heat_wave"  # Percentil 90 - Moderado
    NONE = "none"


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"  # Extreme (Percentil 5/95)
    WARNING = "warning"  # Moderate (Percentil 10/90)
    INFO = "info"  # Informational


@dataclass
class TemperatureAlert:
    """
    Represents a temperature alert.
    """
    station_code: str
    station_name: str
    alert_type: AlertType
    severity: AlertSeverity
    date: datetime
    predicted_value: float
    threshold_value: float
    variable_type: str  # "temp_max" or "temp_min"
    days_ahead: int  # Days in the future of the prediction
    message: str
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary for serialization."""
        return {
            'station_code': self.station_code,
            'station_name': self.station_name,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'date': self.date.isoformat() if isinstance(self.date, datetime) else str(self.date),
            'predicted_value': self.predicted_value,
            'threshold_value': self.threshold_value,
            'variable_type': self.variable_type,
            'days_ahead': self.days_ahead,
            'message': self.message
        }


class AlertDetector:
    """
    Detects temperature alerts in predictions based on thresholds.
    """
    
    def __init__(self, threshold_loader: ThresholdLoader):
        """
        Initialize alert detector.
        
        Args:
            threshold_loader: ThresholdLoader instance
        """
        self.threshold_loader = threshold_loader
        logger.info("AlertDetector initialized")
    
    def detect_alerts(
        self,
        predictions: pd.Series,
        future_dates: pd.DatetimeIndex,
        station_code: str,
        variable_type: str
    ) -> List[TemperatureAlert]:
        """
        Detect alerts in predictions.
        
        Args:
            predictions: Series with predicted values (index should match future_dates)
            future_dates: DatetimeIndex of prediction dates
            station_code: Station code
            variable_type: "temp_max" or "temp_min"
            
        Returns:
            List of TemperatureAlert objects
        """
        alerts = []
        
        # Get thresholds for this station
        thresholds = self.threshold_loader.get_thresholds(station_code)
        if not thresholds:
            logger.debug(f"No thresholds found for station {station_code}, skipping alert detection")
            return alerts
        
        station_name = thresholds.station_name
        
        # Validate inputs
        if len(predictions) != len(future_dates):
            logger.warning(f"Length mismatch: predictions={len(predictions)}, dates={len(future_dates)}")
            min_len = min(len(predictions), len(future_dates))
            predictions = predictions[:min_len]
            future_dates = future_dates[:min_len]
        
        # Detect alerts for each prediction
        # CRITICAL: Only detect appropriate alerts based on variable type
        # - temp_max: Only heat wave alerts (high temperatures are problematic)
        # - temp_min: Only cold wave alerts (low temperatures are problematic)
        for i, (date, pred_value) in enumerate(zip(future_dates, predictions)):
            if pd.isna(pred_value):
                continue
            
            days_ahead = i + 1  # Days in the future (1-indexed)
            
            # For temp_min: Only detect cold wave alerts (low temperatures are problematic)
            if variable_type == "temp_min":
                cold_alert = self._detect_cold_wave_alert(
                    pred_value, date, thresholds, variable_type,
                    station_code, station_name, days_ahead
                )
                if cold_alert:
                    alerts.append(cold_alert)
            
            # For temp_max: Only detect heat wave alerts (high temperatures are problematic)
            elif variable_type == "temp_max":
                heat_alert = self._detect_heat_wave_alert(
                    pred_value, date, thresholds, variable_type,
                    station_code, station_name, days_ahead
                )
                if heat_alert:
                    alerts.append(heat_alert)
        
        if alerts:
            logger.info(f"Detected {len(alerts)} alerts for station {station_code} ({station_name})")
        
        return alerts
    
    def _detect_cold_wave_alert(
        self,
        pred_value: float,
        date: datetime,
        thresholds: TemperatureThresholds,
        variable_type: str,
        station_code: str,
        station_name: str,
        days_ahead: int
    ) -> Optional[TemperatureAlert]:
        """
        Detect cold wave alert (predicted value below threshold).
        
        Args:
            pred_value: Predicted temperature value
            date: Prediction date
            thresholds: Temperature thresholds
            variable_type: "temp_max" or "temp_min"
            station_code: Station code
            station_name: Station name
            days_ahead: Days ahead in prediction
            
        Returns:
            TemperatureAlert if detected, None otherwise
        """
        # Get thresholds
        extreme_threshold = thresholds.get_cold_wave_threshold(variable_type, extreme=True)
        moderate_threshold = thresholds.get_cold_wave_threshold(variable_type, extreme=False)
        
        if extreme_threshold is None or moderate_threshold is None:
            return None
        
        # Check for extreme cold wave (percentil 5)
        if pred_value <= extreme_threshold:
            return TemperatureAlert(
                station_code=station_code,
                station_name=station_name,
                alert_type=AlertType.COLD_WAVE_EXTREME,
                severity=AlertSeverity.CRITICAL,
                date=date,
                predicted_value=pred_value,
                threshold_value=extreme_threshold,
                variable_type=variable_type,
                days_ahead=days_ahead,
                message=(
                    f"⚠️ ALERTA CRÍTICA: Ola de frío extrema detectada "
                    f"({variable_type} = {pred_value:.1f}°C, umbral: {extreme_threshold:.1f}°C) "
                    f"en {date.strftime('%Y-%m-%d')} (+{days_ahead} días)"
                )
            )
        
        # Check for moderate cold wave (percentil 10)
        elif pred_value <= moderate_threshold:
            return TemperatureAlert(
                station_code=station_code,
                station_name=station_name,
                alert_type=AlertType.COLD_WAVE,
                severity=AlertSeverity.WARNING,
                date=date,
                predicted_value=pred_value,
                threshold_value=moderate_threshold,
                variable_type=variable_type,
                days_ahead=days_ahead,
                message=(
                    f"[WARNING] ALERTA: Ola de frío moderada detectada "
                    f"({variable_type} = {pred_value:.1f}°C, umbral: {moderate_threshold:.1f}°C) "
                    f"en {date.strftime('%Y-%m-%d')} (+{days_ahead} días)"
                )
            )
        
        return None
    
    def _detect_heat_wave_alert(
        self,
        pred_value: float,
        date: datetime,
        thresholds: TemperatureThresholds,
        variable_type: str,
        station_code: str,
        station_name: str,
        days_ahead: int
    ) -> Optional[TemperatureAlert]:
        """
        Detect heat wave alert (predicted value above threshold).
        
        Args:
            pred_value: Predicted temperature value
            date: Prediction date
            thresholds: Temperature thresholds
            variable_type: "temp_max" or "temp_min"
            station_code: Station code
            station_name: Station name
            days_ahead: Days ahead in prediction
            
        Returns:
            TemperatureAlert if detected, None otherwise
        """
        # Get thresholds
        extreme_threshold = thresholds.get_heat_wave_threshold(variable_type, extreme=True)
        moderate_threshold = thresholds.get_heat_wave_threshold(variable_type, extreme=False)
        
        if extreme_threshold is None or moderate_threshold is None:
            return None
        
        # Check for extreme heat wave (percentil 95)
        if pred_value >= extreme_threshold:
            return TemperatureAlert(
                station_code=station_code,
                station_name=station_name,
                alert_type=AlertType.HEAT_WAVE_EXTREME,
                severity=AlertSeverity.CRITICAL,
                date=date,
                predicted_value=pred_value,
                threshold_value=extreme_threshold,
                variable_type=variable_type,
                days_ahead=days_ahead,
                message=(
                    f"[CRITICAL] ALERTA CRÍTICA: Ola de calor extrema detectada "
                    f"({variable_type} = {pred_value:.1f}°C, umbral: {extreme_threshold:.1f}°C) "
                    f"en {date.strftime('%Y-%m-%d')} (+{days_ahead} días)"
                )
            )
        
        # Check for moderate heat wave (percentil 90)
        elif pred_value >= moderate_threshold:
            return TemperatureAlert(
                station_code=station_code,
                station_name=station_name,
                alert_type=AlertType.HEAT_WAVE,
                severity=AlertSeverity.WARNING,
                date=date,
                predicted_value=pred_value,
                threshold_value=moderate_threshold,
                variable_type=variable_type,
                days_ahead=days_ahead,
                message=(
                    f"[WARNING] ALERTA: Ola de calor moderada detectada "
                    f"({variable_type} = {pred_value:.1f}°C, umbral: {moderate_threshold:.1f}°C) "
                    f"en {date.strftime('%Y-%m-%d')} (+{days_ahead} días)"
                )
            )
        
        return None
    
    def summarize_alerts(self, alerts: List[TemperatureAlert]) -> Dict:
        """
        Summarize alerts by type and severity.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Dictionary with summary statistics
        """
        if not alerts:
            return {
                'total': 0,
                'by_type': {},
                'by_severity': {},
                'by_station': {}
            }
        
        summary = {
            'total': len(alerts),
            'by_type': {},
            'by_severity': {},
            'by_station': {}
        }
        
        for alert in alerts:
            # Count by type
            alert_type = alert.alert_type.value
            summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1
            
            # Count by severity
            severity = alert.severity.value
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # Count by station
            station_key = f"{alert.station_code}_{alert.station_name}"
            summary['by_station'][station_key] = summary['by_station'].get(station_key, 0) + 1
        
        return summary


