"""
Station Statistics Reporter Service

This module provides a service for generating statistical reports comparing
meteorological data before and after imputation for each station.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import json

from ....core.exceptions import ValidationError


class ReportFormat(Enum):
    """Enumeration for report output formats."""
    JSON = "json"
    CSV = "csv"
    BOTH = "both"


@dataclass
class StatisticalMetrics:
    """Data class for statistical metrics."""
    count: int
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    max: Optional[float]
    median: Optional[float]
    q25: Optional[float]
    q75: Optional[float]
    skewness: Optional[float]
    kurtosis: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'count': self.count,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'median': self.median,
            'q25': self.q25,
            'q75': self.q75,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis
        }


@dataclass
class TrendAnalysis:
    """Data class for trend analysis results."""
    slope: Optional[float]
    intercept: Optional[float]
    r_squared: Optional[float]
    p_value: Optional[float]
    trend_direction: str
    trend_strength: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'slope': self.slope,
            'intercept': self.intercept,
            'r_squared': self.r_squared,
            'p_value': self.p_value,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength
        }


@dataclass
class QualityMetrics:
    """Data class for data quality metrics."""
    completeness: Dict[str, float]
    consistency: Dict[str, float]
    outliers: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'completeness': self.completeness,
            'consistency': self.consistency,
            'outliers': self.outliers
        }


@dataclass
class MissingValueAnalysis:
    """Data class for missing value analysis."""
    original_missing: int
    imputed_missing: int
    original_missing_percentage: float
    imputed_missing_percentage: float
    imputed_values: int
    imputation_rate: float
    remaining_missing: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'original_missing': self.original_missing,
            'imputed_missing': self.imputed_missing,
            'original_missing_percentage': self.original_missing_percentage,
            'imputed_missing_percentage': self.imputed_missing_percentage,
            'imputed_values': self.imputed_values,
            'imputation_rate': self.imputation_rate,
            'remaining_missing': self.remaining_missing
        }


@dataclass
class StationReport:
    """Data class representing a complete station report."""
    station_info: Dict[str, str]
    data_summary: Dict[str, int]
    statistics: Dict[str, StatisticalMetrics]
    trend_analysis: Dict[str, TrendAnalysis]
    quality_metrics: QualityMetrics
    missing_value_analysis: MissingValueAnalysis
    summary_comparison: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'station_info': self.station_info,
            'data_summary': self.data_summary,
            'statistics': {
                'original': self.statistics['original'].to_dict(),
                'imputed': self.statistics['imputed'].to_dict()
            },
            'trend_analysis': {
                'original': self.trend_analysis['original'].to_dict(),
                'imputed': self.trend_analysis['imputed'].to_dict()
            },
            'quality_metrics': self.quality_metrics.to_dict(),
            'missing_value_analysis': self.missing_value_analysis.to_dict(),
            'summary_comparison': self.summary_comparison
        }


class StationStatisticsReporter:
    """
    Service for generating statistical reports comparing data before and after imputation.
    
    This service calculates key statistical metrics for each station and generates
    easy-to-read reports showing the impact of imputation on data quality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the station statistics reporter.
        
        Args:
            config: Configuration dictionary with service settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Results storage
        self.station_reports: Dict[str, StationReport] = {}
        
    def generate_station_report(self, 
                              original_data: pd.DataFrame, 
                              imputed_data: pd.DataFrame,
                              station_name: str,
                              station_code: str,
                              target_column: str) -> Dict[str, Any]:
        """
        Generate a comprehensive statistical report for a single station.
        
        Args:
            original_data: DataFrame with original data
            imputed_data: DataFrame with imputed data
            station_name: Name of the station
            station_code: Code of the station
            target_column: Name of the column to analyze
            
        Returns:
            Dictionary with comprehensive station statistics
            
        Raises:
            ValidationError: If report generation fails
        """
        try:
            self.logger.info(f"Generating statistics report for station: {station_name} ({station_code})")
            
            # Validate input data
            self._validate_input_data(original_data, imputed_data, target_column)
            
            # Calculate all metrics
            original_stats = self._calculate_basic_statistics(original_data, target_column)
            imputed_stats = self._calculate_basic_statistics(imputed_data, target_column)
            
            original_trend = self._calculate_trend(original_data, target_column)
            imputed_trend = self._calculate_trend(imputed_data, target_column)
            
            quality_metrics = self._calculate_quality_metrics(original_data, imputed_data, target_column)
            missing_stats = self._calculate_missing_statistics(original_data, imputed_data, target_column)
            
            # Create comprehensive report
            report = StationReport(
                station_info=self._create_station_info(station_name, station_code, target_column),
                data_summary=self._create_data_summary(original_data, imputed_data),
                statistics={'original': original_stats, 'imputed': imputed_stats},
                trend_analysis={'original': original_trend, 'imputed': imputed_trend},
                quality_metrics=quality_metrics,
                missing_value_analysis=missing_stats,
                summary_comparison=self._generate_summary_comparison(
                    original_stats, imputed_stats, original_trend, imputed_trend
                )
            )
            
            # Store report
            self.station_reports[station_name] = report
            
            self.logger.info(f"Statistics report generated successfully for station {station_name}")
            return report.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error generating statistics report for station {station_name}: {e}")
            raise ValidationError(f"Failed to generate statistics report for station {station_name}: {e}")
    
    def _validate_input_data(self, original_data: pd.DataFrame, imputed_data: pd.DataFrame, 
                           target_column: str) -> None:
        """
        Validate input data for statistical analysis.
        
        Args:
            original_data: DataFrame with original data
            imputed_data: DataFrame with imputed data
            target_column: Name of the column to analyze
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(original_data, pd.DataFrame) or not isinstance(imputed_data, pd.DataFrame):
            raise ValidationError("Both original_data and imputed_data must be pandas DataFrames")
        
        if original_data.empty or imputed_data.empty:
            raise ValidationError("Both original_data and imputed_data must not be empty")
        
        if target_column not in original_data.columns or target_column not in imputed_data.columns:
            raise ValidationError(f"Target column '{target_column}' not found in both datasets")
        
        # Check if target column is numeric
        if not pd.api.types.is_numeric_dtype(original_data[target_column]):
            self.logger.warning(f"Target column '{target_column}' in original data is not numeric")
        
        if not pd.api.types.is_numeric_dtype(imputed_data[target_column]):
            self.logger.warning(f"Target column '{target_column}' in imputed data is not numeric")
    
    def _create_station_info(self, station_name: str, station_code: str, target_column: str) -> Dict[str, str]:
        """Create station information dictionary."""
        return {
            'name': station_name,
            'code': station_code,
            'report_date': datetime.now().isoformat(),
            'target_column': target_column
        }
    
    def _create_data_summary(self, original_data: pd.DataFrame, imputed_data: pd.DataFrame) -> Dict[str, int]:
        """Create data summary dictionary."""
        return {
            'original_records': len(original_data),
            'imputed_records': len(imputed_data),
            'records_difference': len(imputed_data) - len(original_data)
        }
    
    def _calculate_basic_statistics(self, data: pd.DataFrame, target_column: str) -> StatisticalMetrics:
        """
        Calculate basic statistical metrics for the data.
        
        Args:
            data: DataFrame with data to analyze
            target_column: Name of the column to analyze
            
        Returns:
            StatisticalMetrics object with calculated statistics
        """
        try:
            # Convert to numeric, handling errors
            numeric_data = pd.to_numeric(data[target_column], errors='coerce')
            
            # Remove NaN values for calculations
            clean_data = numeric_data.dropna()
            
            if len(clean_data) == 0:
                return StatisticalMetrics(
                    count=0, mean=None, std=None, min=None, max=None,
                    median=None, q25=None, q75=None, skewness=None, kurtosis=None
                )
            
            # Calculate basic statistics
            stats_dict = {
                'count': len(clean_data),
                'mean': float(clean_data.mean()),
                'std': float(clean_data.std()),
                'min': float(clean_data.min()),
                'max': float(clean_data.max()),
                'median': float(clean_data.median()),
                'q25': float(clean_data.quantile(0.25)),
                'q75': float(clean_data.quantile(0.75)),
                'skewness': float(clean_data.skew()),
                'kurtosis': float(clean_data.kurtosis())
            }
            
            # Round numeric values for readability
            for key, value in stats_dict.items():
                if isinstance(value, float) and value is not None:
                    stats_dict[key] = round(value, 3)
            
            return StatisticalMetrics(**stats_dict)
            
        except Exception as e:
            self.logger.warning(f"Error calculating basic statistics: {e}")
            return StatisticalMetrics(
                count=0, mean=None, std=None, min=None, max=None,
                median=None, q25=None, q75=None, skewness=None, kurtosis=None
            )
    
    def _calculate_trend(self, data: pd.DataFrame, target_column: str) -> TrendAnalysis:
        """
        Calculate trend analysis for the time series data.
        
        Args:
            data: DataFrame with time series data
            target_column: Name of the column to analyze
            
        Returns:
            TrendAnalysis object with trend analysis results
        """
        try:
            # Convert to numeric
            numeric_data = pd.to_numeric(data[target_column], errors='coerce')
            clean_data = numeric_data.dropna()
            
            if len(clean_data) < 10:
                return TrendAnalysis(
                    slope=None, intercept=None, r_squared=None, p_value=None,
                    trend_direction='insufficient_data', trend_strength='insufficient_data'
                )
            
            # Calculate linear trend
            x = np.arange(len(clean_data))
            y = clean_data.values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value ** 2
            
            # Determine trend direction and strength
            if p_value < 0.05:  # Statistically significant
                if slope > 0:
                    trend_direction = 'increasing'
                else:
                    trend_direction = 'decreasing'
                
                if abs(r_squared) > 0.7:
                    trend_strength = 'strong'
                elif abs(r_squared) > 0.3:
                    trend_strength = 'moderate'
                else:
                    trend_strength = 'weak'
            else:
                trend_direction = 'no_significant_trend'
                trend_strength = 'no_significant_trend'
            
            return TrendAnalysis(
                slope=round(slope, 6),
                intercept=round(intercept, 3),
                r_squared=round(r_squared, 3),
                p_value=round(p_value, 6),
                trend_direction=trend_direction,
                trend_strength=trend_strength
            )
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend analysis: {e}")
            return TrendAnalysis(
                slope=None, intercept=None, r_squared=None, p_value=None,
                trend_direction='error', trend_strength='error'
            )
    
    def _calculate_quality_metrics(self, original_data: pd.DataFrame, imputed_data: pd.DataFrame, 
                                 target_column: str) -> QualityMetrics:
        """
        Calculate data quality metrics comparing original and imputed data.
        
        Args:
            original_data: DataFrame with original data
            imputed_data: DataFrame with imputed data
            target_column: Name of the column to analyze
            
        Returns:
            QualityMetrics object with quality metrics
        """
        try:
            original_numeric = pd.to_numeric(original_data[target_column], errors='coerce')
            imputed_numeric = pd.to_numeric(imputed_data[target_column], errors='coerce')
            
            # Completeness
            original_completeness = 1 - (original_numeric.isnull().sum() / len(original_numeric))
            imputed_completeness = 1 - (imputed_numeric.isnull().sum() / len(imputed_numeric))
            
            # Consistency (coefficient of variation)
            original_clean = original_numeric.dropna()
            imputed_clean = imputed_numeric.dropna()
            
            original_cv = original_clean.std() / original_clean.mean() if original_clean.mean() != 0 else 0
            imputed_cv = imputed_clean.std() / imputed_clean.mean() if imputed_clean.mean() != 0 else 0
            
            # Outlier detection (using IQR method)
            original_outliers = self._count_outliers(original_clean)
            imputed_outliers = self._count_outliers(imputed_clean)
            
            return QualityMetrics(
                completeness={
                    'original': round(original_completeness, 3),
                    'imputed': round(imputed_completeness, 3),
                    'improvement': round(imputed_completeness - original_completeness, 3)
                },
                consistency={
                    'original_cv': round(original_cv, 3),
                    'imputed_cv': round(imputed_cv, 3),
                    'cv_change': round(imputed_cv - original_cv, 3)
                },
                outliers={
                    'original_count': int(original_outliers),
                    'imputed_count': int(imputed_outliers),
                    'original_percentage': round(original_outliers / len(original_clean) * 100, 2) if len(original_clean) > 0 else 0,
                    'imputed_percentage': round(imputed_outliers / len(imputed_clean) * 100, 2) if len(imputed_clean) > 0 else 0
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Error calculating quality metrics: {e}")
            return QualityMetrics(
                completeness={'original': 0, 'imputed': 0, 'improvement': 0},
                consistency={'original_cv': 0, 'imputed_cv': 0, 'cv_change': 0},
                outliers={'original_count': 0, 'imputed_count': 0, 'original_percentage': 0, 'imputed_percentage': 0}
            )
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers in a series using IQR method."""
        if len(series) == 0:
            return 0
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def _calculate_missing_statistics(self, original_data: pd.DataFrame, imputed_data: pd.DataFrame, 
                                    target_column: str) -> MissingValueAnalysis:
        """
        Calculate missing value statistics.
        
        Args:
            original_data: DataFrame with original data
            imputed_data: DataFrame with imputed data
            target_column: Name of the column to analyze
            
        Returns:
            MissingValueAnalysis object with missing value statistics
        """
        try:
            original_missing = original_data[target_column].isnull().sum()
            imputed_missing = imputed_data[target_column].isnull().sum()
            
            original_total = len(original_data)
            imputed_total = len(imputed_data)
            
            original_missing_pct = (original_missing / original_total) * 100 if original_total > 0 else 0
            imputed_missing_pct = (imputed_missing / imputed_total) * 100 if imputed_total > 0 else 0
            
            imputed_values = original_missing - imputed_missing
            imputation_rate = (imputed_values / original_missing) * 100 if original_missing > 0 else 0
            
            return MissingValueAnalysis(
                original_missing=int(original_missing),
                imputed_missing=int(imputed_missing),
                original_missing_percentage=round(original_missing_pct, 2),
                imputed_missing_percentage=round(imputed_missing_pct, 2),
                imputed_values=int(imputed_values),
                imputation_rate=round(imputation_rate, 2),
                remaining_missing=int(imputed_missing)
            )
            
        except Exception as e:
            self.logger.warning(f"Error calculating missing statistics: {e}")
            return MissingValueAnalysis(
                original_missing=0, imputed_missing=0, original_missing_percentage=0,
                imputed_missing_percentage=0, imputed_values=0, imputation_rate=0, remaining_missing=0
            )
    
    def _generate_summary_comparison(self, original_stats: StatisticalMetrics, 
                                   imputed_stats: StatisticalMetrics,
                                   original_trend: TrendAnalysis, 
                                   imputed_trend: TrendAnalysis) -> Dict[str, Any]:
        """
        Generate a summary comparison of key metrics.
        
        Args:
            original_stats: Statistics for original data
            imputed_stats: Statistics for imputed data
            original_trend: Trend analysis for original data
            imputed_trend: Trend analysis for imputed data
            
        Returns:
            Dictionary with summary comparison
        """
        try:
            # Calculate percentage changes
            def safe_percentage_change(original, imputed):
                if original is None or imputed is None or original == 0:
                    return None
                return round(((imputed - original) / original) * 100, 2)
            
            summary = {
                'key_metrics_changes': {
                    'mean_change_pct': safe_percentage_change(original_stats.mean, imputed_stats.mean),
                    'std_change_pct': safe_percentage_change(original_stats.std, imputed_stats.std),
                    'min_change_pct': safe_percentage_change(original_stats.min, imputed_stats.min),
                    'max_change_pct': safe_percentage_change(original_stats.max, imputed_stats.max)
                },
                'trend_comparison': {
                    'original_direction': original_trend.trend_direction,
                    'imputed_direction': imputed_trend.trend_direction,
                    'original_strength': original_trend.trend_strength,
                    'imputed_strength': imputed_trend.trend_strength,
                    'r_squared_change': safe_percentage_change(
                        original_trend.r_squared, imputed_trend.r_squared
                    )
                },
                'data_quality_impact': {
                    'completeness_improved': imputed_stats.count > original_stats.count,
                    'variance_preserved': abs(imputed_stats.std - original_stats.std) < original_stats.std * 0.1 if original_stats.std else False,
                    'range_preserved': (
                        abs(imputed_stats.min - original_stats.min) < original_stats.std * 0.5 and
                        abs(imputed_stats.max - original_stats.max) < original_stats.std * 0.5
                    ) if original_stats.std else False
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"Error generating summary comparison: {e}")
            return {
                'key_metrics_changes': {},
                'trend_comparison': {},
                'data_quality_impact': {}
            }
    
    def generate_concise_report(self, station_name: str) -> str:
        """
        Generate a concise text report for a station.
        
        Args:
            station_name: Name of the station
            
        Returns:
            String with concise report
        """
        if station_name not in self.station_reports:
            return f"No report available for station: {station_name}"
        
        report = self.station_reports[station_name]
        stats = report.statistics
        missing = report.missing_value_analysis
        trend = report.trend_analysis
        
        concise_report = f"""
Station: {report.station_info['name']} ({report.station_info['code']})
Report Date: {report.station_info['report_date']}

Data Summary:
- Original records: {report.data_summary['original_records']}
- Imputed records: {report.data_summary['imputed_records']}
- Records difference: {report.data_summary['records_difference']}

Statistical Comparison:
- Mean: {stats['original'].mean:.2f} → {stats['imputed'].mean:.2f}
- Standard Deviation: {stats['original'].std:.2f} → {stats['imputed'].std:.2f}
- Min: {stats['original'].min:.2f} → {stats['imputed'].min:.2f}
- Max: {stats['original'].max:.2f} → {stats['imputed'].max:.2f}

Trend Analysis:
- Original trend: {trend['original'].trend_direction} ({trend['original'].trend_strength})
- Imputed trend: {trend['imputed'].trend_direction} ({trend['imputed'].trend_strength})

Missing Value Analysis:
- Original missing: {missing.original_missing} ({missing.original_missing_percentage:.1f}%)
- Imputed values: {missing.imputed_values} ({missing.imputation_rate:.1f}%)
- Remaining missing: {missing.remaining_missing} ({missing.imputed_missing_percentage:.1f}%)

Quality Impact:
- Completeness improved: {report.summary_comparison['data_quality_impact']['completeness_improved']}
- Variance preserved: {report.summary_comparison['data_quality_impact']['variance_preserved']}
- Range preserved: {report.summary_comparison['data_quality_impact']['range_preserved']}
"""
        
        return concise_report.strip()
    
    def save_reports(self, output_path: Path, format: str = ReportFormat.JSON.value) -> None:
        """
        Save all generated reports to files.
        
        Args:
            output_path: Directory path to save reports
            format: Output format (json, csv, or both)
        """
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            if format in [ReportFormat.JSON.value, ReportFormat.BOTH.value]:
                self._save_json_reports(output_path)
            
            if format in [ReportFormat.CSV.value, ReportFormat.BOTH.value]:
                self._save_csv_reports(output_path)
            
            # Save summary CSV
            self._save_summary_csv(output_path)
            
            self.logger.info(f"Reports saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving reports: {e}")
            raise ValidationError(f"Failed to save reports: {e}")
    
    def _save_json_reports(self, output_path: Path) -> None:
        """Save reports in JSON format."""
        for station_name, report in self.station_reports.items():
            safe_station_name = station_name.replace(' ', '_').replace(',', '').replace('.', '')[:20]
            filename = f"{safe_station_name}_statistics_report.json"
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _save_csv_reports(self, output_path: Path) -> None:
        """Save reports in CSV format."""
        for station_name, report in self.station_reports.items():
            safe_station_name = station_name.replace(' ', '_').replace(',', '').replace('.', '')[:20]
            filename = f"{safe_station_name}_statistics_report.csv"
            filepath = output_path / filename
            
            # Create CSV data
            csv_data = []
            stats = report.statistics
            
            # Add basic statistics
            for data_type in ['original', 'imputed']:
                stat = stats[data_type]
                csv_data.append({
                    'metric': f'{data_type}_count', 'value': stat.count
                })
                csv_data.append({
                    'metric': f'{data_type}_mean', 'value': stat.mean
                })
                csv_data.append({
                    'metric': f'{data_type}_std', 'value': stat.std
                })
                csv_data.append({
                    'metric': f'{data_type}_min', 'value': stat.min
                })
                csv_data.append({
                    'metric': f'{data_type}_max', 'value': stat.max
                })
            
            # Add missing value analysis
            missing = report.missing_value_analysis
            csv_data.append({
                'metric': 'original_missing', 'value': missing.original_missing
            })
            csv_data.append({
                'metric': 'imputed_values', 'value': missing.imputed_values
            })
            csv_data.append({
                'metric': 'imputation_rate', 'value': missing.imputation_rate
            })
            
            # Save CSV
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
    
    def _save_summary_csv(self, output_path: Path) -> None:
        """Save a summary CSV with key metrics for all stations."""
        if not self.station_reports:
            return
        
        summary_data = []
        
        for station_name, report in self.station_reports.items():
            stats = report.statistics
            missing = report.missing_value_analysis
            trend = report.trend_analysis
            
            summary_data.append({
                'station_name': station_name,
                'station_code': report.station_info['code'],
                'original_records': report.data_summary['original_records'],
                'imputed_records': report.data_summary['imputed_records'],
                'original_mean': stats['original'].mean,
                'imputed_mean': stats['imputed'].mean,
                'original_std': stats['original'].std,
                'imputed_std': stats['imputed'].std,
                'original_missing': missing.original_missing,
                'imputed_values': missing.imputed_values,
                'imputation_rate': missing.imputation_rate,
                'original_trend': trend['original'].trend_direction,
                'imputed_trend': trend['imputed'].trend_direction,
                'completeness_improved': report.summary_comparison['data_quality_impact']['completeness_improved'],
                'variance_preserved': report.summary_comparison['data_quality_impact']['variance_preserved'],
                'range_preserved': report.summary_comparison['data_quality_impact']['range_preserved']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filepath = output_path / "stations_summary.csv"
        summary_df.to_csv(summary_filepath, index=False, encoding='utf-8')
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """
        Get an overall summary of all station reports.
        
        Returns:
            Dictionary with overall summary statistics
        """
        if not self.station_reports:
            return {
                'total_stations': 0,
                'message': 'No station reports available'
            }
        
        total_stations = len(self.station_reports)
        
        # Calculate averages
        imputation_rates = []
        mean_changes = []
        std_changes = []
        
        for report in self.station_reports.values():
            missing = report.missing_value_analysis
            stats = report.statistics
            
            if missing.imputation_rate is not None:
                imputation_rates.append(missing.imputation_rate)
            
            if stats['original'].mean is not None and stats['imputed'].mean is not None:
                if stats['original'].mean != 0:
                    mean_change = ((stats['imputed'].mean - stats['original'].mean) / stats['original'].mean) * 100
                    mean_changes.append(mean_change)
            
            if stats['original'].std is not None and stats['imputed'].std is not None:
                if stats['original'].std != 0:
                    std_change = ((stats['imputed'].std - stats['original'].std) / stats['original'].std) * 100
                    std_changes.append(std_change)
        
        return {
            'total_stations': total_stations,
            'average_imputation_rate': np.mean(imputation_rates) if imputation_rates else 0,
            'mean_statistics': {
                'mean_change_pct': np.mean(mean_changes) if mean_changes else 0
            },
            'std_statistics': {
                'std_change_pct': np.mean(std_changes) if std_changes else 0
            }
        }
    
    def reset(self) -> None:
        """Reset the service state."""
        self.station_reports.clear()
        self.logger.info("Service state reset") 