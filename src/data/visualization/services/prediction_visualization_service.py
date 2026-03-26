"""
Prediction Visualization Service

Este módulo provee un servicio especializado para generar visualizaciones
de resultados de predicciones meteorológicas.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path
import numpy as np

from ..plotters.prediction_plotter import PredictionPlotter
from src.core.interfaces.prediction_strategy import PredictionResult
from src.data.prediction.services.alert_detector import TemperatureAlert


class PredictionVisualizationService:
    """
    Servicio especializado para visualización de predicciones meteorológicas.
    
    Este servicio maneja la generación de plots para:
    - Series temporales de predicciones
    - Comparaciones histórico vs predicción
    - Intervalos de confianza
    - Descomposición estacional
    - Análisis comprehensivo
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar el servicio de visualización de predicciones.
        
        Args:
            config: Configuración opcional para el plotter
        """
        self.config = config or {}
        self.plotter = PredictionPlotter(self.config)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("PredictionVisualizationService initialized")
    
    def generate_prediction_plots(self, 
                                prediction_result: PredictionResult,
                                station_name: str,
                                output_dir: Path,
                                include_confidence: bool = True,
                                alerts: Optional[List[TemperatureAlert]] = None) -> Dict[str, str]:
        """
        Generar todos los plots de predicción para una estación.
        
        Args:
            prediction_result: Resultado de la predicción
            station_name: Nombre de la estación
            output_dir: Directorio de salida para los plots
            include_confidence: Si incluir intervalos de confianza
            
        Returns:
            Diccionario con las rutas de los archivos generados
        """
        try:
            self.logger.info(f"Generating prediction plots for station: {station_name}")
            
            # Crear directorio de salida
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Force garbage collection before plot generation
            import gc
            gc.collect()
            
            # Preparar datos para los plots
            plot_data = self._prepare_plot_data(prediction_result, include_confidence)
            
            # Generar plots
            generated_files = {}
            
            # 1. Plot de serie temporal (with alerts if available - FASE 3)
            try:
                time_series_file = output_dir / f"{station_name}_time_series_plot.png"
                self.plotter.plot_time_series(plot_data, str(time_series_file), dpi=300, alerts=alerts, station_name=station_name)
                generated_files['time_series_plot'] = str(time_series_file)
                self.logger.info(f"   [OK] Generated time series plot: {time_series_file}")
                gc.collect()  # Clean memory after each plot
            except Exception as e:
                self.logger.error(f"   [ERROR] Failed to generate time series plot: {e}")
            
            # 2. Plot de comparación
            try:
                comparison_file = output_dir / f"{station_name}_comparison_plot.png"
                self.plotter.plot_comparison(plot_data, str(comparison_file), dpi=300, station_name=station_name)
                generated_files['comparison_plot'] = str(comparison_file)
                self.logger.info(f"   [OK] Generated comparison plot: {comparison_file}")
                gc.collect()  # Clean memory after each plot
            except Exception as e:
                self.logger.error(f"   [ERROR] Failed to generate comparison plot: {e}")
            
            # 3. Plot de intervalos de confianza
            if include_confidence and self._has_confidence_data(prediction_result):
                try:
                    confidence_file = output_dir / f"{station_name}_confidence_intervals_plot.png"
                    self.plotter.plot_confidence_intervals(plot_data, str(confidence_file), dpi=300, station_name=station_name)
                    generated_files['confidence_intervals_plot'] = str(confidence_file)
                    self.logger.info(f"   [OK] Generated confidence intervals plot: {confidence_file}")
                    gc.collect()  # Clean memory after each plot
                except Exception as e:
                    self.logger.error(f"   [ERROR] Failed to generate confidence intervals plot: {e}")
            
            # 4. Plot de descomposición estacional
            if self._has_imf_data(prediction_result):
                try:
                    seasonal_file = output_dir / f"{station_name}_seasonal_decomposition_plot.png"
                    self.plotter.plot_seasonal_decomposition(plot_data, str(seasonal_file), dpi=300)
                    generated_files['seasonal_decomposition_plot'] = str(seasonal_file)
                    self.logger.info(f"   [OK] Generated seasonal decomposition plot: {seasonal_file}")
                    gc.collect()  # Clean memory after each plot
                except Exception as e:
                    self.logger.error(f"   [ERROR] Failed to generate seasonal decomposition plot: {e}")
            
            # 5. Plot comprehensivo
            try:
                comprehensive_file = output_dir / f"{station_name}_comprehensive_plot.png"
                self.plotter.plot_comprehensive(plot_data, str(comprehensive_file), dpi=300)
                generated_files['comprehensive_plot'] = str(comprehensive_file)
                self.logger.info(f"   [OK] Generated comprehensive plot: {comprehensive_file}")
                gc.collect()  # Clean memory after each plot
            except Exception as e:
                self.logger.error(f"   [ERROR] Failed to generate comprehensive plot: {e}")
            
            self.logger.info(f"Generated {len(generated_files)} prediction plots for station {station_name}")
            return generated_files
            
        except Exception as e:
            error_msg = f"Failed to generate prediction plots for station {station_name}: {e}"
            self.logger.error(error_msg)
            raise
    
    def generate_summary_report(self, 
                              prediction_result: PredictionResult,
                              station_name: str,
                              output_dir: Path) -> str:
        """
        Generar reporte resumen de predicciones.
        
        Args:
            prediction_result: Resultado de la predicción
            station_name: Nombre de la estación
            output_dir: Directorio de salida
            
        Returns:
            Ruta del archivo de reporte generado
        """
        try:
            self.logger.info(f"Generating prediction summary report for station: {station_name}")
            
            # Crear directorio de salida
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Preparar datos para el reporte
            report_data = self._prepare_report_data(prediction_result, station_name)
            
            # Generar reporte CSV
            report_file = output_dir / f"{station_name}_prediction_summary.csv"
            report_df = pd.DataFrame([report_data])
            report_df.to_csv(report_file, index=False)
            
            self.logger.info(f"Prediction summary report saved: {report_file}")
            return str(report_file)
            
        except Exception as e:
            error_msg = f"Failed to generate prediction summary report for station {station_name}: {e}"
            self.logger.error(error_msg)
            raise
    
    def _prepare_plot_data(self, prediction_result: PredictionResult, include_confidence: bool = True) -> pd.DataFrame:
        """
        Preparar datos para los plots de predicción.
        CRITICAL FIX: Uses real dates from original data to avoid gaps.
        
        Args:
            prediction_result: Resultado de la predicción
            include_confidence: Si incluir intervalos de confianza
            
        Returns:
            DataFrame preparado para plotting
        """
        plot_data = []
        
        # Agregar datos históricos (últimos 30 días para contexto)
        if prediction_result.original_data is not None:
            original_values = prediction_result.original_data.iloc[:, 0]  # Primera columna
            context_days = min(30, len(original_values))
            
            # CRITICAL FIX: Use real dates from original data index if available
            if isinstance(original_values.index, pd.DatetimeIndex):
                # Use actual dates from the original series
                historical_dates = original_values.index[-context_days:]
                self.logger.debug(f"Using real dates from original data: {historical_dates[0]} to {historical_dates[-1]}")
            elif prediction_result.future_dates is not None and len(prediction_result.future_dates) > 0:
                # Fallback: generate dates backwards from first prediction date
                last_future_date = prediction_result.future_dates[0]
                historical_dates = pd.date_range(
                    end=last_future_date - pd.Timedelta(days=1),
                    periods=context_days,
                    freq='D'
                )
                self.logger.debug(f"Generated historical dates backwards from first prediction: {historical_dates[0]} to {historical_dates[-1]}")
            else:
                # Last resort: use current date
                context_days = min(30, len(original_values))
                today = pd.Timestamp.now().date()
                historical_dates = pd.date_range(
                    end=today,
                    periods=context_days,
                    freq='D'
                )
                self.logger.warning("Using current date as fallback for historical dates")
            
            # Agregar datos históricos con fechas reales
            for i, date in enumerate(historical_dates):
                value_idx = len(original_values) - context_days + i
                if value_idx >= 0 and value_idx < len(original_values):
                    plot_data.append({
                        'date': date,
                        'type': 'historical',
                        'value': original_values.iloc[value_idx]
                    })
            
            # CRITICAL: Verify connection - last historical date should be day before first prediction
            if prediction_result.future_dates is not None and len(prediction_result.future_dates) > 0:
                last_historical_date = historical_dates[-1]
                first_prediction_date = prediction_result.future_dates[0]
                expected_gap = (first_prediction_date - last_historical_date).days
                
                if expected_gap != 1:
                    self.logger.warning(f"Date gap detected: {expected_gap} days between last historical ({last_historical_date}) and first prediction ({first_prediction_date})")
                else:
                    self.logger.debug(f"Perfect date continuity: last historical {last_historical_date}, first prediction {first_prediction_date}")
        
        # CRITICAL: Add bridge point for perfect continuity
        # Include last historical point as first prediction point to ensure NO gap
        bridge_added = False
        if not plot_data and prediction_result.original_data is not None:
            # If we have historical data, get the last point
            original_values = prediction_result.original_data.iloc[:, 0]
            if isinstance(original_values.index, pd.DatetimeIndex) and len(original_values) > 0:
                last_hist_date = original_values.index[-1]
                last_hist_value = original_values.iloc[-1]
                
                # Check if first prediction date is next day
                if prediction_result.future_dates is not None and len(prediction_result.future_dates) > 0:
                    first_pred_date = prediction_result.future_dates[0]
                    if (first_pred_date - last_hist_date).days == 1:
                        # Add bridge point: last historical as first prediction point
                        plot_data.append({
                            'date': last_hist_date,
                            'type': 'bridge',  # Special type for continuity
                            'value': last_hist_value
                        })
                        bridge_added = True
        
        # Agregar predicciones
        if prediction_result.final_prediction is not None and prediction_result.future_dates is not None and len(prediction_result.future_dates) > 0:
            for i, (date, pred_value) in enumerate(zip(prediction_result.future_dates, prediction_result.final_prediction)):
                row_data = {
                    'date': date,
                    'type': 'prediction',
                    'value': pred_value
                }
                
                # Agregar intervalos de confianza si están disponibles
                if include_confidence and prediction_result.confidence_intervals is not None:
                    if i < len(prediction_result.confidence_intervals):
                        conf_interval = prediction_result.confidence_intervals[i]
                        if isinstance(conf_interval, (list, tuple)) and len(conf_interval) >= 2:
                            row_data['confidence_lower'] = conf_interval[0]
                            row_data['confidence_upper'] = conf_interval[1]
                
                # Agregar componentes IMF si están disponibles
                if prediction_result.imf_predictions is not None:
                    for imf_idx, imf_preds in prediction_result.imf_predictions.items():
                        if i < len(imf_preds):
                            row_data[f'imf_{imf_idx + 1}'] = imf_preds[i]
                
                # Agregar señal reconstruida si está disponible
                if prediction_result.reconstructed_signal is not None and i < len(prediction_result.reconstructed_signal):
                    row_data['reconstructed'] = prediction_result.reconstructed_signal[i]
                
                plot_data.append(row_data)
        
        # CRITICAL: Ensure perfect continuity in the DataFrame
        # Sort by date to ensure proper ordering
        plot_df = pd.DataFrame(plot_data)
        if not plot_df.empty and 'date' in plot_df.columns:
            plot_df = plot_df.sort_values('date').reset_index(drop=True)
            
            # Verify continuity: check if last historical and first prediction are consecutive
            historical = plot_df[plot_df['type'] == 'historical']
            predictions = plot_df[plot_df['type'] == 'prediction']
            
            if not historical.empty and not predictions.empty:
                last_hist_date = historical.iloc[-1]['date']
                first_pred_date = predictions.iloc[0]['date']
                gap = (first_pred_date - last_hist_date).days
                
                if gap == 1:
                    self.logger.debug(f"Perfect continuity verified in plot data: gap = {gap} day")
                else:
                    self.logger.warning(f"Continuity issue in plot data: gap = {gap} days")
        
        return plot_df
    
    def _prepare_report_data(self, prediction_result: PredictionResult, station_name: str) -> Dict[str, Any]:
        """
        Preparar datos para el reporte de predicciones.
        
        Args:
            prediction_result: Resultado de la predicción
            station_name: Nombre de la estación
            
        Returns:
            Diccionario con datos del reporte
        """
        report_data = {
            'station_name': station_name,
            'prediction_date': pd.Timestamp.now().isoformat(),
            'prediction_length': len(prediction_result.final_prediction) if prediction_result.final_prediction is not None else 0,
            'processing_time': prediction_result.processing_time,
            'success': prediction_result.success
        }
        
        # Agregar métricas de calidad si están disponibles
        if prediction_result.prediction_quality_metrics:
            metrics = prediction_result.prediction_quality_metrics
            report_data.update({
                'mean_consistency': metrics.get('mean_consistency', 0),
                'trend_consistency': metrics.get('trend_consistency', 0),
                'diversity_score': metrics.get('diversity_score', 0),
                'num_imfs_used': metrics.get('num_imfs_used', 0)
            })
        else:
            # Default values when no quality metrics
            report_data.update({
                'mean_consistency': 0,
                'trend_consistency': 0,
                'diversity_score': 0,
                'num_imfs_used': 0
            })
        
        # Agregar estadísticas de predicciones
        if prediction_result.final_prediction is not None and len(prediction_result.final_prediction) > 0:
            predictions_array = np.array(prediction_result.final_prediction)
            report_data.update({
                'prediction_mean': float(np.mean(predictions_array)),
                'prediction_std': float(np.std(predictions_array)),
                'prediction_min': float(np.min(predictions_array)),
                'prediction_max': float(np.max(predictions_array)),
                'prediction_range': float(np.max(predictions_array) - np.min(predictions_array))
            })
        else:
            # Default values when no predictions
            report_data.update({
                'prediction_mean': 0.0,
                'prediction_std': 0.0,
                'prediction_min': 0.0,
                'prediction_max': 0.0,
                'prediction_range': 0.0
            })
        
        # Agregar información de fechas
        if prediction_result.future_dates is not None and len(prediction_result.future_dates) > 0:
            report_data.update({
                'prediction_start_date': prediction_result.future_dates[0].isoformat(),
                'prediction_end_date': prediction_result.future_dates[-1].isoformat()
            })
        
        return report_data
    
    def _has_confidence_data(self, prediction_result: PredictionResult) -> bool:
        """Verificar si hay datos de intervalos de confianza."""
        return (prediction_result.confidence_intervals is not None and 
                len(prediction_result.confidence_intervals) > 0)
    
    def _has_imf_data(self, prediction_result: PredictionResult) -> bool:
        """Verificar si hay datos de componentes IMF."""
        return (prediction_result.imf_predictions is not None and 
                len(prediction_result.imf_predictions) > 0)
    
    def plot_single_prediction(self, 
                             prediction_result: PredictionResult,
                             plot_type: str = 'time_series',
                             save_path: Optional[str] = None,
                             **kwargs) -> None:
        """
        Generar un plot específico de predicción.
        
        Args:
            prediction_result: Resultado de la predicción
            plot_type: Tipo de plot a generar
            save_path: Ruta donde guardar el plot (opcional)
            **kwargs: Parámetros adicionales
        """
        try:
            # Preparar datos
            plot_data = self._prepare_plot_data(prediction_result, include_confidence=True)
            
            # Generar plot
            if plot_type == 'time_series':
                self.plotter.plot_time_series(plot_data, save_path, **kwargs)
            elif plot_type == 'comparison':
                self.plotter.plot_comparison(plot_data, save_path, **kwargs)
            elif plot_type == 'confidence_intervals':
                self.plotter.plot_confidence_intervals(plot_data, save_path, **kwargs)
            elif plot_type == 'seasonal_decomposition':
                self.plotter.plot_seasonal_decomposition(plot_data, save_path, **kwargs)
            elif plot_type == 'comprehensive':
                self.plotter.plot_comprehensive(plot_data, save_path, **kwargs)
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")
                
        except Exception as e:
            error_msg = f"Failed to generate {plot_type} plot: {e}"
            self.logger.error(error_msg)
            raise 