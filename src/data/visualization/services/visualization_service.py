"""
Visualization Service

Este módulo provee un servicio unificado para todas las visualizaciones
del sistema de predicción meteorológica.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path
import numpy as np

from .eemd_visualization_service import EEMDVisualizationService
from .prediction_visualization_service import PredictionVisualizationService
from src.core.interfaces.prediction_strategy import EEMDResult, PredictionResult

logger = logging.getLogger(__name__)


class VisualizationService:
    """
    Servicio unificado para visualizaciones.
    
    Este servicio unifica:
    - Visualizaciones EEMD
    - Visualizaciones de predicciones
    - Gestión de archivos de salida
    - Configuración de plots
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar el servicio de visualización.
        
        Args:
            config: Configuración opcional para los plotters
        """
        self.config = config or {}
        self.logger = logger
        
        # Inicializar servicios especializados
        self.eemd_visualization_service = EEMDVisualizationService(self.config)
        self.prediction_visualization_service = PredictionVisualizationService(self.config)
        
        self.logger.info("VisualizationService initialized")
    
    def generate_eemd_plots(self, 
                           eemd_result: EEMDResult,
                           station_name: str,
                           output_dir: Path,
                           original_series: Optional[pd.Series] = None) -> Dict[str, str]:
        """
        Generar plots de EEMD.
        
        Args:
            eemd_result: Resultado de la descomposición EEMD
            station_name: Nombre de la estación
            output_dir: Directorio de salida
            original_series: Serie temporal original (opcional)
            
        Returns:
            Diccionario con las rutas de los archivos generados
        """
        try:
            self.logger.info(f"Generating EEMD plots for station: {station_name}")
            
            # Crear directorio específico para EEMD
            eemd_output_dir = output_dir / "eemd_plots" / station_name
            eemd_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generar plots usando el servicio especializado
            generated_files = self.eemd_visualization_service.generate_eemd_plots(
                eemd_result, station_name, eemd_output_dir, original_series
            )
            
            self.logger.info(f"Generated {len(generated_files)} EEMD plots for {station_name}")
            return generated_files
            
        except Exception as e:
            self.logger.error(f"Failed to generate EEMD plots for {station_name}: {e}")
            raise
    
    def generate_prediction_plots(self, 
                                prediction_result: PredictionResult,
                                station_name: str,
                                output_dir: Path,
                                include_confidence: bool = True,
                                alerts: Optional[List] = None) -> Dict[str, str]:
        """
        Generar plots de predicciones.
        
        Args:
            prediction_result: Resultado de la predicción
            station_name: Nombre de la estación
            output_dir: Directorio de salida
            include_confidence: Si incluir intervalos de confianza
            alerts: Lista de alertas de temperatura (FASE 3)
            
        Returns:
            Diccionario con las rutas de los archivos generados
        """
        try:
            self.logger.info(f"Generating prediction plots for station: {station_name}")
            
            # Usar el directorio directamente (sin crear subdirectorio duplicado)
            prediction_output_dir = output_dir
            prediction_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Force garbage collection before plot generation
            import gc
            gc.collect()
            
            # Generar plots usando el servicio especializado (FASE 3: include alerts)
            generated_files = self.prediction_visualization_service.generate_prediction_plots(
                prediction_result, station_name, prediction_output_dir, include_confidence, alerts=alerts
            )
            
            if generated_files and len(generated_files) > 0:
                self.logger.info(f"Generated {len(generated_files)} prediction plots for {station_name}")
                return generated_files
            else:
                self.logger.warning(f"No plots generated for {station_name}")
                return {}
            
        except Exception as e:
            self.logger.error(f"Failed to generate prediction plots for {station_name}: {e}")
            raise
    
    def generate_comprehensive_plots(self,
                                   eemd_result: EEMDResult,
                                   prediction_result: PredictionResult,
                                   station_name: str,
                                   output_dir: Path) -> Dict[str, str]:
        """
        Generar plots comprehensivos que combinen EEMD y predicciones.
        
        Args:
            eemd_result: Resultado de la descomposición EEMD
            prediction_result: Resultado de la predicción
            station_name: Nombre de la estación
            output_dir: Directorio de salida
            
        Returns:
            Diccionario con las rutas de los archivos generados
        """
        try:
            self.logger.info(f"Generating comprehensive plots for station: {station_name}")
            
            # Crear directorio para plots comprehensivos
            comprehensive_output_dir = output_dir / "comprehensive_plots" / station_name
            comprehensive_output_dir.mkdir(parents=True, exist_ok=True)
            
            generated_files = {}
            
            # Generar plots EEMD
            eemd_files = self.generate_eemd_plots(eemd_result, station_name, comprehensive_output_dir)
            generated_files.update({f"eemd_{k}": v for k, v in eemd_files.items()})
            
            # Generar plots de predicciones
            prediction_files = self.generate_prediction_plots(prediction_result, station_name, comprehensive_output_dir)
            generated_files.update({f"prediction_{k}": v for k, v in prediction_files.items()})
            
            # Generar plot combinado (EEMD + Predicciones)
            combined_file = comprehensive_output_dir / f"{station_name}_combined_analysis.png"
            self._generate_combined_plot(eemd_result, prediction_result, station_name, combined_file)
            generated_files['combined_analysis'] = str(combined_file)
            
            self.logger.info(f"Generated {len(generated_files)} comprehensive plots for {station_name}")
            return generated_files
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive plots for {station_name}: {e}")
            raise
    
    def _generate_combined_plot(self,
                              eemd_result: EEMDResult,
                              prediction_result: PredictionResult,
                              station_name: str,
                              output_file: Path) -> None:
        """
        Generar plot combinado de EEMD y predicciones.
        
        Args:
            eemd_result: Resultado de la descomposición EEMD
            prediction_result: Resultado de la predicción
            station_name: Nombre de la estación
            output_file: Archivo de salida
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Configurar estilo
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Crear figura con subplots
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            fig.suptitle(f'Comprehensive Analysis - {station_name}', fontsize=16, fontweight='bold')
            
            # Plot 1: Serie original vs reconstruida
            if hasattr(eemd_result, 'original_series') and eemd_result.original_series is not None:
                original_series = eemd_result.original_series
                reconstructed_series = np.sum(eemd_result.imfs, axis=1)
                
                axes[0].plot(original_series.index, original_series.values, 
                           label='Original', alpha=0.7, linewidth=1)
                axes[0].plot(original_series.index, reconstructed_series, 
                           label='Reconstructed', alpha=0.7, linewidth=1)
                axes[0].set_title('Original vs Reconstructed Series')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Plot 2: IMFs principales
            if eemd_result.imfs is not None and eemd_result.imfs.shape[1] > 0:
                num_imfs = min(5, eemd_result.imfs.shape[1])  # Mostrar máximo 5 IMFs
                for i in range(num_imfs):
                    axes[1].plot(eemd_result.imfs[:, i], 
                               label=f'IMF {i+1}', alpha=0.8, linewidth=1)
                axes[1].set_title('Main IMFs')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Predicciones
            if hasattr(prediction_result, 'final_prediction') and prediction_result.final_prediction is not None:
                prediction_series = prediction_result.final_prediction
                if hasattr(prediction_result, 'future_dates') and prediction_result.future_dates is not None:
                    future_dates = prediction_result.future_dates
                    axes[2].plot(future_dates, prediction_series, 
                               label='Predictions', color='red', linewidth=2)
                    axes[2].set_title('Future Predictions')
                    axes[2].legend()
                    axes[2].grid(True, alpha=0.3)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Guardar plot
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Combined plot saved: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate combined plot: {e}")
            raise
    
    def generate_summary_report(self,
                              station_name: str,
                              eemd_result: EEMDResult,
                              prediction_result: PredictionResult,
                              output_dir: Path) -> str:
        """
        Generar reporte resumido de análisis.
        
        Args:
            station_name: Nombre de la estación
            eemd_result: Resultado de la descomposición EEMD
            prediction_result: Resultado de la predicción
            output_dir: Directorio de salida
            
        Returns:
            Ruta del archivo de reporte generado
        """
        try:
            self.logger.info(f"Generating summary report for station: {station_name}")
            
            # Crear directorio para reportes
            reports_dir = output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = reports_dir / f"{station_name}_summary_report.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"SUMMARY REPORT - {station_name}\n")
                f.write("=" * 50 + "\n\n")
                
                # Información EEMD
                f.write("EEMD ANALYSIS:\n")
                f.write("-" * 20 + "\n")
                if hasattr(eemd_result, 'imfs') and eemd_result.imfs is not None:
                    f.write(f"Number of IMFs: {eemd_result.imfs.shape[1]}\n")
                if hasattr(eemd_result, 'decomposition_quality'):
                    quality = eemd_result.decomposition_quality
                    f.write(f"Orthogonality score: {quality.get('orthogonality_score', 'N/A'):.4f}\n")
                    f.write(f"Reconstruction error: {quality.get('reconstruction_error', 'N/A'):.4f}\n")
                f.write("\n")
                
                # Información de predicciones
                f.write("PREDICTION ANALYSIS:\n")
                f.write("-" * 20 + "\n")
                if hasattr(prediction_result, 'final_prediction') and prediction_result.final_prediction is not None:
                    f.write(f"Prediction steps: {len(prediction_result.final_prediction)}\n")
                if hasattr(prediction_result, 'processing_time'):
                    f.write(f"Processing time: {prediction_result.processing_time:.2f}s\n")
                f.write("\n")
                
                # Métricas de calidad
                f.write("QUALITY METRICS:\n")
                f.write("-" * 20 + "\n")
                if hasattr(prediction_result, 'quality_metrics'):
                    metrics = prediction_result.quality_metrics
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.4f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
            
            self.logger.info(f"Summary report saved: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """
        Obtener formatos soportados para guardado.
        
        Returns:
            Lista de formatos soportados
        """
        return ['png', 'jpg', 'pdf', 'svg', 'tiff']
    
    def cleanup_old_plots(self, output_dir: Path, days_old: int = 30) -> int:
        """
        Limpiar plots antiguos.
        
        Args:
            output_dir: Directorio de salida
            days_old: Días de antigüedad para eliminar
            
        Returns:
            Número de archivos eliminados
        """
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = 0
            
            for plot_file in output_dir.rglob("*.png"):
                if plot_file.stat().st_mtime < cutoff_date.timestamp():
                    plot_file.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old plot files")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old plots: {e}")
            return 0 