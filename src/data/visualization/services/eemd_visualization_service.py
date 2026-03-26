"""
EEMD Visualization Service

Este módulo provee un servicio especializado para generar visualizaciones
de resultados de EEMD (Ensemble Empirical Mode Decomposition).
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path
import numpy as np

from ..plotters.eemd_plotter import EEMDPlotter
from src.core.interfaces.prediction_strategy import EEMDResult

logger = logging.getLogger(__name__)


class EEMDVisualizationService:
    """
    Servicio especializado para visualización de resultados EEMD.
    
    Este servicio maneja la generación de plots para:
    - IMFs (Intrinsic Mode Functions)
    - Correlaciones con la serie original
    - Varianza explicada por cada IMF
    - Métricas de calidad de la descomposición
    - Análisis comprehensivo
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar el servicio de visualización EEMD.
        
        Args:
            config: Configuración opcional para el plotter
        """
        self.config = config or {}
        self.plotter = EEMDPlotter(self.config)
        self.logger = logger
        
        self.logger.info("EEMDVisualizationService initialized")
    
    def generate_eemd_plots(self, 
                           eemd_result: EEMDResult,
                           station_name: str,
                           output_dir: Path,
                           original_series: Optional[pd.Series] = None) -> Dict[str, str]:
        """
        Generar todos los plots de EEMD para una estación.
        
        Args:
            eemd_result: Resultado de la descomposición EEMD
            station_name: Nombre de la estación
            output_dir: Directorio de salida para los plots
            original_series: Serie temporal original (opcional)
            
        Returns:
            Diccionario con las rutas de los archivos generados
        """
        try:
            self.logger.info(f"Generating EEMD plots for station: {station_name}")
            
            # Crear directorio de salida
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Preparar datos para los plots
            plot_data = self._prepare_plot_data(eemd_result)
            
            # Generar plots
            generated_files = {}
            
            # 1. Plot de IMFs
            imfs_file = output_dir / f"{station_name}_imfs_plot.png"
            self.plotter.plot_imfs(plot_data, str(imfs_file), dpi=300)
            generated_files['imfs_plot'] = str(imfs_file)
            
            # 2. Plot de correlaciones
            correlations_file = output_dir / f"{station_name}_correlations_plot.png"
            self.plotter.plot_correlations(
                eemd_result.correlations, 
                str(correlations_file), 
                dpi=300
            )
            generated_files['correlations_plot'] = str(correlations_file)
            
            # 3. Plot de varianza explicada
            variance_file = output_dir / f"{station_name}_variance_plot.png"
            
            # Safe handling of variance_explained
            variance_values = []
            try:
                if isinstance(eemd_result.variance_explained, pd.DataFrame) and 'explained_ratio' in eemd_result.variance_explained.columns:
                    variance_values = eemd_result.variance_explained['explained_ratio'].tolist()
                else:
                    self.logger.info(f"Variance explained data not available for {station_name}. Using empty variance plot.")
            except Exception as e:
                self.logger.info(f"Error processing variance explained for {station_name}: {e}. Using empty variance plot.")
            
            self.plotter.plot_variance(variance_values, str(variance_file), dpi=300)
            generated_files['variance_plot'] = str(variance_file)
            
            # 4. Plot de métricas de calidad
            quality_file = output_dir / f"{station_name}_quality_plot.png"
            quality_metrics = {
                'orthogonality_score': eemd_result.orthogonality_score,
                'reconstruction_error': eemd_result.decomposition_quality.get('reconstruction_error', 0),
                'mean_imf_quality': eemd_result.decomposition_quality.get('mean_imf_quality', 0),
                'seasonality_strength': eemd_result.decomposition_quality.get('seasonality_strength', 0)
            }
            self.plotter.plot_quality(quality_metrics, str(quality_file), dpi=300)
            generated_files['quality_plot'] = str(quality_file)
            
            # 5. Plot comprehensivo
            comprehensive_file = output_dir / f"{station_name}_comprehensive_plot.png"
            self.plotter.plot_comprehensive(
                plot_data, 
                str(comprehensive_file), 
                dpi=300,
                original_data=original_series,
                correlations=eemd_result.correlations,
                variance_explained=variance_values,
                quality_metrics=quality_metrics
            )
            generated_files['comprehensive_plot'] = str(comprehensive_file)
            
            # 6. Plot de reconstrucción (si hay datos originales)
            if original_series is not None:
                reconstruction_file = output_dir / f"{station_name}_reconstruction_plot.png"
                self._plot_reconstruction(
                    original_series, 
                    eemd_result, 
                    str(reconstruction_file)
                )
                generated_files['reconstruction_plot'] = str(reconstruction_file)
            
            self.logger.info(f"Generated {len(generated_files)} EEMD plots for {station_name}")
            return generated_files
            
        except Exception as e:
            self.logger.error(f"Error generating EEMD plots for {station_name}: {e}")
            raise
    
    def _prepare_plot_data(self, eemd_result: EEMDResult) -> pd.DataFrame:
        """
        Preparar datos para los plots.
        
        Args:
            eemd_result: Resultado de la descomposición EEMD
            
        Returns:
            DataFrame con los datos preparados para plotting
        """
        # Crear DataFrame con los IMFs
        imfs_data = {}
        for i in range(eemd_result.imfs.shape[1]):
            imfs_data[f'imf_{i+1}'] = eemd_result.imfs[:, i]
        
        return pd.DataFrame(imfs_data)
    
    def _plot_reconstruction(self, 
                           original_series: pd.Series, 
                           eemd_result: EEMDResult, 
                           save_path: str) -> None:
        """
        Plot de reconstrucción de la serie original.
        
        Args:
            original_series: Serie temporal original
            eemd_result: Resultado de la descomposición EEMD
            save_path: Ruta donde guardar el plot
        """
        try:
            import matplotlib.pyplot as plt
            
            # Reconstruct series by summing all IMFs
            reconstructed_series = np.sum(eemd_result.imfs, axis=1)
            
            # Crear el plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Handle dimension mismatch due to downsampling
            original_length = len(original_series)
            reconstructed_length = len(reconstructed_series)
            
            if original_length != reconstructed_length:
                # This is expected when downsampling is applied during EEMD processing
                self.logger.info(f"Downsampling detected: original={original_length}, reconstructed={reconstructed_length}")
                
                # Use the shorter length for plotting
                plot_length = min(original_length, reconstructed_length)
                
                # Truncate both series to the same length
                original_plot = original_series.iloc[:plot_length]
                reconstructed_plot = reconstructed_series[:plot_length]
                
                # Create x-axis values
                if isinstance(original_series.index, pd.DatetimeIndex):
                    x_values = original_series.index[:plot_length]
                else:
                    x_values = range(plot_length)
                
                self.logger.info(f"Using truncated data for reconstruction plot: {plot_length} points")
            else:
                # No dimension mismatch, use original data
                original_plot = original_series
                reconstructed_plot = reconstructed_series
                
                if isinstance(original_series.index, pd.DatetimeIndex):
                    x_values = original_series.index
                else:
                    x_values = range(len(original_series))
            
            # Plot de la serie original vs reconstruida
            ax1.plot(x_values, original_plot.values, 
                    label='Original', linewidth=1, color='black')
            ax1.plot(x_values, reconstructed_plot, 
                    label='Reconstructed', linewidth=1, color='red', alpha=0.7)
            ax1.set_title('Original vs Reconstructed Series', fontweight='bold')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot de la diferencia
            difference = original_plot.values - reconstructed_plot
            ax2.plot(x_values, difference, 
                    linewidth=1, color='blue', alpha=0.7)
            ax2.set_title('Reconstruction Error', fontweight='bold')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Error')
            ax2.grid(True, alpha=0.3)
            
            # Agregar estadísticas
            mse = np.mean(difference**2)
            mae = np.mean(np.abs(difference))
            ax2.text(0.02, 0.98, f'MSE: {mse:.6f}\nMAE: {mae:.6f}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Reconstruction plot saved: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating reconstruction plot: {e}")
            raise
    
    def generate_summary_report(self, 
                               eemd_result: EEMDResult,
                               station_name: str,
                               output_dir: Path) -> str:
        """
        Generar un reporte de resumen de la descomposición EEMD.
        
        Args:
            eemd_result: Resultado de la descomposición EEMD
            station_name: Nombre de la estación
            output_dir: Directorio de salida
            
        Returns:
            Ruta del archivo de reporte generado
        """
        try:
            report_file = output_dir / f"{station_name}_eemd_summary.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"EEMD DECOMPOSITION SUMMARY - {station_name}\n")
                f.write("=" * 80 + "\n\n")
                
                # Información básica
                f.write("BASIC INFORMATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Station: {station_name}\n")
                f.write(f"Number of IMFs: {eemd_result.num_imfs}\n")
                f.write(f"Best sd_thresh: {eemd_result.best_sd_thresh:.4f}\n")
                f.write(f"Series length: {eemd_result.imfs.shape[0]}\n\n")
                
                # Métricas de calidad
                f.write("QUALITY METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Orthogonality score: {eemd_result.orthogonality_score:.6f}\n")
                f.write(f"Reconstruction error: {eemd_result.decomposition_quality.get('reconstruction_error', 0):.6f}\n")
                f.write(f"Reconstruction correlation: {eemd_result.decomposition_quality.get('reconstruction_correlation', 0):.6f}\n")
                f.write(f"Mean IMF quality: {eemd_result.decomposition_quality.get('mean_imf_quality', 0):.6f}\n")
                f.write(f"Seasonality strength: {eemd_result.decomposition_quality.get('seasonality_strength', 0):.6f}\n\n")
                
                # Correlaciones
                f.write("IMF CORRELATIONS WITH ORIGINAL SERIES:\n")
                f.write("-" * 40 + "\n")
                if isinstance(eemd_result.correlations, list):
                    for i, corr in enumerate(eemd_result.correlations):
                        f.write(f"IMF {i+1}: {corr:.6f}\n")
                else:
                    f.write("Correlations data not available\n")
                f.write("\n")
                
                # Varianza explicada
                f.write("VARIANCE EXPLAINED BY IMFS:\n")
                f.write("-" * 40 + "\n")
                try:
                    if isinstance(eemd_result.variance_explained, pd.DataFrame) and 'explained_ratio' in eemd_result.variance_explained.columns:
                        for i, row in eemd_result.variance_explained.iterrows():
                            imf_idx = row['imf_index']
                            var_ratio = row['explained_ratio']
                            f.write(f"IMF {imf_idx}: {var_ratio:.6f} ({var_ratio*100:.2f}%)\n")
                    else:
                        f.write("Variance explained data not available\n")
                except Exception as e:
                    f.write(f"Error processing variance explained: {e}\n")
                f.write("\n")
                
                # Clasificación de IMFs
                try:
                    if 'imf_classifications' in eemd_result.decomposition_quality:
                        classifications = eemd_result.decomposition_quality['imf_classifications']
                        f.write("IMF CLASSIFICATIONS:\n")
                        f.write("-" * 40 + "\n")
                        for category, imfs in classifications.items():
                            if imfs and isinstance(imfs, list):
                                f.write(f"{category}: IMFs {[i+1 for i in imfs]}\n")
                        f.write("\n")
                except Exception as e:
                    f.write(f"Error processing IMF classifications: {e}\n")
                
                # Patrones meteorológicos
                try:
                    if 'meteorological_patterns' in eemd_result.decomposition_quality:
                        patterns = eemd_result.decomposition_quality['meteorological_patterns']
                        f.write("METEOROLOGICAL PATTERNS:\n")
                        f.write("-" * 40 + "\n")
                        for pattern, imf_idx in patterns.items():
                            if imf_idx is not None:
                                f.write(f"{pattern}: IMF {imf_idx + 1}\n")
                        f.write("\n")
                except Exception as e:
                    f.write(f"Error processing meteorological patterns: {e}\n")
                
                f.write("=" * 80 + "\n")
                f.write("End of EEMD Summary Report\n")
                f.write("=" * 80 + "\n")
            
            self.logger.info(f"EEMD summary report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"Error generating EEMD summary report: {e}")
            # Return a basic report even if there's an error
            try:
                basic_report = output_dir / f"{station_name}_eemd_summary_basic.txt"
                with open(basic_report, 'w', encoding='utf-8') as f:
                    f.write(f"EEMD SUMMARY - {station_name}\n")
                    f.write(f"Error generating full report: {e}\n")
                    f.write(f"Number of IMFs: {eemd_result.num_imfs}\n")
                    f.write(f"Best sd_thresh: {eemd_result.best_sd_thresh:.4f}\n")
                return str(basic_report)
            except:
                return "" 