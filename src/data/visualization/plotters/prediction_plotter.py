"""
Prediction Plotter

Este módulo provee herramientas para visualizar resultados de predicciones meteorológicas.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib.collections import LineCollection

# FASE 3: Import for alert visualization
try:
    from src.data.prediction.services.alert_detector import TemperatureAlert
except ImportError:
    TemperatureAlert = None  # Type hint fallback
from ....core.interfaces.visualization_strategy import VisualizationStrategyInterface


class PredictionPlotter(VisualizationStrategyInterface):
    """
    Plotter especializado en visualización de predicciones meteorológicas.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.setup_style()
    
    def setup_style(self) -> None:
        """Configure matplotlib style for prediction plots."""
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_visualization(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """
        Create prediction visualization.
        
        Args:
            data: DataFrame containing prediction data
            **kwargs: Additional parameters
            
        Returns:
            Matplotlib Figure object
        """
        plot_type = kwargs.get('plot_type', 'time_series')
        
        if plot_type == 'time_series':
            return self._create_time_series_plot(data, **kwargs)
        elif plot_type == 'comparison':
            return self._create_comparison_plot(data, **kwargs)
        elif plot_type == 'confidence_intervals':
            return self._create_confidence_intervals_plot(data, **kwargs)
        elif plot_type == 'seasonal_decomposition':
            return self._create_seasonal_decomposition_plot(data, **kwargs)
        elif plot_type == 'comprehensive':
            return self._create_comprehensive_plot(data, **kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def save_visualization(self, figure: plt.Figure, filepath: str, **kwargs) -> None:
        """
        Save the visualization to a file.
        
        Args:
            figure: Matplotlib Figure object to save
            filepath: Path where to save the visualization
            **kwargs: Additional saving parameters
        """
        dpi = kwargs.get('dpi', 300)
        bbox_inches = kwargs.get('bbox_inches', 'tight')
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        figure.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(figure)
    
    def get_visualization_info(self) -> Dict[str, Any]:
        """
        Get information about the visualization.
        
        Returns:
            Dictionary containing visualization metadata
        """
        return {
            'plotter_type': 'PredictionPlotter',
            'available_plots': ['time_series', 'comparison', 'confidence_intervals', 'seasonal_decomposition', 'comprehensive'],
            'config': self.config
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data is suitable for prediction plotting.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid for plotting, False otherwise
        """
        from src.core.validators.data_validator import DataValidator
        validator = DataValidator()
        
        try:
            # Check required columns
            required_columns = ['date', 'value', 'type']
            validator.validate_dataframe_structure(data, required_columns)
            
            # Check data types
            column_types = {
                'date': 'datetime64[ns]',
                'value': 'float64',
                'type': 'object'
            }
            validator.validate_data_types(data, column_types)
            
            return True
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported output formats for saving.
        
        Returns:
            List of supported file formats
        """
        return ['png', 'jpg', 'pdf', 'svg', 'tiff']
    
    def set_style(self, style_name: str) -> None:
        """
        Set the visualization style.
        
        Args:
            style_name: Name of the style to apply
        """
        if style_name == 'default':
            plt.style.use('default')
            sns.set_palette("husl")
        elif style_name == 'seaborn':
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        elif style_name == 'classic':
            plt.style.use('classic')
        elif style_name == 'bmh':
            plt.style.use('bmh')
        else:
            # Default to default style
            plt.style.use('default')
            sns.set_palette("husl")
    
    def _create_time_series_plot(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """
        Create time series prediction plot with PERFECT continuity - absolutely NO gaps.
        
        PROFESSIONAL SOLUTION: Plots everything as ONE continuous line to ensure
        matplotlib connects all points seamlessly without any visual gaps.
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Separate and sort data (include bridge points for continuity)
        historical_data = data[data['type'] == 'historical'].copy().sort_values('date')
        bridge_data = data[data['type'] == 'bridge'].copy().sort_values('date') if 'bridge' in data['type'].values else pd.DataFrame()
        prediction_data = data[data['type'] == 'prediction'].copy().sort_values('date')
        
        # CRITICAL: Combine bridge with predictions for continuity
        if not bridge_data.empty:
            # Bridge point connects historical to predictions
            prediction_data = pd.concat([bridge_data, prediction_data]).sort_values('date').reset_index(drop=True)
        
        # CRITICAL: Create ONE continuous line with NO gaps
        if not historical_data.empty and not prediction_data.empty:
            # Verify dates are consecutive (should be exactly 1 day apart)
            last_historical = historical_data.iloc[-1]
            first_prediction = prediction_data.iloc[0]
            date_gap = (first_prediction['date'] - last_historical['date']).days
            
            if date_gap == 1:
                # PROFESSIONAL SOLUTION: Plot as ONE continuous line
                # Combine all data into single continuous DataFrame
                complete_line = pd.concat([
                    historical_data,
                    prediction_data
                ]).sort_values('date').reset_index(drop=True)
                
                # Convert to datetime for proper plotting
                dates = pd.to_datetime(complete_line['date'])
                values = complete_line['value'].values
                
                # Find transition point index
                transition_idx = len(historical_data)
                
                # SOLUTION: Plot entire line as ONE continuous line first (ensures NO gap)
                # This creates the base continuous line that matplotlib will render without gaps
                ax.plot(dates, values, 
                       color='gray', linewidth=2.5, alpha=0.4, label='_nolegend_', zorder=1)
                
                # Now overlay with colored segments for visual distinction
                # Historical segment (blue) - all historical points
                hist_dates = dates.iloc[:transition_idx]
                hist_values = values[:transition_idx]
                if len(hist_dates) > 0:
                    ax.plot(hist_dates, hist_values, 
                           color='blue', linewidth=2, label='Datos Históricos', alpha=0.8, zorder=3)
                
                # Prediction segment (red) - CRITICAL: starts from last historical point
                # This ensures perfect continuity because it includes the shared transition point
                pred_dates = dates.iloc[transition_idx - 1:]  # Include last historical
                pred_values = values[transition_idx - 1:]      # Include last historical value
                if len(pred_dates) > 0:
                    ax.plot(pred_dates, pred_values, 
                           color='red', linewidth=2, label='Predicciones', alpha=0.8, zorder=2)
                
            else:
                # Fallback: plot separately if dates are not consecutive (shouldn't happen)
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Date gap detected: {date_gap} days. Plotting separately.")
                if not historical_data.empty:
                    ax.plot(historical_data['date'], historical_data['value'], 
                           color='blue', linewidth=2, label='Datos Históricos', alpha=0.8)
                if not prediction_data.empty:
                    ax.plot(prediction_data['date'], prediction_data['value'], 
                           color='red', linewidth=2, label='Predicciones', alpha=0.8)
        else:
            # Plot separately if one is missing
            if not historical_data.empty:
                ax.plot(historical_data['date'], historical_data['value'], 
                       color='blue', linewidth=2, label='Datos Históricos', alpha=0.8)
            if not prediction_data.empty:
                ax.plot(prediction_data['date'], prediction_data['value'], 
                       color='red', linewidth=2, label='Predicciones', alpha=0.8)
        
        # Add confidence intervals if available
        if not prediction_data.empty and 'confidence_lower' in prediction_data.columns and 'confidence_upper' in prediction_data.columns:
            ax.fill_between(prediction_data['date'], 
                          prediction_data['confidence_lower'], 
                          prediction_data['confidence_upper'], 
                          color='red', alpha=0.2, label='Intervalo de Confianza', zorder=0)
        
        # FASE 3: Add alert markers if available
        alerts = kwargs.get('alerts', [])
        if alerts:
            self._plot_alerts(ax, alerts, prediction_data)
        
        # Customize plot - ESPAÑOL + Nombre de estación
        station_name = kwargs.get('station_name', '')
        title = 'Predicciones Meteorológicas'
        if station_name:
            # Limpiar nombre de estación (remover sufijos como "_Imputed")
            clean_station_name = station_name.replace('_Imputed', '').replace('_', ' ')
            title += f' - {clean_station_name}'
        
        if alerts:
            critical_count = sum(1 for a in alerts if a.severity.value == 'critical')
            warning_count = sum(1 for a in alerts if a.severity.value == 'warning')
            if critical_count > 0:
                title += f' - ⚠️ {critical_count} Alerta{"s" if critical_count > 1 else ""} Crítica{"s" if critical_count > 1 else ""}'
            elif warning_count > 0:
                title += f' - ⚠️ {warning_count} Advertencia{"s" if warning_count > 1 else ""}'
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Temperatura (°C)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def _create_comparison_plot(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create comparison plot between different prediction methods."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Separate data by type
        historical_data = data[data['type'] == 'historical']
        prediction_data = data[data['type'] == 'prediction']
        
        # Top plot: Time series comparison
        if not historical_data.empty:
            ax1.plot(historical_data['date'], historical_data['value'], 
                    color='blue', linewidth=2, label='Histórico', alpha=0.8)
        
        if not prediction_data.empty:
            ax1.plot(prediction_data['date'], prediction_data['value'], 
                    color='red', linewidth=2, label='Predicho', alpha=0.8)
        
        station_name = kwargs.get('station_name', '')
        title_comp = 'Valores Históricos vs Predichos'
        if station_name:
            clean_station_name = station_name.replace('_Imputed', '').replace('_', ' ')
            title_comp += f' - {clean_station_name}'
        ax1.set_title(title_comp, fontweight='bold')
        ax1.set_ylabel('Temperatura (°C)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Residuals/Errors or Information
        if not historical_data.empty and not prediction_data.empty:
            # Calculate residuals for overlapping period
            min_pred_date = prediction_data['date'].min()
            max_hist_date = historical_data['date'].max()
            
            if min_pred_date <= max_hist_date:
                # There is overlap - calculate residuals
                overlap_hist = historical_data[historical_data['date'] >= min_pred_date]
                overlap_pred = prediction_data[prediction_data['date'] <= max_hist_date]
                
                if len(overlap_hist) == len(overlap_pred):
                    residuals = overlap_hist['value'].values - overlap_pred['value'].values
                    ax2.plot(overlap_hist['date'], residuals, color='green', linewidth=1)
                    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    station_name = kwargs.get('station_name', '')
                    title_resid = 'Residuos de Predicción'
                    if station_name:
                        clean_station_name = station_name.replace('_Imputed', '').replace('_', ' ')
                        title_resid += f' - {clean_station_name}'
                    ax2.set_title(title_resid, fontweight='bold')
                    ax2.set_ylabel('Residuo (°C)')
                    ax2.grid(True, alpha=0.3)
                else:
                    # Length mismatch - show info
                    ax2.text(0.5, 0.5, 'No se pueden calcular residuos\n(Desajuste de longitud en período de solapamiento)', 
                            ha='center', va='center', transform=ax2.transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
                    ax2.set_title('Residuos de Predicción - Sin Datos', fontweight='bold')
            else:
                # No temporal overlap - show informative message
                ax2.text(0.5, 0.5, 
                        f'Sin solapamiento temporal para cálculo de residuos\n\n'
                        f'Datos históricos terminan: {max_hist_date.strftime("%Y-%m-%d")}\n'
                        f'Predicciones inician: {min_pred_date.strftime("%Y-%m-%d")}\n\n'
                        f'Los residuos solo se pueden calcular cuando\n'
                        f'los datos históricos y predichos se solapan en el tiempo.',
                        ha='center', va='center', transform=ax2.transAxes, fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                ax2.set_title('Residuos de Predicción - Sin Solapamiento', fontweight='bold')
                ax2.set_ylabel('Información')
        else:
            # Missing data - show info
            missing_info = []
            if historical_data.empty:
                missing_info.append("Datos históricos")
            if prediction_data.empty:
                missing_info.append("Datos de predicción")
            
            ax2.text(0.5, 0.5, f'Datos faltantes para cálculo de residuos:\n{", ".join(missing_info)}', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            ax2.set_title('Residuos de Predicción - Datos Faltantes', fontweight='bold')
        
        ax2.set_xlabel('Fecha')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def _create_confidence_intervals_plot(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create plot with confidence intervals."""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Separate data
        historical_data = data[data['type'] == 'historical']
        prediction_data = data[data['type'] == 'prediction']
        
        # Plot historical data
        if not historical_data.empty:
            ax.plot(historical_data['date'], historical_data['value'], 
                   color='blue', linewidth=2, label='Datos Históricos', alpha=0.8)
        
        # Plot predictions with confidence intervals
        if not prediction_data.empty:
            # Main prediction line
            ax.plot(prediction_data['date'], prediction_data['value'], 
                   color='red', linewidth=2, label='Predicciones', alpha=0.8)
            
            # Confidence intervals
            if 'confidence_lower' in prediction_data.columns and 'confidence_upper' in prediction_data.columns:
                ax.fill_between(prediction_data['date'], 
                              prediction_data['confidence_lower'], 
                              prediction_data['confidence_upper'], 
                              color='red', alpha=0.2, label='Intervalo de Confianza 95%')
            
            # Multiple confidence levels if available
            confidence_levels = [col for col in prediction_data.columns if 'confidence_' in col and col != 'confidence_lower' and col != 'confidence_upper']
            colors = ['orange', 'yellow', 'lightblue']
            
            for i, level in enumerate(confidence_levels[:3]):  # Limit to 3 levels
                if level in prediction_data.columns:
                    level_data = prediction_data[level]
                    ax.plot(prediction_data['date'], level_data, 
                           color=colors[i], linewidth=1, alpha=0.6, 
                           label=f'{level.replace("confidence_", "")}')
        
        station_name = kwargs.get('station_name', '')
        title_conf = 'Predicciones con Intervalos de Confianza'
        if station_name:
            clean_station_name = station_name.replace('_Imputed', '').replace('_', ' ')
            title_conf += f' - {clean_station_name}'
        ax.set_title(title_conf, fontsize=16, fontweight='bold')
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Temperatura (°C)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def _create_seasonal_decomposition_plot(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create seasonal decomposition plot."""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Get prediction data
        prediction_data = data[data['type'] == 'prediction']
        
        if not prediction_data.empty:
            # Extract IMF components if available
            imf_cols = [col for col in prediction_data.columns if col.startswith('imf_')]
            
            if len(imf_cols) >= 3:
                # Plot first 3 IMFs
                for i, imf_col in enumerate(imf_cols[:3]):
                    axes[i].plot(prediction_data['date'], prediction_data[imf_col], 
                               linewidth=1, color=f'C{i}')
                    axes[i].set_title(f'Componente {imf_col.upper()}', fontweight='bold')
                    axes[i].set_ylabel('Amplitud')
                    axes[i].grid(True, alpha=0.3)
                
                # Plot reconstructed signal
                if 'reconstructed' in prediction_data.columns:
                    axes[3].plot(prediction_data['date'], prediction_data['reconstructed'], 
                               linewidth=2, color='green', label='Reconstructed')
                    axes[3].plot(prediction_data['date'], prediction_data['value'], 
                               linewidth=1, color='red', alpha=0.7, label='Original')
                    axes[3].set_title('Señal Reconstruida', fontweight='bold')
                    axes[3].set_ylabel('Temperatura (°C)')
                    axes[3].legend()
                    axes[3].grid(True, alpha=0.3)
            else:
                # Fallback: plot original data
                axes[0].plot(prediction_data['date'], prediction_data['value'], 
                           linewidth=2, color='blue')
                axes[0].set_title('Datos de Predicción', fontweight='bold')
                axes[0].set_ylabel('Temperatura (°C)')
                axes[0].grid(True, alpha=0.3)
                
                # Hide other subplots
                for i in range(1, 4):
                    axes[i].set_visible(False)
        
        # Set x-axis labels
        for ax in axes:
            if ax.get_visible():
                ax.set_xlabel('Fecha')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def _create_comprehensive_plot(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create comprehensive prediction analysis plot."""
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main time series plot (top row, full width)
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_main_time_series(ax_main, data)
        
        # 2. Historical vs Prediction comparison (second row, left)
        ax_comp = fig.add_subplot(gs[1, 0])
        self._plot_comparison(ax_comp, data)
        
        # 3. Confidence intervals (second row, middle)
        ax_conf = fig.add_subplot(gs[1, 1])
        self._plot_confidence_intervals(ax_conf, data)
        
        # 4. Residuals (second row, right)
        ax_resid = fig.add_subplot(gs[1, 2])
        self._plot_residuals(ax_resid, data)
        
        # 5. IMF components (bottom two rows)
        prediction_data = data[data['type'] == 'prediction']
        imf_cols = [col for col in prediction_data.columns if col.startswith('imf_')]
        
        for i, imf_col in enumerate(imf_cols[:6]):  # Show up to 6 IMFs
            row = 2 + (i // 3)
            col = i % 3
            ax_imf = fig.add_subplot(gs[row, col])
            self._plot_imf_component(ax_imf, prediction_data, imf_col)
        
        station_name = kwargs.get('station_name', '')
        suptitle = 'Análisis Comprehensivo de Predicciones'
        if station_name:
            clean_station_name = station_name.replace('_Imputed', '').replace('_', ' ')
            suptitle += f' - {clean_station_name}'
        plt.suptitle(suptitle, fontsize=16, fontweight='bold')
        return fig
    
    def _plot_main_time_series(self, ax, data: pd.DataFrame) -> None:
        """Plot main time series."""
        historical_data = data[data['type'] == 'historical']
        prediction_data = data[data['type'] == 'prediction']
        
        if not historical_data.empty:
            ax.plot(historical_data['date'], historical_data['value'], 
                   color='blue', linewidth=2, label='Histórico', alpha=0.8)
        
        if not prediction_data.empty:
            ax.plot(prediction_data['date'], prediction_data['value'], 
                   color='red', linewidth=2, label='Predicciones', alpha=0.8)
        
        station_name = kwargs.get('station_name', '')
        title_main = 'Serie Temporal Meteorológica'
        if station_name:
            clean_station_name = station_name.replace('_Imputed', '').replace('_', ' ')
            title_main += f' - {clean_station_name}'
        ax.set_title(title_main, fontweight='bold')
        ax.set_ylabel('Temperatura (°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_comparison(self, ax, data: pd.DataFrame) -> None:
        """Plot comparison between historical and predicted values."""
        historical_data = data[data['type'] == 'historical']
        prediction_data = data[data['type'] == 'prediction']
        
        if not historical_data.empty and not prediction_data.empty:
            # Calculate comparison for overlapping period
            min_pred_date = prediction_data['date'].min()
            max_hist_date = historical_data['date'].max()
            
            if min_pred_date <= max_hist_date:
                overlap_hist = historical_data[historical_data['date'] >= min_pred_date]
                overlap_pred = prediction_data[prediction_data['date'] <= max_hist_date]
                
                if len(overlap_hist) == len(overlap_pred):
                    ax.scatter(overlap_hist['value'], overlap_pred['value'], alpha=0.6)
                    ax.plot([overlap_hist['value'].min(), overlap_hist['value'].max()], 
                           [overlap_hist['value'].min(), overlap_hist['value'].max()], 
                           'r--', alpha=0.8)
                else:
                    # Length mismatch - show info
                    ax.text(0.5, 0.5, 'No overlap data\nfor comparison', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            else:
                # No temporal overlap - show info
                ax.text(0.5, 0.5, 'No temporal\noverlap', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            # Missing data - show info
            ax.text(0.5, 0.5, 'Missing data\nfor comparison', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        ax.set_title('Histórico vs Predicho', fontweight='bold')
        ax.set_xlabel('Histórico (°C)')
        ax.set_ylabel('Predicho (°C)')
        ax.grid(True, alpha=0.3)
    
    def _plot_confidence_intervals(self, ax, data: pd.DataFrame) -> None:
        """Plot confidence intervals."""
        prediction_data = data[data['type'] == 'prediction']
        
        if not prediction_data.empty and 'confidence_lower' in prediction_data.columns:
            ax.fill_between(prediction_data['date'], 
                          prediction_data['confidence_lower'], 
                          prediction_data['confidence_upper'], 
                          alpha=0.3, color='red')
            ax.plot(prediction_data['date'], prediction_data['value'], 
                   color='red', linewidth=1)
        
        ax.set_title('Intervalos de Confianza', fontweight='bold')
        ax.set_ylabel('Temperatura (°C)')
        ax.grid(True, alpha=0.3)
    
    def _plot_residuals(self, ax, data: pd.DataFrame) -> None:
        """Plot residuals."""
        historical_data = data[data['type'] == 'historical']
        prediction_data = data[data['type'] == 'prediction']
        
        if not historical_data.empty and not prediction_data.empty:
            # Calculate residuals for overlapping period
            min_pred_date = prediction_data['date'].min()
            max_hist_date = historical_data['date'].max()
            
            if min_pred_date <= max_hist_date:
                overlap_hist = historical_data[historical_data['date'] >= min_pred_date]
                overlap_pred = prediction_data[prediction_data['date'] <= max_hist_date]
                
                if len(overlap_hist) == len(overlap_pred):
                    residuals = overlap_hist['value'].values - overlap_pred['value'].values
                    ax.plot(overlap_hist['date'], residuals, color='green', linewidth=1)
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                else:
                    # Length mismatch - show info
                    ax.text(0.5, 0.5, 'No overlap data\nfor residuals', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            else:
                # No temporal overlap - show info
                ax.text(0.5, 0.5, 'No temporal\noverlap', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            # Missing data - show info
            ax.text(0.5, 0.5, 'Datos faltantes\npara residuos', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        ax.set_title('Residuos', fontweight='bold')
        ax.set_ylabel('Residuo (°C)')
        ax.grid(True, alpha=0.3)
    
    def _plot_imf_component(self, ax, data: pd.DataFrame, imf_col: str) -> None:
        """Plot IMF component."""
        if imf_col in data.columns:
            ax.plot(data['date'], data[imf_col], linewidth=1, color='purple')
            ax.set_title(f'{imf_col.upper()}', fontweight='bold', fontsize=10)
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
    
    def _plot_alerts(self, ax, alerts: List, prediction_data: pd.DataFrame) -> None:
        """
        Plot alert markers on the time series plot (FASE 3).
        
        Args:
            ax: Matplotlib axes
            alerts: List of TemperatureAlert objects
            prediction_data: DataFrame with prediction data
        """
        if not alerts or prediction_data.empty:
            return
        
        # Group alerts by type for different markers
        critical_cold = [a for a in alerts if a.severity.value == 'critical' and 'cold' in a.alert_type.value]
        warning_cold = [a for a in alerts if a.severity.value == 'warning' and 'cold' in a.alert_type.value]
        critical_heat = [a for a in alerts if a.severity.value == 'critical' and 'heat' in a.alert_type.value]
        warning_heat = [a for a in alerts if a.severity.value == 'warning' and 'heat' in a.alert_type.value]
        
        # Plot critical cold wave alerts (blue, large marker, down arrow)
        if critical_cold:
            for alert in critical_cold:
                ax.scatter([alert.date], [alert.predicted_value], 
                         color='blue', marker='v', s=200, alpha=0.8, 
                         edgecolors='darkblue', linewidths=2, zorder=5,
                         label='Alerta Crítica de Frío' if alert == critical_cold[0] else '')
        
        # Plot warning cold wave alerts (light blue, medium marker)
        if warning_cold:
            for alert in warning_cold:
                ax.scatter([alert.date], [alert.predicted_value], 
                         color='lightblue', marker='v', s=150, alpha=0.7, 
                         edgecolors='blue', linewidths=1.5, zorder=4,
                         label='Alerta de Ola de Frío' if alert == warning_cold[0] else '')
        
        # Plot critical heat wave alerts (red, large marker, up arrow)
        if critical_heat:
            for alert in critical_heat:
                ax.scatter([alert.date], [alert.predicted_value], 
                         color='red', marker='^', s=200, alpha=0.8, 
                         edgecolors='darkred', linewidths=2, zorder=5,
                         label='Alerta Crítica de Calor' if alert == critical_heat[0] else '')
        
        # Plot warning heat wave alerts (orange, medium marker)
        if warning_heat:
            for alert in warning_heat:
                ax.scatter([alert.date], [alert.predicted_value], 
                         color='orange', marker='^', s=150, alpha=0.7, 
                         edgecolors='red', linewidths=1.5, zorder=4,
                         label='Alerta de Ola de Calor' if alert == warning_heat[0] else '')
        
        # Add threshold lines for context (only if we have prediction data)
        # CRITICAL: Only show thresholds relevant to the variable type being processed
        if not prediction_data.empty and alerts:
            # Get variable type from first alert (all alerts should have same variable_type)
            variable_type = alerts[0].variable_type if alerts else None
            
            # Get unique thresholds from alerts, but only for the correct variable type
            thresholds = {}
            for alert in alerts:
                # Only include thresholds for alerts matching the current variable type
                if alert.variable_type == variable_type:
                    key = (alert.alert_type.value, alert.variable_type)
                    if key not in thresholds:
                        thresholds[key] = alert.threshold_value
            
            # Plot threshold lines - only for the relevant type
            date_range = pd.to_datetime(prediction_data['date'])
            for (alert_type, var_type), threshold in thresholds.items():
                # For temp_min: Only show cold wave thresholds
                if variable_type == "temp_min" and 'cold' in alert_type:
                    ax.axhline(y=threshold, color='blue', linestyle='--', alpha=0.5, 
                              linewidth=1, label=f'Umbral de Frío (temp_min)' if len(thresholds) <= 2 else '')
                # For temp_max: Only show heat wave thresholds
                elif variable_type == "temp_max" and 'heat' in alert_type:
                    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, 
                              linewidth=1, label=f'Umbral de Calor (temp_max)' if len(thresholds) <= 2 else '')
    
    def plot_time_series(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> None:
        """Plot time series and optionally save."""
        fig = self._create_time_series_plot(data, **kwargs)
        if save_path:
            self.save_visualization(fig, save_path, **kwargs)
        else:
            plt.show()
    
    def plot_comparison(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> None:
        """Plot comparison and optionally save."""
        fig = self._create_comparison_plot(data, **kwargs)
        if save_path:
            self.save_visualization(fig, save_path, **kwargs)
        else:
            plt.show()
    
    def plot_confidence_intervals(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> None:
        """Plot confidence intervals and optionally save."""
        fig = self._create_confidence_intervals_plot(data, **kwargs)
        if save_path:
            self.save_visualization(fig, save_path, **kwargs)
        else:
            plt.show()
    
    def plot_seasonal_decomposition(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> None:
        """Plot seasonal decomposition and optionally save."""
        fig = self._create_seasonal_decomposition_plot(data, **kwargs)
        if save_path:
            self.save_visualization(fig, save_path, **kwargs)
        else:
            plt.show()
    
    def plot_comprehensive(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> None:
        """Plot comprehensive analysis and optionally save."""
        fig = self._create_comprehensive_plot(data, **kwargs)
        if save_path:
            self.save_visualization(fig, save_path, **kwargs)
        else:
            plt.show() 