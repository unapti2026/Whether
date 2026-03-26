"""
EEMD Plotter

Este módulo provee herramientas para visualizar resultados de EEMD (Ensemble Empirical Mode Decomposition).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from pathlib import Path
import seaborn as sns
from ....core.interfaces.visualization_strategy import VisualizationStrategyInterface

class EEMDPlotter(VisualizationStrategyInterface):
    """
    Plotter especializado en visualización de resultados EEMD.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.setup_style()
    
    def setup_style(self) -> None:
        """Configure matplotlib style for EEMD plots."""
        plt.style.use('default')
        sns.set_palette("husl")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data is suitable for EEMD plotting.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid for plotting, False otherwise
        """
        from src.core.validators.data_validator import DataValidator
        validator = DataValidator()
        
        try:
            # Check required columns
            required_columns = ['date', 'value']
            validator.validate_dataframe_structure(data, required_columns)
            
            # Check data types
            column_types = {
                'date': 'datetime64[ns]',
                'value': 'float64'
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
        
    def create_visualization(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """
        Create EEMD visualization.
        
        Args:
            data: DataFrame containing EEMD data
            **kwargs: Additional parameters
            
        Returns:
            Matplotlib Figure object
        """
        plot_type = kwargs.get('plot_type', 'imfs')
        
        if plot_type == 'imfs':
            return self._create_imfs_plot(data, **kwargs)
        elif plot_type == 'correlations':
            return self._create_correlations_plot(data, **kwargs)
        elif plot_type == 'variance':
            return self._create_variance_plot(data, **kwargs)
        elif plot_type == 'quality':
            return self._create_quality_plot(data, **kwargs)
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
            'plotter_type': 'EEMDPlotter',
            'available_plots': ['imfs', 'correlations', 'variance', 'quality', 'comprehensive'],
            'config': self.config
        }
    
    def _create_imfs_plot(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create IMFs visualization."""
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Get IMF columns
        imf_cols = [col for col in data.columns if col.startswith('imf_')]
        imf_cols.sort(key=lambda x: int(x.split('_')[1]))
        
        # Plot each IMF
        for i, imf_col in enumerate(imf_cols):
            if i < len(axes):
                ax = axes[i]
                ax.plot(data.index, data[imf_col], linewidth=0.8, color=f'C{i}')
                ax.set_title(f'{imf_col.upper()}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Time')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for i in range(len(imf_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('EEMD Intrinsic Mode Functions (IMFs)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _create_correlations_plot(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create correlations visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Get correlation data
        correlations = kwargs.get('correlations', [])
        imf_names = [f'IMF {i+1}' for i in range(len(correlations))]
        
        # Bar plot of correlations
        bars = ax1.bar(imf_names, correlations, color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.set_title('IMF Correlations with Original Series', fontweight='bold')
        ax1.set_xlabel('IMF')
        ax1.set_ylabel('Correlation Coefficient')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Heatmap of IMF correlations
        if len(correlations) > 1:
            # Create correlation matrix (simplified)
            corr_matrix = np.zeros((len(correlations), len(correlations)))
            np.fill_diagonal(corr_matrix, correlations)
            
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       xticklabels=imf_names, yticklabels=imf_names, ax=ax2)
            ax2.set_title('IMF Correlation Matrix', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _create_variance_plot(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create variance explained visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Get variance data
        variance_explained = kwargs.get('variance_explained', [])
        imf_names = [f'IMF {i+1}' for i in range(len(variance_explained))]
        
        # Bar plot of variance explained
        bars = ax1.bar(imf_names, variance_explained, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        ax1.set_title('Variance Explained by IMFs', fontweight='bold')
        ax1.set_xlabel('IMF')
        ax1.set_ylabel('Variance Explained')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for bar, var in zip(bars, variance_explained):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{var:.1%}', ha='center', va='bottom', fontsize=9)
        
        # Cumulative variance plot
        cumulative_var = np.cumsum(variance_explained)
        ax2.plot(imf_names, cumulative_var, 'o-', linewidth=2, markersize=8, color='orange')
        ax2.fill_between(imf_names, cumulative_var, alpha=0.3, color='orange')
        ax2.set_title('Cumulative Variance Explained', fontweight='bold')
        ax2.set_xlabel('IMF')
        ax2.set_ylabel('Cumulative Variance')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, (name, cum_var) in enumerate(zip(imf_names, cumulative_var)):
            ax2.text(i, cum_var + 0.01, f'{cum_var:.1%}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def _create_quality_plot(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create quality metrics visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        quality_metrics = kwargs.get('quality_metrics', {})
        
        # Orthogonality score
        orthogonality = quality_metrics.get('orthogonality_score', 0)
        axes[0, 0].bar(['Orthogonality'], [orthogonality], color='lightcoral', alpha=0.7)
        axes[0, 0].set_title('Orthogonality Score', fontweight='bold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].text(0, orthogonality + 0.01, f'{orthogonality:.4f}', 
                       ha='center', va='bottom', fontweight='bold')
        
        # Reconstruction error
        recon_error = quality_metrics.get('reconstruction_error', 0)
        axes[0, 1].bar(['Reconstruction Error'], [recon_error], color='lightblue', alpha=0.7)
        axes[0, 1].set_title('Reconstruction Error', fontweight='bold')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].text(0, recon_error + 0.01, f'{recon_error:.4f}', 
                       ha='center', va='bottom', fontweight='bold')
        
        # Mean IMF quality
        mean_quality = quality_metrics.get('mean_imf_quality', 0)
        axes[1, 0].bar(['Mean IMF Quality'], [mean_quality], color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Mean IMF Quality', fontweight='bold')
        axes[1, 0].set_ylabel('Quality Score')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].text(0, mean_quality + 0.01, f'{mean_quality:.4f}', 
                       ha='center', va='bottom', fontweight='bold')
        
        # Seasonality strength
        seasonality = quality_metrics.get('seasonality_strength', 0)
        axes[1, 1].bar(['Seasonality Strength'], [seasonality], color='gold', alpha=0.7)
        axes[1, 1].set_title('Seasonality Strength', fontweight='bold')
        axes[1, 1].set_ylabel('Strength')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].text(0, seasonality + 0.01, f'{seasonality:.4f}', 
                       ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('EEMD Quality Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _create_comprehensive_plot(self, data: pd.DataFrame, **kwargs) -> plt.Figure:
        """Create comprehensive EEMD visualization."""
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Original series
        ax1 = fig.add_subplot(gs[0, :2])
        original_data = kwargs.get('original_data', data.iloc[:, 0] if len(data.columns) > 0 else pd.Series())
        if not original_data.empty:
            # Handle dimension mismatch due to downsampling
            original_length = len(original_data)
            data_length = len(data)
            
            if original_length != data_length:
                # Use the shorter length for plotting
                plot_length = min(original_length, data_length)
                original_plot = original_data.iloc[:plot_length]
                
                if isinstance(original_data.index, pd.DatetimeIndex):
                    x_values = original_data.index[:plot_length]
                else:
                    x_values = range(plot_length)
            else:
                original_plot = original_data
                if isinstance(original_data.index, pd.DatetimeIndex):
                    x_values = original_data.index
                else:
                    x_values = range(len(original_data))
            
            ax1.plot(x_values, original_plot.values, linewidth=1, color='black')
            ax1.set_title('Original Time Series', fontweight='bold')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Value')
            ax1.grid(True, alpha=0.3)
        
        # IMFs (first 4)
        imf_cols = [col for col in data.columns if col.startswith('imf_')][:4]
        for i, imf_col in enumerate(imf_cols):
            ax = fig.add_subplot(gs[1, i])
            ax.plot(data.index, data[imf_col], linewidth=0.8, color=f'C{i}')
            ax.set_title(f'{imf_col.upper()}', fontweight='bold', fontsize=10)
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        
        # Correlations
        ax_corr = fig.add_subplot(gs[2, :2])
        correlations = kwargs.get('correlations', [])
        if correlations:
            imf_names = [f'IMF {i+1}' for i in range(len(correlations))]
            bars = ax_corr.bar(imf_names, correlations, color='skyblue', alpha=0.7)
            ax_corr.set_title('IMF Correlations', fontweight='bold')
            ax_corr.set_xlabel('IMF')
            ax_corr.set_ylabel('Correlation')
            ax_corr.grid(True, alpha=0.3)
            ax_corr.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, corr in zip(bars, correlations):
                height = bar.get_height()
                ax_corr.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{corr:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Variance explained
        ax_var = fig.add_subplot(gs[2, 2:])
        variance_explained = kwargs.get('variance_explained', [])
        if variance_explained:
            imf_names = [f'IMF {i+1}' for i in range(len(variance_explained))]
            bars = ax_var.bar(imf_names, variance_explained, color='lightgreen', alpha=0.7)
            ax_var.set_title('Variance Explained', fontweight='bold')
            ax_var.set_xlabel('IMF')
            ax_var.set_ylabel('Variance')
            ax_var.grid(True, alpha=0.3)
            ax_var.tick_params(axis='x', rotation=45)
            
            # Add percentage labels
            for bar, var in zip(bars, variance_explained):
                height = bar.get_height()
                ax_var.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{var:.1%}', ha='center', va='bottom', fontsize=9)
        
        # Quality metrics summary
        ax_qual = fig.add_subplot(gs[3, :])
        quality_metrics = kwargs.get('quality_metrics', {})
        
        if quality_metrics:
            metrics_names = ['Orthogonality', 'Reconstruction Error', 'Mean Quality', 'Seasonality']
            metrics_values = [
                quality_metrics.get('orthogonality_score', 0),
                quality_metrics.get('reconstruction_error', 0),
                quality_metrics.get('mean_imf_quality', 0),
                quality_metrics.get('seasonality_strength', 0)
            ]
            
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold']
            bars = ax_qual.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
            ax_qual.set_title('Quality Metrics Summary', fontweight='bold')
            ax_qual.set_ylabel('Score')
            ax_qual.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, metrics_values):
                height = bar.get_height()
                ax_qual.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Comprehensive EEMD Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_imfs(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> None:
        """Plot IMFs and optionally save."""
        fig = self._create_imfs_plot(data, **kwargs)
        if save_path:
            self.save_visualization(fig, save_path, **kwargs)
        else:
            plt.show()
    
    def plot_correlations(self, correlations: List[float], save_path: Optional[str] = None, **kwargs) -> None:
        """Plot correlations and optionally save."""
        data = pd.DataFrame({'correlations': correlations})
        fig = self._create_correlations_plot(data, correlations=correlations, **kwargs)
        if save_path:
            self.save_visualization(fig, save_path, **kwargs)
        else:
            plt.show()
    
    def plot_variance(self, variance_explained: List[float], save_path: Optional[str] = None, **kwargs) -> None:
        """Plot variance explained and optionally save."""
        data = pd.DataFrame({'variance': variance_explained})
        fig = self._create_variance_plot(data, variance_explained=variance_explained, **kwargs)
        if save_path:
            self.save_visualization(fig, save_path, **kwargs)
        else:
            plt.show()
    
    def plot_quality(self, quality_metrics: Dict[str, float], save_path: Optional[str] = None, **kwargs) -> None:
        """Plot quality metrics and optionally save."""
        data = pd.DataFrame([quality_metrics])
        fig = self._create_quality_plot(data, quality_metrics=quality_metrics, **kwargs)
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