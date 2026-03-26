"""
Prediction Process

This module demonstrates the use of the prediction services
following SOLID principles and Clean Code practices.
"""

import sys
from pathlib import Path
import logging
import argparse
import re

# Add parent directory to path (robust path resolution)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.interfaces.prediction_strategy import PredictionConfig
from src.data.prediction.services.prediction_processor import PredictionProcessor
from src.config.logging_config import setup_logging
from src.config.prediction_config_factory import PredictionConfigFactory

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def create_prediction_config(variable_type: str = "temp_max", max_stations: int = None) -> PredictionConfig:
    """
    Create a prediction configuration with default parameters.
    
    Args:
        variable_type: Type of meteorological variable
        max_stations: Maximum number of stations to process
        
    Returns:
        PredictionConfig instance
    """
    return PredictionConfig(
        variable_type=variable_type,
        max_stations=max_stations,
        prediction_steps=30,
        num_lags=7,
        train_test_split=0.8,
        
        # EEMD Configuration
        eemd_sd_thresh_values=[0.05, 0.1, 0.15, 0.2],
        eemd_nensembles=20,
        eemd_noise_factor=0.1,
        eemd_max_imfs=10,
        eemd_orthogonality_threshold=0.1,
        eemd_correlation_threshold=0.1,
        
        # Model Configuration
        svr_kernel='rbf',
        svr_c=1.0,
        svr_gamma='scale',
        sarimax_order=(1, 1, 0),
        sarimax_seasonal_order=(1, 0, 0, 365)
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Meteorological Data Prediction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    python prediction_process.py                    # Process precipitation (default)
    python prediction_process.py --variable temp_max    # Process temp_max
    python prediction_process.py --variable temp_min    # Process temp_min
    python prediction_process.py --variable humidity     # Process humidity
    python prediction_process.py --help            # Show this help
    
PIPELINE INTEGRATION:
    # Called from pipeline with parameters
    python prediction_process.py --variable precipitation --max-stations 5
        """
    )
    
    parser.add_argument(
        '--variable',
        default='precipitation',
        choices=['temp_max', 'temp_min', 'precipitation', 'humidity'],
        help='Meteorological variable to process (default: precipitation)'
    )
    
    parser.add_argument(
        '--max-stations',
        type=int,
        default=None,
        help='Maximum number of stations to process (default: None, processes all stations)'
    )
    
    parser.add_argument(
        '--preset',
        default='development',
        choices=['development', 'production', 'high_quality'],
        help='Configuration preset to use (default: development)'
    )
    
    parser.add_argument(
        '--prediction-steps',
        type=int,
        default=30,
        help='Number of prediction steps (default: 30, legacy mode)'
    )
    
    parser.add_argument(
        '--prediction-horizon-weeks',
        type=int,
        default=3,
        help='Prediction horizon in weeks (default: 3, overrides legacy percentage)'
    )
    
    parser.add_argument(
        '--prediction-horizon-days',
        type=int,
        default=None,
        help='Prediction horizon in days (overrides weeks if specified)'
    )
    
    parser.add_argument(
        '--use-legacy-horizon',
        action='store_true',
        help='Use legacy horizon calculation (20%% of series size) instead of fixed horizon'
    )
    
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable validation'
    )
    
    parser.add_argument(
        '--no-logging',
        action='store_true',
        help='Disable logging'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    from src.config.logging_config import setup_logging
    setup_logging()
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("METEOROLOGICAL PREDICTION PROCESS")
    logger.info("=" * 80)
    
    try:
        # Get available presets
        presets = PredictionConfigFactory.get_available_presets()
        logger.info("Available configuration presets:")
        for name, description in presets.items():
            logger.info(f"   - {name}: {description}")
        
        # Create configuration using factory
        config = PredictionConfigFactory.create_from_preset(
            preset_name=args.preset,
            variable_type=args.variable,
            max_stations=args.max_stations
        )
        
        # Update configuration with command line arguments
        config.prediction_steps = args.prediction_steps
        
        # Configure prediction horizon (NEW: Fixed horizon system)
        if args.use_legacy_horizon:
            # Legacy mode: use percentage-based horizon
            config.use_fixed_horizon = False
            logger.info("Using legacy horizon calculation (percentage-based)")
        else:
            # Fixed horizon mode (default)
            config.use_fixed_horizon = True
            if args.prediction_horizon_days:
                config.prediction_horizon_days = args.prediction_horizon_days
                config.prediction_horizon_weeks = None  # Clear weeks if days specified
                logger.info(f"Using fixed horizon: {args.prediction_horizon_days} days")
            else:
                config.prediction_horizon_weeks = args.prediction_horizon_weeks
                logger.info(f"Using fixed horizon: {args.prediction_horizon_weeks} weeks ({args.prediction_horizon_weeks * 7} days)")
        
        # Handle plot generation flag
        config.enable_plots = not args.no_plots
        
        # CRITICAL: MEMORY OPTIMIZATION PARAMETERS
        # These parameters control the data limitation for SARIMAX
        # You can adjust these values to find the optimal balance between memory usage and performance
        
        # SARIMAX Data Limitation Parameters
        config.sarimax_data_limitation = True  # Enable data limitation
        config.sarimax_max_years_large_series = 2    # For series > 5000 points: use last 2 years
        config.sarimax_max_years_medium_series = 1.5  # For series > 3000 points: use last 1.5 years
        config.sarimax_max_years_small_series = 1     # For series > 1000 points: use last 1 year
        
        # Memory Optimization Flags
        config.enable_memory_cleanup = True
        config.force_garbage_collection = True
        config.use_chunked_training = True
        
        # Performance Tuning
        config.sarimax_max_iterations_large = 30   # Reduced iterations for large series
        config.sarimax_max_iterations_medium = 50  # Medium iterations for medium series
        config.sarimax_max_iterations_small = 100  # Standard iterations for small series
        
        # Timeout Settings (seconds)
        config.sarimax_timeout_large = 20   # 20 seconds for large series
        config.sarimax_timeout_medium = 40  # 40 seconds for medium series
        config.sarimax_timeout_small = 60   # 60 seconds for small series
        
        # Validate configuration
        PredictionConfigFactory.validate_config(config)
        
        logger.info(f"Configuration:")
        logger.info(f"   - Variable: {config.variable_type}")
        logger.info(f"   - Max stations: {config.max_stations}")
        logger.info(f"   - Prediction steps: {config.prediction_steps}")
        logger.info(f"   - Horizon mode: {'Fixed' if config.use_fixed_horizon else 'Legacy (percentage)'}")
        if config.use_fixed_horizon:
            if config.prediction_horizon_days:
                logger.info(f"   - Fixed horizon: {config.prediction_horizon_days} days")
            else:
                logger.info(f"   - Fixed horizon: {config.prediction_horizon_weeks} weeks ({config.prediction_horizon_weeks * 7} days)")
        else:
            logger.info(f"   - Legacy horizon ratio: {config.legacy_horizon_ratio * 100:.0f}%")
        logger.info(f"   - EEMD ensembles: {config.eemd_nensembles}")
        logger.info(f"   - SVR kernel: {config.svr_kernel}")
        logger.info(f"   - SARIMAX order: {config.sarimax_order}")
        logger.info(f"   - EEMD sd_thresh values: {config.eemd_sd_thresh_values}")
        logger.info(f"   - Plot generation: {'Enabled' if config.enable_plots else 'Disabled'}")
        
        # Display memory optimization settings
        logger.info(f"Memory Optimization Settings:")
        logger.info(f"   - SARIMAX data limitation: {config.sarimax_data_limitation}")
        logger.info(f"   - Large series (>5000): {config.sarimax_max_years_large_series} years")
        logger.info(f"   - Medium series (>3000): {config.sarimax_max_years_medium_series} years")
        logger.info(f"   - Small series (>1000): {config.sarimax_max_years_small_series} years")
        logger.info(f"   - Memory cleanup: {config.enable_memory_cleanup}")
        logger.info(f"   - Force garbage collection: {config.force_garbage_collection}")
        logger.info(f"   - Chunked training: {config.use_chunked_training}")
        
        # Initialize processor
        processor = PredictionProcessor(config)
        
        # Load imputed data
        logger.info(f"Loading imputed data...")
        imputed_data = processor.load_imputed_data(config.variable_type)
        
        if imputed_data.empty:
            logger.error("No imputed data found. Please run imputation process first.")
            return
        
        # Apply station limiting if specified
        # CRITICAL: Sort stations by code to ensure consistency with imputation process
        if args.max_stations and args.max_stations > 0:
            unique_stations = imputed_data['Estación'].unique()
            
            # Sort stations by code for consistency
            # Extract code from station name (format: {code}_{station_name}_Imputed)
            station_code_map = {}
            for station in unique_stations:
                try:
                    # Try to extract code from station name
                    # Station name format: {code}_{station_name}_Imputed
                    code_match = re.search(r'^(\d+)_', station)
                    if code_match:
                        station_code_map[station] = int(code_match.group(1))
                    else:
                        # Fallback: try to find any number
                        code_match = re.search(r'(\d+)', station)
                        station_code_map[station] = int(code_match.group(1)) if code_match else 999999
                except (ValueError, AttributeError):
                    station_code_map[station] = 999999
            
            # Sort stations by code (ascending)
            sorted_stations = sorted(unique_stations, key=lambda s: station_code_map.get(s, 999999))
            
            if len(sorted_stations) > args.max_stations:
                logger.info(f'Limiting to {args.max_stations} stations (from {len(sorted_stations)} total, sorted by code)')
                selected_stations = sorted_stations[:args.max_stations]
                imputed_data = imputed_data[imputed_data['Estación'].isin(selected_stations)]
                logger.info(f'Data limited to {len(selected_stations)} stations (sorted by code for consistency)')
                logger.info(f'Selected stations (in order):')
                for i, station in enumerate(selected_stations, 1):
                    code = station_code_map.get(station, 'N/A')
                    logger.info(f'  {i}. [{code}] {station}')
        
        # Process stations
        logger.info(f"Processing stations with architecture...")
        summary = processor.process_stations(imputed_data)
        
        # Save processing summary
        processor.save_processing_summary(summary)
        
        logger.info("=" * 80)
        logger.info("PREDICTION PROCESS COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Summary:")
        logger.info(f"   - Total stations: {summary['total_stations']}")
        logger.info(f"   - Successful: {summary['successful_stations']}")
        logger.info(f"   - Failed: {summary['failed_stations']}")
        logger.info(f"   - Success rate: {summary['success_rate']:.1f}%")
        logger.info("=" * 80)
        
        # Display improvements made
        logger.info("Improvements Made:")
        logger.info("   1. Dependency injection implemented")
        logger.info("   2. Factory pattern for configuration")
        logger.info("   3. Interface-based design")
        logger.info("   4. Better error handling and validation")
        logger.info("   5. Improved separation of concerns")
        logger.info("   6. Enhanced quality metrics")
        logger.info("   7. Professional logging structure")
        logger.info("   8. Memory optimization implemented")
        logger.info("   9. Adaptive configuration system")
        logger.info("  10. Garbage collection management")
        logger.info("  11. SARIMAX data limitation implemented")
        logger.info("  12. Precipitation-specific prediction methods")
        
    except Exception as e:
        logger.error(f"Prediction process failed: {e}")
        raise


if __name__ == "__main__":
    main() 