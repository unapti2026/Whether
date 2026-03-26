"""
Meteorological Data Processing Pipeline

This script orchestrates the complete meteorological data processing pipeline:
1. Preprocessing: Clean and structure raw data
2. Imputation: Fill missing values
3. Prediction: Generate future predictions

The pipeline is configurable for different variables and station counts.
"""

import argparse
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path to import modules
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.preprocessing_config import VariableType
from src.config.settings import METEOROLOGICAL_VARIABLES


class PipelineStage(Enum):
    """Pipeline processing stages."""
    PREPROCESS = "preprocess"
    IMPUTATION = "imputation"
    PREDICTION = "prediction"
    ALL = "all"


@dataclass
class PipelineConfig:
    """Configuration for the meteorological pipeline."""
    
    # Variable configuration
    variable_type: VariableType
    variable_name: str
    
    # Station configuration
    max_stations: Optional[int] = None  # None = all stations
    
    # Prediction horizon configuration (NEW)
    prediction_horizon_weeks: Optional[int] = None
    prediction_horizon_days: Optional[int] = None
    use_legacy_horizon: bool = False
    
    # Pipeline stages to execute
    stages: PipelineStage = PipelineStage.ALL
    
    # Processing configuration
    enable_validation: bool = True
    enable_logging: bool = True
    enable_plots: bool = True
    
    # Output configuration
    output_base_dir: Optional[str] = None  # Will use PathManager default if None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.variable_type.value not in METEOROLOGICAL_VARIABLES:
            raise ValueError(f"Unsupported variable type: {self.variable_type.value}")
        
        if self.max_stations is not None and self.max_stations <= 0:
            raise ValueError("max_stations must be positive or None")
    
    @property
    def variable_info(self) -> Dict[str, Any]:
        """Get variable information from settings."""
        return METEOROLOGICAL_VARIABLES[self.variable_type.value]
    
    @property
    def output_dirs(self) -> Dict[str, Path]:
        """Get output directories for each stage."""
        from src.config.path_manager import get_path_manager
        path_manager = get_path_manager()
        
        if self.output_base_dir:
            base_path = Path(self.output_base_dir)
        else:
            base_path = path_manager.output_dir
        
        return {
            'preprocess': base_path / "preprocessing" / self.variable_name,
            'imputation': base_path / "imputation" / self.variable_name,
            'prediction': base_path / "prediction" / self.variable_name
        }


class MeteorologicalPipeline:
    """
    Main pipeline orchestrator for meteorological data processing.
    
    This class coordinates the execution of preprocessing, imputation, and prediction
    stages with comprehensive logging and error handling.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = self._setup_logger()
        self.start_time = None
        self.results = {}
        
        # Validate configuration
        self._validate_config()
        
        self.logger.info("Meteorological Pipeline initialized")
        self.logger.info(f"   Variable: {self.config.variable_name}")
        self.logger.info(f"   Max stations: {self.config.max_stations if self.config.max_stations else 'All'}")
        self.logger.info(f"   Stages: {self.config.stages.value}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the pipeline."""
        logger = logging.getLogger(f"meteorological_pipeline_{self.config.variable_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            from src.config.path_manager import get_path_manager
            path_manager = get_path_manager()
            log_dir = path_manager.logs_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / f"pipeline_{self.config.variable_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _validate_config(self) -> None:
        """Validate pipeline configuration."""
        # Check if data files exist
        from src.config.path_manager import get_path_manager
        path_manager = get_path_manager()
        excel_file = path_manager.get_variable_data_file(self.config.variable_name, "xlsx")
        csv_file = path_manager.get_variable_data_file(self.config.variable_name, "csv")
        
        if not excel_file.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_file}")
        
        self.logger.info(f"Data files validated:")
        self.logger.info(f"   Excel: {excel_file}")
        self.logger.info(f"   CSV: {csv_file}")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline.
        
        Returns:
            Dictionary with results from all stages
        """
        self.start_time = time.time()
        
        self.logger.info("Starting Meteorological Pipeline")
        self.logger.info("=" * 60)
        
        try:
            # Determine which stages to run
            stages_to_run = self._get_stages_to_run()
            
            # Execute each stage
            for stage in stages_to_run:
                self._execute_stage(stage)
            
            # Generate final summary
            summary = self._generate_summary()
            
            self.logger.info("=" * 60)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"   Total time: {time.time() - self.start_time:.2f}s")
            self.logger.info(f"   Results: {self.output_dirs}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _get_stages_to_run(self) -> list:
        """Determine which stages to run based on configuration."""
        if self.config.stages == PipelineStage.ALL:
            return [PipelineStage.PREPROCESS, PipelineStage.IMPUTATION, PipelineStage.PREDICTION]
        elif self.config.stages == PipelineStage.PREPROCESS:
            return [PipelineStage.PREPROCESS]
        elif self.config.stages == PipelineStage.IMPUTATION:
            return [PipelineStage.PREPROCESS, PipelineStage.IMPUTATION]
        elif self.config.stages == PipelineStage.PREDICTION:
            return [PipelineStage.PREPROCESS, PipelineStage.IMPUTATION, PipelineStage.PREDICTION]
        else:
            raise ValueError(f"Unknown stage: {self.config.stages}")
    
    def _execute_stage(self, stage: PipelineStage) -> None:
        """
        Execute a specific pipeline stage.
        
        Args:
            stage: Stage to execute
        """
        stage_start_time = time.time()
        
        self.logger.info(f"Executing {stage.value.upper()} stage...")
        self.logger.info("-" * 40)
        
        try:
            if stage == PipelineStage.PREPROCESS:
                result = self._run_preprocessing()
            elif stage == PipelineStage.IMPUTATION:
                result = self._run_imputation()
            elif stage == PipelineStage.PREDICTION:
                result = self._run_prediction()
            else:
                raise ValueError(f"Unknown stage: {stage}")
            
            stage_time = time.time() - stage_start_time
            self.results[stage.value] = {
                'success': True,
                'processing_time': stage_time,
                'result': result
            }
            
            self.logger.info(f"{stage.value.title()} completed in {stage_time:.2f}s")
            
        except Exception as e:
            stage_time = time.time() - stage_start_time
            self.results[stage.value] = {
                'success': False,
                'processing_time': stage_time,
                'error': str(e)
            }
            
            self.logger.error(f"{stage.value.title()} failed: {e}")
            
            # For critical stages, stop the pipeline
            if stage in [PipelineStage.PREPROCESS, PipelineStage.IMPUTATION]:
                raise
    
    def _run_preprocessing(self) -> Dict[str, Any]:
        """
        Run the preprocessing stage.
        
        Returns:
            Preprocessing results
        """
        self.logger.info("Running preprocessing...")
        
        # Import and run preprocessing
        import subprocess
        import sys
        
        # Build command
        cmd = [
            sys.executable, "preprocess_meteorological.py",
            self.config.variable_name
        ]
        
        # Add optional parameters
        if self.config.max_stations:
            cmd.extend(["--max-stations", str(self.config.max_stations)])
        
        if not self.config.enable_validation:
            cmd.append("--no-validation")
        
        if not self.config.enable_logging:
            cmd.append("--no-logging")
        
        if not self.config.enable_plots:
            cmd.append("--no-plots")
        
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        # Execute preprocessing (don't capture output to avoid blocking - let it stream to console)
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Preprocessing failed with return code {result.returncode}")
        
        self.logger.info("Preprocessing completed successfully")
        return {'output': '', 'return_code': result.returncode}
    
    def _run_imputation(self) -> Dict[str, Any]:
        """
        Run the imputation stage.
        
        Returns:
            Imputation results
        """
        self.logger.info("Running imputation...")
        
        # Import and run imputation
        import subprocess
        import sys
        
        # Build command
        cmd = [
            sys.executable, "imputation_process.py"
        ]
        
        # Add required parameters
        cmd.extend(["--variable", self.config.variable_name])
        
        # Add output directory to ensure consistency
        from src.config.path_manager import get_path_manager
        path_manager = get_path_manager()
        output_base = path_manager.get_output_subdir("imputation", self.config.variable_name).parent
        cmd.extend(["--output-dir", str(output_base)])
        
        # Add optional parameters
        if self.config.max_stations:
            cmd.extend(["--max-stations", str(self.config.max_stations)])
        
        if not self.config.enable_validation:
            cmd.append("--no-validation")
        
        if not self.config.enable_logging:
            cmd.append("--no-logging")
        
        if not self.config.enable_plots:
            cmd.append("--no-plots")
        
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        # Execute imputation (don't capture output to avoid blocking - let it stream to console)
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Imputation failed with return code {result.returncode}")
        
        self.logger.info("Imputation completed successfully")
        return {'output': '', 'return_code': result.returncode}
    
    def _run_prediction(self) -> Dict[str, Any]:
        """
        Run the prediction stage.
        
        Returns:
            Prediction results
        """
        self.logger.info("Running prediction...")
        
        # Import and run prediction
        import subprocess
        import sys
        
        # Build command
        cmd = [
            sys.executable, "prediction_process.py"
        ]
        
        # Add required parameters
        cmd.extend(["--variable", self.config.variable_name])
        
        # Add optional parameters
        if self.config.max_stations:
            cmd.extend(["--max-stations", str(self.config.max_stations)])
        
        # Add prediction horizon parameters (NEW)
        if self.config.prediction_horizon_days:
            cmd.extend(["--prediction-horizon-days", str(self.config.prediction_horizon_days)])
        elif self.config.prediction_horizon_weeks:
            cmd.extend(["--prediction-horizon-weeks", str(self.config.prediction_horizon_weeks)])
        
        if self.config.use_legacy_horizon:
            cmd.append("--use-legacy-horizon")
        
        if not self.config.enable_validation:
            cmd.append("--no-validation")
        
        if not self.config.enable_logging:
            cmd.append("--no-logging")
        
        if not self.config.enable_plots:
            cmd.append("--no-plots")
        
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        # Execute prediction (don't capture output to avoid blocking - let it stream to console)
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Prediction failed with return code {result.returncode}")
        
        self.logger.info("Prediction completed successfully")
        return {'output': '', 'return_code': result.returncode}
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of pipeline execution.
        
        Returns:
            Pipeline summary
        """
        total_time = time.time() - self.start_time
        
        # Calculate success rates
        successful_stages = sum(1 for result in self.results.values() if result['success'])
        total_stages = len(self.results)
        success_rate = (successful_stages / total_stages * 100) if total_stages > 0 else 0
        
        # Calculate total processing time
        total_processing_time = sum(
            result['processing_time'] for result in self.results.values()
        )
        
        summary = {
            'pipeline_config': {
                'variable_name': self.config.variable_name,
                'variable_type': self.config.variable_type.value,
                'max_stations': self.config.max_stations,
                'stages_executed': list(self.results.keys())
            },
            'execution_summary': {
                'total_time': total_time,
                'total_processing_time': total_processing_time,
                'successful_stages': successful_stages,
                'total_stages': total_stages,
                'success_rate': success_rate,
                'start_time': self.start_time,
                'end_time': time.time()
            },
            'stage_results': self.results,
            'output_directories': {
                stage: str(path) for stage, path in self.config.output_dirs.items()
            }
        }
        
        return summary
    
    @property
    def output_dirs(self) -> Dict[str, Path]:
        """Get output directories."""
        return self.config.output_dirs


def create_pipeline_config(variable_name: str,
                          max_stations: Optional[int] = None,
                          stages: str = "all",
                          enable_validation: Optional[bool] = None,
                          enable_logging: Optional[bool] = None,
                          enable_plots: Optional[bool] = None,
                          prediction_horizon_weeks: Optional[int] = None,
                          prediction_horizon_days: Optional[int] = None,
                          use_legacy_horizon: bool = False,
                          output_base_dir: Optional[str] = None) -> PipelineConfig:
    """
    Create a pipeline configuration with priority: YAML -> ENV -> CLI.
    
    Args:
        variable_name: Name of the variable to process
        max_stations: Maximum number of stations (None = use config)
        stages: Stages to execute
        enable_validation: Enable validation (None = use config)
        enable_logging: Enable logging (None = use config)
        enable_plots: Enable plot generation (None = use config)
        prediction_horizon_weeks: Prediction horizon in weeks (None = use config)
        prediction_horizon_days: Prediction horizon in days (None = use config)
        use_legacy_horizon: Use legacy percentage-based horizon
        output_base_dir: Output directory (None = use config)
        
    Returns:
        Pipeline configuration
    """
    from src.config.yaml_config_loader import YAMLConfigLoader
    
    cli_args = {
        'max_stations': max_stations,
        'enable_validation': enable_validation,
        'enable_logging': enable_logging,
        'enable_plots': enable_plots,
        'prediction_horizon_weeks': prediction_horizon_weeks,
        'prediction_horizon_days': prediction_horizon_days,
        'output_dir': output_base_dir
    }
    
    config_loader = YAMLConfigLoader()
    merged_config = config_loader.get_merged_config(cli_args)
    
    variable_mapping = {
        'temp_max': VariableType.TEMP_MAX,
        'temp_min': VariableType.TEMP_MIN,
        'precipitation': VariableType.PRECIPITATION,
        'humidity': VariableType.HUMIDITY
    }
    
    if variable_name not in variable_mapping:
        raise ValueError(f"Unsupported variable: {variable_name}. Supported: {list(variable_mapping.keys())}")
    
    stage_mapping = {
        'all': PipelineStage.ALL,
        'preprocess': PipelineStage.PREPROCESS,
        'imputation': PipelineStage.IMPUTATION,
        'prediction': PipelineStage.PREDICTION
    }
    
    if stages not in stage_mapping:
        raise ValueError(f"Unsupported stages: {stages}. Supported: {list(stage_mapping.keys())}")
    
    processing = merged_config.get('processing', {})
    output = merged_config.get('output', {})
    prediction = merged_config.get('prediction', {})
    
    return PipelineConfig(
        variable_type=variable_mapping[variable_name],
        variable_name=variable_name,
        max_stations=processing.get('max_stations') if max_stations is None else max_stations,
        stages=stage_mapping[stages],
        enable_validation=processing.get('enable_validation', True) if enable_validation is None else enable_validation,
        enable_logging=processing.get('enable_logging', True) if enable_logging is None else enable_logging,
        enable_plots=output.get('enable_plots', True) if enable_plots is None else enable_plots,
        prediction_horizon_weeks=prediction.get('default_horizon_weeks') if prediction_horizon_weeks is None else prediction_horizon_weeks,
        prediction_horizon_days=prediction.get('default_horizon_days') if prediction_horizon_days is None else prediction_horizon_days,
        use_legacy_horizon=use_legacy_horizon,
        output_base_dir=output.get('base_dir') if output_base_dir is None else output_base_dir
    )


def main():
    """Main function to run the pipeline from command line."""
    parser = argparse.ArgumentParser(
        description="Meteorological Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all stages for precipitation with all stations
  python meteorological_pipeline.py precipitation
  
  # Process only preprocessing for temperature max with 5 stations
  python meteorological_pipeline.py temp_max --max-stations 5 --stages preprocess
  
  # Process imputation and prediction for humidity
  python meteorological_pipeline.py humidity --stages imputation
  
  # Process all stages for temp_min with 10 stations and no plots
  python meteorological_pipeline.py temp_min --max-stations 10 --no-plots
        """
    )
    
    parser.add_argument(
        'variable',
        choices=['temp_max', 'temp_min', 'precipitation', 'humidity'],
        help='Variable to process'
    )
    
    parser.add_argument(
        '--max-stations',
        type=int,
        default=None,
        help='Maximum number of stations to process (default: all)'
    )
    
    parser.add_argument(
        '--stages',
        choices=['all', 'preprocess', 'imputation', 'prediction'],
        default='all',
        help='Pipeline stages to execute (default: all)'
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
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Base output directory (default: uses PathManager)'
    )
    
    parser.add_argument(
        '--prediction-horizon-weeks',
        type=int,
        default=None,
        help='Prediction horizon in weeks (default: from config file)'
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
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = create_pipeline_config(
            variable_name=args.variable,
            max_stations=args.max_stations,
            stages=args.stages,
            enable_validation=not args.no_validation,
            enable_logging=not args.no_logging,
            enable_plots=not args.no_plots,
            prediction_horizon_weeks=args.prediction_horizon_weeks,
            prediction_horizon_days=args.prediction_horizon_days,
            use_legacy_horizon=args.use_legacy_horizon
        )
        
        # Create and run pipeline
        pipeline = MeteorologicalPipeline(config)
        summary = pipeline.run()
        
        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Variable: {summary['pipeline_config']['variable_name']}")
        print(f"Stages executed: {', '.join(summary['pipeline_config']['stages_executed'])}")
        print(f"Success rate: {summary['execution_summary']['success_rate']:.1f}%")
        print(f"Total time: {summary['execution_summary']['total_time']:.2f}s")
        print(f"Output directories:")
        for stage, path in summary['output_directories'].items():
            print(f"  {stage}: {path}")
        
        return 0
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
