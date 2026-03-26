#!/usr/bin/env python3
"""
Weather Prediction System - Main Entry Point

Professional CLI interface for the meteorological prediction system.
This is the single entry point for all operations.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.path_manager import get_path_manager
from src.config.user_messages import get_message
from src.config.preprocessing_config import VariableType
from src.config.settings import METEOROLOGICAL_VARIABLES


class WeatherPredictCLI:
    """Main CLI interface for weather prediction system."""
    
    def __init__(self):
        """Initialize CLI with path manager."""
        self.path_manager = get_path_manager()
        self.path_manager.ensure_directories()
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("weather_predict")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Sistema de Predicción Meteorológica",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Ejemplos:
  # Procesar temperatura máxima con todas las estaciones
  weather-predict temp_max
  
  # Procesar con horizonte de 4 semanas
  weather-predict temp_max --horizon-weeks 4
  
  # Procesar solo 5 estaciones
  weather-predict temp_max --max-stations 5
  
  # Solo etapa de predicción
  weather-predict temp_max --stage prediction
            """
        )
        
        parser.add_argument(
            'variable',
            choices=['temp_max', 'temp_min', 'precipitation', 'humidity'],
            help='Variable meteorológica a procesar'
        )
        
        parser.add_argument(
            '--horizon-weeks',
            type=int,
            default=None,
            help='Horizonte de predicción en semanas (default: desde archivo de configuración)'
        )
        
        parser.add_argument(
            '--horizon-days',
            type=int,
            default=None,
            help='Horizonte de predicción en días (sobrescribe semanas)'
        )
        
        parser.add_argument(
            '--max-stations',
            type=int,
            default=None,
            help='Número máximo de estaciones a procesar (default: todas)'
        )
        
        parser.add_argument(
            '--stage',
            choices=['all', 'preprocess', 'imputation', 'prediction'],
            default='all',
            help='Etapa a ejecutar (default: all)'
        )
        
        parser.add_argument(
            '--config',
            type=str,
            default=None,
            help='Archivo de configuración YAML (opcional)'
        )
        
        parser.add_argument(
            '--data-dir',
            type=str,
            default=None,
            help='Directorio de datos de entrada (sobrescribe configuración)'
        )
        
        parser.add_argument(
            '--output-dir',
            type=str,
            default=None,
            help='Directorio de salida (sobrescribe configuración)'
        )
        
        parser.add_argument(
            '--no-plots',
            action='store_true',
            help='Deshabilitar generación de gráficos'
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Modo verbose (más información)'
        )
        
        return parser.parse_args()
    
    def validate_inputs(self, args: argparse.Namespace) -> None:
        """Validate input arguments and files."""
        # Check variable data file exists
        data_file = self.path_manager.get_variable_data_file(args.variable, "xlsx")
        if not data_file.exists():
            raise FileNotFoundError(
                get_message("errors", "file_not_found") + f": {data_file}"
            )
        
        # Check thresholds file if processing temp_max or temp_min
        if args.variable in ['temp_max', 'temp_min']:
            if not self.path_manager.thresholds_file.exists():
                self.logger.warning(
                    f"Archivo de umbrales no encontrado: {self.path_manager.thresholds_file}"
                )
    
    def run(self) -> int:
        """Execute the CLI."""
        try:
            args = self.parse_arguments()
            
            if args.verbose:
                self.logger.setLevel(logging.DEBUG)
            
            self.logger.info(get_message("pipeline", "initialization"))
            self.logger.info(f"  {get_message('pipeline', 'variable')}: {args.variable}")
            
            self.validate_inputs(args)
            
            # Import pipeline here to avoid circular imports
            from notebooks.meteorological_pipeline import (
                MeteorologicalPipeline,
                create_pipeline_config,
                PipelineStage
            )
            
            # Map stage string to PipelineStage
            stage_mapping = {
                'all': PipelineStage.ALL,
                'preprocess': PipelineStage.PREPROCESS,
                'imputation': PipelineStage.IMPUTATION,
                'prediction': PipelineStage.PREDICTION
            }
            
            # Create configuration (YAML -> ENV -> CLI priority handled in create_pipeline_config)
            config = create_pipeline_config(
                variable_name=args.variable,
                max_stations=args.max_stations,
                stages=args.stage,
                enable_plots=None if args.no_plots else True,
                prediction_horizon_weeks=args.horizon_weeks,
                prediction_horizon_days=args.horizon_days,
                output_base_dir=args.output_dir
            )
            
            # Create and run pipeline
            pipeline = MeteorologicalPipeline(config)
            summary = pipeline.run()
            
            # Print summary
            self._print_summary(summary)
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("\nOperación cancelada por el usuario")
            return 130
        except Exception as e:
            self.logger.error(f"{get_message('errors', 'processing_failed')}: {e}")
            if args.verbose if 'args' in locals() else False:
                import traceback
                traceback.print_exc()
            return 1
    
    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Print execution summary."""
        print("\n" + "=" * 60)
        print(get_message("summary", "title"))
        print("=" * 60)
        print(f"{get_message('summary', 'variable')}: {summary['pipeline_config']['variable_name']}")
        print(f"{get_message('summary', 'stages_executed')}: {', '.join(summary['pipeline_config']['stages_executed'])}")
        print(f"{get_message('summary', 'success_rate')}: {summary['execution_summary']['success_rate']:.1f}%")
        print(f"{get_message('summary', 'total_time')}: {summary['execution_summary']['total_time']:.2f}s")
        print(f"{get_message('summary', 'output_directories')}:")
        for stage, path in summary['output_directories'].items():
            print(f"  {stage}: {path}")


def main():
    """Main entry point."""
    cli = WeatherPredictCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())

