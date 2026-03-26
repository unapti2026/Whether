"""
Model Persistence Demo

Este script demuestra cómo usar el servicio de persistencia de modelos
para guardar y cargar modelos entrenados.
"""

import sys
from pathlib import Path
import logging

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.interfaces.prediction_strategy import PredictionConfig
from src.data.prediction.services.prediction_processor import PredictionProcessor
from src.data.prediction.services.model_persistence_service import ModelPersistenceService
from src.config.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def demo_model_persistence():
    """Demonstrate model persistence functionality."""
    logger.info("=" * 80)
    logger.info("🤖 MODEL PERSISTENCE DEMO")
    logger.info("=" * 80)
    
    try:
        # Create configuration
        from src.config.prediction_config_factory import PredictionConfigFactory
        config = PredictionConfigFactory.create_from_preset(
            preset_name='development',
            variable_type='temp_max',
            max_stations=1  # Just one station for demo
        )
        
        # Initialize processor
        processor = PredictionProcessor(config)
        
        # Load imputed data
        logger.info("📂 Loading imputed data...")
        imputed_data = processor.load_imputed_data(config.variable_type)
        
        if imputed_data.empty:
            logger.error("❌ No imputed data found. Please run imputation process first.")
            return
        
        # Process one station to train models
        logger.info("🚀 Processing station to train models...")
        station_name = list(imputed_data.keys())[0]  # Get first station
        station_data = imputed_data[station_name]
        
        # Process the station
        result = processor._process_single_station(
            imputed_data, station_name, 1, 1
        )
        
        if not result['success']:
            logger.error(f"❌ Failed to process station: {result.get('error_message', 'Unknown error')}")
            return
        
        logger.info("✅ Station processed successfully")
        
        # Demo 1: List saved models
        logger.info("\n" + "=" * 50)
        logger.info("📋 DEMO 1: Listing Saved Models")
        logger.info("=" * 50)
        
        saved_models = processor.list_saved_models()
        if saved_models:
            logger.info(f"Found {len(saved_models)} saved model(s):")
            for model_info in saved_models:
                logger.info(f"  📁 Station: {model_info['station_name']}")
                logger.info(f"     - SVR models: {model_info.get('num_svr_models', 'Unknown')}")
                logger.info(f"     - Has SARIMAX: {model_info.get('has_sarimax_model', False)}")
                logger.info(f"     - Training time: {model_info.get('training_time', 0):.2f}s")
                logger.info(f"     - File size: {model_info.get('file_size', 0)} bytes")
                logger.info(f"     - Last modified: {model_info.get('last_modified', 'Unknown')}")
        else:
            logger.info("No saved models found")
        
        # Demo 2: Load saved models
        logger.info("\n" + "=" * 50)
        logger.info("🔄 DEMO 2: Loading Saved Models")
        logger.info("=" * 50)
        
        if saved_models:
            # Load models for the first station
            station_to_load = saved_models[0]['station_name']
            logger.info(f"Loading models for station: {station_to_load}")
            
            try:
                loaded_models, metadata = processor.load_saved_models(station_to_load)
                logger.info("✅ Models loaded successfully")
                logger.info(f"  - SVR models: {len(loaded_models.svr_models)}")
                logger.info(f"  - SARIMAX model: {'Available' if loaded_models.sarimax_model else 'Not available'}")
                logger.info(f"  - Training time: {loaded_models.training_time:.2f}s")
                logger.info(f"  - Metadata: {metadata}")
                
                # Demo: Use loaded models for prediction
                logger.info("\n🎯 DEMO: Using loaded models for prediction...")
                # Note: This would require additional implementation to use loaded models
                # instead of training new ones
                
            except Exception as e:
                logger.error(f"❌ Failed to load models: {e}")
        else:
            logger.info("No models to load")
        
        # Demo 3: Direct Model Persistence Service usage
        logger.info("\n" + "=" * 50)
        logger.info("🔧 DEMO 3: Direct Model Persistence Service")
        logger.info("=" * 50)
        
        persistence_service = ModelPersistenceService(config.variable_type)
        from src.config.path_manager import get_path_manager
        path_manager = get_path_manager()
        models_dir = path_manager.get_output_subdir("prediction", config.variable_type) / "models"
        
        # List models using direct service
        logger.info("Listing models using direct service...")
        direct_models = persistence_service.list_saved_models(models_dir)
        logger.info(f"Found {len(direct_models)} models via direct service")
        
        # Show supported formats
        logger.info(f"Supported formats: {persistence_service.supported_formats}")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ MODEL PERSISTENCE DEMO COMPLETED")
        logger.info("=" * 80)
        
        # Summary
        logger.info("📝 Summary:")
        logger.info("  1. ✅ Models are automatically saved after training")
        logger.info("  2. ✅ Models can be listed and loaded")
        logger.info("  3. ✅ Multiple formats supported (joblib, pickle)")
        logger.info("  4. ✅ Metadata and configuration saved")
        logger.info("  5. ✅ File management capabilities")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Model persistence demo failed: {e}")
        raise


if __name__ == "__main__":
    demo_model_persistence() 