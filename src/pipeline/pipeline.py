"""
Main pipeline orchestrator for surgical robotics
Coordinates data handling, training, and evaluation
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import shutil

from .data_handler import DataHandler
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

class CompletePipeline:
    """Complete surgical robotics pipeline orchestrator"""
    
    def __init__(self, kaggle_dataset: str = "kvasir-seg", output_dir: str = "data"):
        self.kaggle_dataset = kaggle_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_handler = DataHandler(kaggle_dataset, output_dir)
        self.trainer = ModelTrainer(self.output_dir / "checkpoints")
        self.evaluator = ModelEvaluator(self.output_dir / "results")
        
        logger.info("CompletePipeline initialized")
    
    def run_complete_pipeline(self, il_epochs: int = 50, rl_timesteps: int = 50000) -> bool:
        """Run the complete pipeline"""
        logger.info("🚀 Starting Complete Surgical Robotics Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Download dataset
            if not self.data_handler.download_kaggle_dataset():
                logger.error("❌ Dataset download failed")
                return False
            
            # Step 2: Process and split dataset
            if not self.data_handler.process_and_split_dataset():
                logger.error("❌ Dataset processing failed")
                return False
            
            # Step 3: Train IL model
            il_model_path = self.trainer.train_il_model(
                self.data_handler.get_train_dir(), 
                epochs=il_epochs
            )
            
            # Step 4: Train RL model
            rl_model_path = self.trainer.train_rl_model(timesteps=rl_timesteps)
            
            # Step 5: Evaluate models
            results = self.evaluator.evaluate_models(
                il_model_path=il_model_path,
                rl_model_path=rl_model_path,
                val_dir=self.data_handler.get_val_dir()
            )
            
            # Step 6: Print results
            self.evaluator.print_results(results)
            
            # Step 7: Cleanup
            self.cleanup_temp_files()
            
            logger.info("🎉 Complete pipeline finished successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        logger.info("🧹 Cleaning up temporary files...")
        
        # Remove temporary files
        temp_dirs = [
            self.data_handler.raw_data_dir,
            self.data_handler.processed_data_dir
        ]
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info(f"Removed: {temp_dir}")
        
        # Remove evaluation files
        eval_files = [
            "complete_pipeline.py",
            "run_pipeline.py",
            "run_final.py",
            "requirements_pipeline.txt",
            "test_pipeline.py",
            "PIPELINE_GUIDE.md",
            "run_complete.sh"
        ]
        
        for file_path in eval_files:
            if Path(file_path).exists():
                Path(file_path).unlink()
                logger.info(f"Removed: {file_path}")
        
        logger.info("✅ Cleanup completed")
    
    def get_data_handler(self) -> DataHandler:
        """Get data handler instance"""
        return self.data_handler
    
    def get_trainer(self) -> ModelTrainer:
        """Get trainer instance"""
        return self.trainer
    
    def get_evaluator(self) -> ModelEvaluator:
        """Get evaluator instance"""
        return self.evaluator
