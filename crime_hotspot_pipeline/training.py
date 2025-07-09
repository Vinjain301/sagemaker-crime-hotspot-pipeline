"""
SageMaker Training Job Script
Handles model training and evaluation
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
import json
import joblib
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append('/opt/ml/code')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    # Model hyperparameters
    parser.add_argument('--model_type', type=str, default='random_forest')
    parser.add_argument('--use_smote', type=str, default='true')
    
    args = parser.parse_args()
    
    logger.info(f"Starting training with model type: {args.model_type}")
    
    try:
        # Import modules
        from model.model_train import ModelTrainer
        from model.model_evaluate import ModelEvaluator
        from data.data_splitter import TimeAwareSplitter
        
        # Load data
        logger.info("Loading training data...")
        train_df = pd.read_parquet(os.path.join(args.train, 'train.parquet'))
        test_df = pd.read_parquet(os.path.join(args.test, 'test.parquet'))
        
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        # Split features and target
        splitter = TimeAwareSplitter()
        features_to_exclude = ['is_hourly_hotspot', 'Hourly_Crime_Rate_per_1000', 'Datetime_Key']
        
        X_train, y_train = splitter.split_features_target(
            train_df, 'is_hourly_hotspot', features_to_exclude
        )
        X_test, y_test = splitter.split_features_target(
            test_df, 'is_hourly_hotspot', features_to_exclude
        )
        
        # Initialize trainer
        trainer = ModelTrainer('/opt/ml/code/config/model_config.yaml')
        
        # Train model
        use_smote = args.use_smote.lower() == 'true'
        model = trainer.train_model(
            X_train, y_train,
            model_type=args.model_type,
            use_smote=use_smote,
            tune_hyperparameters=False
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(
            model, X_test, y_test,
            model_name=f"{args.model_type}_{'smote' if use_smote else 'no_smote'}"
        )
        
        # Log metrics
        logger.info("Model Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
        
        # Save model
        model_path = os.path.join(args.model_dir, 'model.joblib')
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save evaluation metrics
        metrics_path = os.path.join(args.model_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise


if __name__ == "__main__":
    main()