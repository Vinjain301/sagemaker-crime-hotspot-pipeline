"""
SageMaker Evaluation Script
Evaluates model and outputs metrics for conditional step
"""
import argparse
import os
import json
import pandas as pd
import joblib
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tarfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/opt/ml/processing/model')
    parser.add_argument('--test-path', type=str, default='/opt/ml/processing/test')
    parser.add_argument('--output-path', type=str, default='/opt/ml/processing/evaluation')
    args = parser.parse_args()
    
    logger.info("Starting model evaluation")
    
    # Create output directory
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        model_file = os.path.join(args.model_path, 'model.tar.gz')
        
        # Extract if needed
        if os.path.exists(model_file) and model_file.endswith('.tar.gz'):
            logger.info("Extracting model archive")
            with tarfile.open(model_file) as tar:
                tar.extractall(args.model_path)
        
        # Find the model file
        model_files = list(Path(args.model_path).glob('model.joblib'))
        if not model_files:
            model_files = list(Path(args.model_path).glob('*/model.joblib'))
        
        if not model_files:
            raise FileNotFoundError(f"No model.joblib found in {args.model_path}")
        
        model = joblib.load(str(model_files[0]))
        logger.info("Model loaded successfully")
        
        # Load test data
        test_file = os.path.join(args.test_path, 'test.parquet')
        if not os.path.exists(test_file):
            # Try to find any parquet file
            test_files = list(Path(args.test_path).glob('*.parquet'))
            if not test_files:
                raise FileNotFoundError(f"No test data found in {args.test_path}")
            test_file = str(test_files[0])
        
        test_df = pd.read_parquet(test_file)
        logger.info(f"Loaded test data with shape: {test_df.shape}")
        
        # Split features and target
        features_to_exclude = ['is_hourly_hotspot', 'Hourly_Crime_Rate_per_1000', 'Datetime_Key', 'AREA NAME']
        
        # Get numeric feature columns
        numeric_cols = test_df.select_dtypes(include=['number']).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in features_to_exclude]
        
        X_test = test_df[feature_cols]
        y_test = test_df['is_hourly_hotspot']
        
        logger.info(f"Features shape: {X_test.shape}, Target shape: {y_test.shape}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = None
        
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except:
                logger.warning("Could not get probability predictions")
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0))
        }
        
        if y_proba is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))
            except:
                logger.warning("Could not calculate ROC AUC")
                metrics['roc_auc'] = 0.0
        
        # Save metrics for conditional step
        metrics_path = os.path.join(args.output_path, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Evaluation complete. Metrics: {metrics}")

        sagemaker_metrics = {"metric_groups": [{"name": "ModelQuality", "metric_data": [{"name": key,"type": "number","value": float(value)} for key, value in metrics.items()]}]}
        evaluation_path = os.path.join(args.output_path, 'evaluation.json')
        with open(evaluation_path, 'w') as f:
            json.dump(sagemaker_metrics, f, indent=4)

        
        # Save detailed evaluation report
        report = {
            'metrics': metrics,
            'test_size': len(y_test),
            'positive_class_ratio': float(y_test.sum() / len(y_test)),
            'model_type': model.__class__.__name__,
            'feature_count': len(feature_cols),
            'feature_names': feature_cols
        }
        
        report_path = os.path.join(args.output_path, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        # Write error metrics for pipeline to handle
        error_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'error': str(e)
        }
        metrics_path = os.path.join(args.output_path, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(error_metrics, f, indent=4)
        raise


if __name__ == "__main__":
    main()