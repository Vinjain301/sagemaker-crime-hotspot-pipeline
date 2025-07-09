"""
SageMaker Inference Script
Handles model inference for deployed endpoints
"""
import joblib
import os
import json
import pandas as pd
import numpy as np


def model_fn(model_dir):
    """Load model for inference"""
    model_path = os.path.join(model_dir, 'model.joblib')
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return pd.DataFrame(input_data)
    elif request_content_type == 'text/csv':
        return pd.read_csv(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions"""
    # Ensure we have the right columns
    predictions = model.predict(input_data)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_data)
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
    else:
        return {'predictions': predictions.tolist()}


def output_fn(prediction, content_type):
    """Format output"""
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")