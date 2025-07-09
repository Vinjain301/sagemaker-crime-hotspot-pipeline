"""
Model Training Module
Handles training of various models for crime hotspot prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
import yaml

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training of multiple model types"""
    
    def __init__(self, model_config_path: str):
        """
        Initialize ModelTrainer with configuration
        
        Args:
            model_config_path: Path to model configuration YAML
        """
        with open(model_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.trained_models = {}
        
    def train_model(self,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   model_type: str,
                   use_smote: bool = False,
                   tune_hyperparameters: bool = False) -> Any:
        """
        Train a specific model type
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model to train
            use_smote: Whether to apply SMOTE
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} model")
        
        # Apply SMOTE if requested
        if use_smote:
            X_train, y_train = self._apply_smote(X_train, y_train)
        
        # Get model
        if tune_hyperparameters:
            model = self._train_with_tuning(X_train, y_train, model_type)
        else:
            model = self._train_baseline(X_train, y_train, model_type)
        
        # Store trained model
        self.trained_models[model_type] = model
        
        logger.info(f"Successfully trained {model_type} model")
        return model
    
    def _apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE for class balancing"""
        logger.info("Applying SMOTE for class balancing")
        
        smote_config = self.config.get('smote', {})
        smote = SMOTE(
            random_state=smote_config.get('random_state', 42),
            sampling_strategy=smote_config.get('sampling_strategy', 'auto')
        )
        
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        logger.info(f"SMOTE complete. Original size: {len(X)}, Balanced size: {len(X_balanced)}")
        return X_balanced, y_balanced
    
    def _train_baseline(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> Any:
        """Train model with baseline parameters"""
        model_configs = self.config['models']
        
        if model_type == 'random_forest':
            params = model_configs['random_forest']['baseline']
            model = RandomForestClassifier(**params)
            
        elif model_type == 'xgboost':
            params = model_configs['xgboost']['baseline']
            model = XGBClassifier(**params)
            
        elif model_type == 'catboost':
            params = model_configs['catboost']['baseline']
            model = CatBoostClassifier(**params, verbose=False)
            
        elif model_type == 'catboost_weighted':
            params = model_configs['catboost']['baseline'].copy()
            params.update(model_configs['catboost']['with_weights'])
            model = CatBoostClassifier(**params, verbose=False)
            
        elif model_type == 'extra_trees':
            params = model_configs['extra_trees']
            model = ExtraTreesClassifier(**params)
            
        elif model_type == 'balanced_random_forest':
            params = model_configs['balanced_random_forest']
            model = BalancedRandomForestClassifier(**params)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X, y)
        return model
    
    def _train_with_tuning(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> Any:
        """Train model with hyperparameter tuning"""
        model_configs = self.config['models']
        
        if model_type == 'random_forest':
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_dist = model_configs['random_forest']['hyperparameter_ranges']
            
        elif model_type == 'xgboost':
            base_model = XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            param_dist = model_configs['xgboost']['hyperparameter_ranges']
            
        else:
            raise ValueError(f"Hyperparameter tuning not configured for {model_type}")
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring='f1',
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info(f"Starting hyperparameter search for {model_type}")
        random_search.fit(X, y)
        
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def save_model(self, model_name: str, output_path: str) -> bool:
        """
        Save trained model to disk
        
        Args:
            model_name: Name of the model to save
            output_path: Path to save the model
            
        Returns:
            bool: Success status
        """
        if model_name not in self.trained_models:
            logger.error(f"Model {model_name} not found in trained models")
            return False
        
        try:
            model = self.trained_models[model_name]
            joblib.dump(model, output_path)
            logger.info(f"Saved model to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path: str) -> Optional[Any]:
        """
        Load model from disk
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model or None if error
        """
        try:
            model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def get_available_models(self) -> list:
        """Get list of available model types"""
        return list(self.config['models'].keys()) + ['catboost_weighted']