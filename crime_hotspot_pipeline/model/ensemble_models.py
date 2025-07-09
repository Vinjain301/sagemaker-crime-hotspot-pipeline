"""
Ensemble Models Module
Implements ensemble methods for improved predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

logger = logging.getLogger(__name__)


class EnsembleModels:
    """Handles ensemble model creation and training"""
    
    def __init__(self, base_models: Dict[str, Any]):
        """
        Initialize with base models
        
        Args:
            base_models: Dictionary of trained base models
        """
        self.base_models = base_models
        
    def create_voting_ensemble(self, 
                             voting: str = 'soft',
                             weights: List[float] = None) -> VotingClassifier:
        """
        Create voting classifier from base models
        
        Args:
            voting: 'hard' or 'soft' voting
            weights: Optional weights for each model
            
        Returns:
            VotingClassifier
        """
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
            n_jobs=-1
        )
        
        logger.info(f"Created voting ensemble with {len(estimators)} models")
        return ensemble
    
    def create_stacking_ensemble(self,
                               final_estimator=None,
                               cv: int = 5) -> StackingClassifier:
        """
        Create stacking classifier
        
        Args:
            final_estimator: Meta-learner (default: LogisticRegression)
            cv: Cross-validation folds
            
        Returns:
            StackingClassifier
        """
        if final_estimator is None:
            final_estimator = LogisticRegression(random_state=42)
        
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=cv,
            n_jobs=-1
        )
        
        logger.info(f"Created stacking ensemble with {len(estimators)} base models")
        return ensemble
    
    def train_ensemble(self,
                      ensemble_type: str,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      **kwargs) -> Any:
        """
        Train ensemble model
        
        Args:
            ensemble_type: 'voting' or 'stacking'
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional arguments for ensemble creation
            
        Returns:
            Trained ensemble model
        """
        if ensemble_type == 'voting':
            ensemble = self.create_voting_ensemble(**kwargs)
        elif ensemble_type == 'stacking':
            ensemble = self.create_stacking_ensemble(**kwargs)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        logger.info(f"Training {ensemble_type} ensemble...")
        ensemble.fit(X_train, y_train)
        
        return ensemble