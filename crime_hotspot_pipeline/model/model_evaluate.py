"""
Model Evaluation Module
Handles evaluation metrics and reporting for trained models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.inspection import permutation_importance
import json

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles model evaluation and metric calculation"""
    
    def __init__(self):
        """Initialize ModelEvaluator"""
        self.evaluation_results = {}
        
    def evaluate_model(self,
                      model: Any,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      model_name: str,
                      calculate_importance: bool = True) -> Dict[str, Any]:
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            calculate_importance: Whether to calculate feature importance
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = None
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        # Add confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # Add classification report
        metrics['classification_report'] = classification_report(
            y_test, y_pred, output_dict=True
        )
        
        # Calculate feature importance if requested
        if calculate_importance:
            metrics['feature_importance'] = self._calculate_feature_importance(
                model, X_test, y_test
            )
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        logger.info(f"Evaluation complete for {model_name}")
        return metrics
    
    def _calculate_metrics(self, 
                          y_true: pd.Series, 
                          y_pred: np.ndarray,
                          y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate standard evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Add AUC if probabilities available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                logger.warning("Could not calculate ROC AUC")
                metrics['roc_auc'] = None
        
        return metrics
    
    def _calculate_feature_importance(self,
                                    model: Any,
                                    X_test: pd.DataFrame,
                                    y_test: pd.Series) -> Dict[str, Any]:
        """Calculate feature importance"""
        importance_dict = {}
        
        # Model-based importance
        if hasattr(model, 'feature_importances_'):
            importance_dict['model_based'] = dict(
                zip(X_test.columns, model.feature_importances_)
            )
        
        # Permutation importance (sample if dataset is large)
        try:
            if len(X_test) > 5000:
                X_sampled = X_test.sample(500, random_state=42)
                y_sampled = y_test.loc[X_sampled.index]
            else:
                X_sampled = X_test
                y_sampled = y_test
            
            perm_importance = permutation_importance(
                model, X_sampled, y_sampled,
                n_repeats=3, random_state=42, n_jobs=-1
            )
            
            importance_dict['permutation'] = dict(
                zip(X_test.columns, perm_importance.importances_mean)
            )
            
        except Exception as e:
            logger.warning(f"Could not calculate permutation importance: {e}")
        
        return importance_dict
    
    def plot_confusion_matrix(self,
                            model_name: str,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix for a model
        
        Args:
            model_name: Name of the model
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results for {model_name}")
        
        cm = np.array(self.evaluation_results[model_name]['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Hotspot', 'Hotspot'],
                   yticklabels=['Not Hotspot', 'Hotspot'],
                   ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_name}')
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved confusion matrix plot to {save_path}")
        
        return fig
    
    def plot_feature_importance(self,
                              model_name: str,
                              importance_type: str = 'model_based',
                              top_n: int = 20,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance
        
        Args:
            model_name: Name of the model
            importance_type: Type of importance ('model_based' or 'permutation')
            top_n: Number of top features to show
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results for {model_name}")
        
        importance_data = self.evaluation_results[model_name].get('feature_importance', {})
        
        if importance_type not in importance_data:
            raise ValueError(f"No {importance_type} importance for {model_name}")
        
        # Get importance values
        importance_dict = importance_data[importance_type]
        importance_df = pd.DataFrame.from_dict(
            importance_dict, orient='index', columns=['importance']
        ).sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        importance_df.sort_values('importance').plot(kind='barh', ax=ax)
        ax.set_xlabel('Importance')
        ax.set_title(f'{importance_type.title()} Feature Importance - {model_name}')
        ax.legend().remove()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        return fig
    
    def compare_models(self, model_names: Optional[list] = None) -> pd.DataFrame:
        """
        Compare metrics across models
        
        Args:
            model_names: List of model names to compare (None for all)
            
        Returns:
            DataFrame with comparison metrics
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        comparison_data = []
        
        for model_name in model_names:
            if model_name in self.evaluation_results:
                metrics = self.evaluation_results[model_name]
                row = {
                    'model': model_name,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'roc_auc': metrics.get('roc_auc', np.nan)
                }
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('f1_score', ascending=False)
    
    def save_evaluation_report(self, output_path: str) -> bool:
        """
        Save evaluation results to JSON file
        
        Args:
            output_path: Path to save the report
            
        Returns:
            bool: Success status
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.evaluation_results, f, indent=4)
            logger.info(f"Saved evaluation report to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")
            return False