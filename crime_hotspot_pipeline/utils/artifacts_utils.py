"""
Artifacts Management Module
Handles versioning and storage of pipeline artifacts
"""
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages versioned artifacts from pipeline runs"""
    
    def __init__(self, s3_handler, base_prefix: str = "Crime_Hotspot_Prediction"):
        """
        Initialize ArtifactManager
        
        Args:
            s3_handler: S3Handler instance
            base_prefix: Base S3 prefix for artifacts
        """
        self.s3_handler = s3_handler
        self.base_prefix = base_prefix
        self.run_id = self._generate_run_id()
        
    def _generate_run_id(self) -> str:
        """Generate unique run ID with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"run_{timestamp}_{random_suffix}"
    
    def get_artifact_path(self, artifact_type: str, filename: str) -> str:
        """
        Get versioned S3 path for artifact
        
        Args:
            artifact_type: Type of artifact (model, evaluation, plot, etc.)
            filename: Name of the file
            
        Returns:
            S3 key with versioning
        """
        return f"{self.base_prefix}/runs/{self.run_id}/{artifact_type}/{filename}"
    
    def save_model(self, model: Any, model_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Save model with versioning
        
        Args:
            model: Trained model
            model_name: Name of the model
            metadata: Optional metadata
            
        Returns:
            S3 path where model was saved
        """
        # Save model
        model_key = self.get_artifact_path("models", f"{model_name}.pkl")
        self.s3_handler.upload_pickle(model, model_key)
        
        # Save metadata
        if metadata:
            metadata['run_id'] = self.run_id
            metadata['timestamp'] = datetime.now().isoformat()
            metadata['model_name'] = model_name
            
            metadata_key = self.get_artifact_path("models", f"{model_name}_metadata.json")
            self.s3_handler.upload_json(metadata, metadata_key)
        
        logger.info(f"Saved model to {model_key}")
        return model_key
    
    def save_evaluation_results(self, results: Dict[str, Any], model_name: str) -> str:
        """Save evaluation results with versioning"""
        key = self.get_artifact_path("evaluation", f"{model_name}_evaluation.json")
        
        # Add run metadata
        results['run_id'] = self.run_id
        results['timestamp'] = datetime.now().isoformat()
        
        self.s3_handler.upload_json(results, key)
        logger.info(f"Saved evaluation results to {key}")
        return key
    
    def save_plot(self, fig: Any, plot_name: str, subfolder: str = "") -> str:
        """Save plot with versioning"""
        if subfolder:
            key = self.get_artifact_path(f"plots/{subfolder}", f"{plot_name}.png")
        else:
            key = self.get_artifact_path("plots", f"{plot_name}.png")
        
        # Save to temporary file then upload
        temp_path = f"/tmp/{plot_name}.png"
        fig.savefig(temp_path, bbox_inches='tight', dpi=150)
        
        self.s3_handler.upload_file(temp_path, key)
        
        # Clean up
        os.remove(temp_path)
        logger.info(f"Saved plot to {key}")
        return key
    
    def save_run_summary(self, config: Dict[str, Any], results: Dict[str, Any]) -> str:
        """Save overall run summary"""
        summary = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'configuration': config,
            'results_summary': results
        }
        
        key = self.get_artifact_path("", "run_summary.json")
        self.s3_handler.upload_json(summary, key)
        
        # Also save to latest
        latest_key = f"{self.base_prefix}/latest/run_summary.json"
        self.s3_handler.upload_json(summary, latest_key)
        
        logger.info(f"Saved run summary to {key}")
        return key
    
    def list_runs(self) -> list:
        """List all previous runs"""
        prefix = f"{self.base_prefix}/runs/"
        objects = self.s3_handler.list_objects(prefix)
        
        # Extract unique run IDs
        run_ids = set()
        for obj in objects:
            parts = obj.split('/')
            if len(parts) > 2:
                run_ids.add(parts[2])
        
        return sorted(list(run_ids), reverse=True)