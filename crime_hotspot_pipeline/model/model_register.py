"""
Model Registry Module
Handles model registration to SageMaker Model Registry
"""
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.sklearn.model import SKLearnModel
from typing import Dict, Any, Optional
import logging
import json

logger = logging.getLogger(__name__)


class ModelRegistrar:
    """Handles model registration to SageMaker Model Registry"""
    
    def __init__(self, role: str, sagemaker_session=None):
        """
        Initialize ModelRegistrar
        
        Args:
            role: SageMaker execution role
            sagemaker_session: Optional SageMaker session
        """
        self.role = role
        self.sagemaker_session = sagemaker_session or sagemaker.Session()
        self.sm_client = boto3.client('sagemaker')
        
    def create_model_package_group(self, group_name: str, description: str = None) -> str:
        """
        Create model package group if it doesn't exist
        
        Args:
            group_name: Name of the model package group
            description: Optional description
            
        Returns:
            ARN of the model package group
        """
        try:
            response = self.sm_client.describe_model_package_group(
                ModelPackageGroupName=group_name
            )
            logger.info(f"Model package group {group_name} already exists")
            return response['ModelPackageGroupArn']
            
        except self.sm_client.exceptions.ValidationException:
            # Group doesn't exist, create it
            response = self.sm_client.create_model_package_group(
                ModelPackageGroupName=group_name,
                ModelPackageGroupDescription=description or f"Models for {group_name}"
            )
            logger.info(f"Created model package group: {group_name}")
            return response['ModelPackageGroupArn']
    
    def register_model(self,
                      model_data: str,
                      model_name: str,
                      model_metrics: Dict[str, Any],
                      framework_version: str = "1.0-1",
                      instance_types: list = None,
                      approval_status: str = "PendingManualApproval") -> str:
        """
        Register model to SageMaker Model Registry
        
        Args:
            model_data: S3 URI of model artifacts
            model_name: Name for the model
            model_metrics: Evaluation metrics
            framework_version: Framework version
            instance_types: List of supported instance types
            approval_status: Initial approval status
            
        Returns:
            Model package ARN
        """
        if instance_types is None:
            instance_types = ["ml.m5.xlarge", "ml.m5.2xlarge"]
        
        # Create model package group
        group_name = "crime-hotspot-models"
        self.create_model_package_group(group_name)
        
        # Prepare model metrics
        model_quality = self._format_model_metrics(model_metrics)
        
        # Create model package
        model_package_args = {
            "ModelPackageGroupName": group_name,
            "ModelPackageDescription": f"{model_name} for crime hotspot prediction",
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": sagemaker.image_uris.retrieve(
                            "sklearn",
                            self.sagemaker_session.boto_region_name,
                            version=framework_version
                        ),
                        "ModelDataUrl": model_data,
                        "Environment": {
                            "SAGEMAKER_PROGRAM": "inference.py",
                            "SAGEMAKER_SUBMIT_DIRECTORY": model_data
                        }
                    }
                ],
                "SupportedTransformInstanceTypes": instance_types,
                "SupportedRealtimeInferenceInstanceTypes": instance_types,
                "SupportedContentTypes": ["application/json", "text/csv"],
                "SupportedResponseMIMETypes": ["application/json", "text/csv"],
            },
            "ModelApprovalStatus": approval_status,
            "ModelMetrics": {
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "S3Uri": self._upload_metrics(model_quality, model_name)
                    }
                }
            }
        }
        
        response = self.sm_client.create_model_package(**model_package_args)
        model_package_arn = response["ModelPackageArn"]
        
        logger.info(f"Registered model {model_name} with ARN: {model_package_arn}")
        return model_package_arn
    
    def _format_model_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Format metrics for Model Registry"""
        formatted_metrics = {
            "classification_metrics": {
                "accuracy": {
                    "value": metrics.get("accuracy", 0),
                    "standard_deviation": "NaN"
                },
                "precision": {
                    "value": metrics.get("precision", 0),
                    "standard_deviation": "NaN"
                },
                "recall": {
                    "value": metrics.get("recall", 0),
                    "standard_deviation": "NaN"
                },
                "f1_score": {
                    "value": metrics.get("f1_score", 0),
                    "standard_deviation": "NaN"
                },
                "auc": {
                    "value": metrics.get("roc_auc", 0),
                    "standard_deviation": "NaN"
                }
            }
        }
        return formatted_metrics
    
    def _upload_metrics(self, metrics: Dict[str, Any], model_name: str) -> str:
        """Upload metrics to S3 and return URI"""
        # This would use your S3Handler to upload
        # For now, returning a placeholder
        metrics_key = f"model_registry/metrics/{model_name}_metrics.json"
        # self.s3_handler.upload_json(metrics, metrics_key)
        return f"s3://{self.sagemaker_session.default_bucket()}/{metrics_key}"
    
    def update_model_approval_status(self,
                                   model_package_arn: str,
                                   status: str,
                                   comment: str = "") -> None:
        """
        Update model approval status
        
        Args:
            model_package_arn: ARN of the model package
            status: New status (Approved/Rejected)
            comment: Optional comment
        """
        self.sm_client.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus=status,
            ApprovalDescription=comment
        )
        logger.info(f"Updated model status to {status}")
    
    def get_latest_approved_model(self, group_name: str) -> Optional[str]:
        """Get latest approved model from a group"""
        response = self.sm_client.list_model_packages(
            ModelPackageGroupName=group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1
        )
        
        if response["ModelPackageSummaryList"]:
            return response["ModelPackageSummaryList"][0]["ModelPackageArn"]
        return None