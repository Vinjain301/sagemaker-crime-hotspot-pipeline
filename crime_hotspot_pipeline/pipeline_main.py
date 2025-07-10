"""
Main Pipeline Orchestration
Builds and runs the SageMaker pipeline for crime hotspot prediction
"""
import os
import sys
import yaml
import argparse
from datetime import datetime
import logging
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model_metrics import ModelMetrics, MetricsSource
import sagemaker
import boto3
from sagemaker.workflow.functions import JsonGet

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger_utils import setup_logger

logger = setup_logger("pipeline_orchestrator")


class CrimeHotspotPipeline:
    """Main pipeline orchestrator for crime hotspot prediction"""
    
    def __init__(self, config_path: str, script_path: str = None):
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to main configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.script_path = script_path or "processing.py"
        
        # Initialize SageMaker session
        self.sagemaker_session = sagemaker.Session()
        self.pipeline_session = PipelineSession()
        
        # Get execution role
        self.role = self.config['pipeline'].get('role_arn') or sagemaker.get_execution_role()
        
        # S3 paths
        self.bucket = self.config['s3']['bucket']
        self.pipeline_name = self.config['pipeline']['name']
        
        # Initialize pipeline parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize pipeline parameters"""
        self.model_type = ParameterString(
            name="ModelType",
            default_value="random_forest"
        )
        
        self.use_smote = ParameterString(
            name="UseSmote",
            default_value="false"
        )
        
        self.train_test_split_ratio = ParameterFloat(
            name="TrainTestSplitRatio",
            default_value=0.8
        )
    
    def create_processing_step(self) -> ProcessingStep:
        """Create data processing step"""
        
        # Create processor
        processor = SKLearnProcessor(
            framework_version="1.0-1",
            role=self.role,
            instance_type=self.config['processing']['instance_type'],
            instance_count=self.config['processing']['instance_count'],
            sagemaker_session=self.pipeline_session,)
        
        # Define inputs
        inputs = [

            ProcessingInput(
                source="data/",
                destination="/opt/ml/processing/input/code/data/"),
            
            ProcessingInput(
                source="features/",
                destination="/opt/ml/processing/input/code/features/"),
            
            ProcessingInput(
                source="config/",
                destination="/opt/ml/processing/input/config"),
            
            ProcessingInput(
                source=f"s3://{self.bucket}/{self.config['s3']['raw_data_prefix']}/",
                destination="/opt/ml/processing/input/raw"),
            
            ProcessingInput(
                source=f"s3://{self.bucket}/weather_data_final.csv",
                destination="/opt/ml/processing/input/weather"),
            
            ProcessingInput(
                source="requirements.txt",  # path to local file
                destination="/opt/ml/processing/input/requirements")
        ]
        
        # Define outputs
        outputs = [
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/output/train"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/output/test"
            ),
            ProcessingOutput(
                output_name="features",
                source="/opt/ml/processing/output/features"
            )
        ]
        print("DEBUG: Local contents of config/")
        for f in os.listdir("config"):
            print(" -", f)
        
        # Create processing step
        step_process = ProcessingStep(
            name="DataProcessingStep",
            processor=processor,
            inputs=inputs,
            outputs=outputs,
            code="processing.py",
            job_arguments=[
                "--train-test-split-ratio", self.train_test_split_ratio.to_string()
            ]
        )
        
        return step_process
    
    def create_training_step(self, processing_step: ProcessingStep) -> TrainingStep:
        """Create model training step"""
        
        # Create estimator
        estimator = SKLearn(
            entry_point="training.py",
            role=self.role,
            instance_type=self.config['training']['instance_type'],
            instance_count=self.config['training']['instance_count'],
            framework_version="1.0-1",
            source_dir=".",
            sagemaker_session=self.pipeline_session,
            hyperparameters={
                "model_type": self.model_type,
                "use_smote": self.use_smote
            }
        )
        
        # Create training step
        step_train = TrainingStep(
            name="ModelTrainingStep",
            estimator=estimator,
            inputs={
                "train": processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                "test": processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri
            }
        )
        
        return step_train
    
    def create_evaluation_step(self, training_step: TrainingStep) -> ProcessingStep:
        """Create model evaluation step"""
        
        processor = SKLearnProcessor(
            framework_version="1.0-1",
            role=self.role,
            instance_type=self.config['processing']['instance_type'],
            instance_count=self.config['processing']['instance_count'],
            sagemaker_session=self.pipeline_session,
        )
        
        step_eval = ProcessingStep(
            name="ModelEvaluationStep",
            processor=processor,
            inputs=[
                ProcessingInput(
                    source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model"
                ),
                ProcessingInput(
                    source=self.processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation"
                )
            ],
            code="evaluation.py",
            property_files=[
                PropertyFile(
                    name="EvaluationMetrics",
                    output_name="evaluation",
                    path="metrics.json"
                )
            ]
        )
        
        return step_eval
    
    def create_conditional_step(self, evaluation_step: ProcessingStep, training_step: TrainingStep) -> ConditionStep:
        """create conditional step based on model performance"""
        # Get F1 threshold from config
        f1_threshold = self.config.get('model_registry', {}).get('f1_threshold', 0.7)
        
        # Define the condition - register model only if F1 score >= threshold
        # THIS IS WHERE THE F1 THRESHOLD IS DEFINED!
        cond_gte = ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name="ModelEvaluationStep",property_file="EvaluationMetrics", json_path= "f1_score"),
            right=f1_threshold  # <-- F1 THRESHOLD FROM CONFIG (default 0.7)
        )
        
        logger.info(f"Conditional registration set with F1 threshold: {f1_threshold}")
        
        # Create registration step
        register_step = self.create_registration_step(training_step, evaluation_step)
        
        # Create condition step
        step_cond = ConditionStep(
            name="CheckModelQuality",
            conditions=[cond_gte],
            if_steps=[register_step],  # Register if F1 >= threshold
            else_steps=[]  # Do nothing if F1 < threshold
        )
        
        return step_cond
    

    
    def create_registration_step(self, training_step: TrainingStep, evaluation_step: ProcessingStep) -> RegisterModel:
        """Create model registration step"""
        
        model = SKLearnModel(
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            role=self.role,
            sagemaker_session=self.pipeline_session,
            framework_version="1.0-1",
            entry_point="inference.py"
        )
        
        # Get metrics from evaluation
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=Join(on="/",values=[evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,"evaluation.json"]),
                content_type="application/json"
            )
        )
        
        step_register = RegisterModel(
            name="RegisterModel",
            model=model,
            content_types=["application/json", "text/csv"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge", "ml.m5.2xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name="crime-hotspot-models",
            approval_status="PendingManualApproval",
            model_metrics=model_metrics
        )
        
        return step_register
    
    def create_pipeline(self) -> Pipeline:
        """Create the complete pipeline"""
        
        # Create steps
        self.processing_step = self.create_processing_step()  # Store as instance variable
        training_step = self.create_training_step(self.processing_step)
        evaluation_step = self.create_evaluation_step(training_step)
        conditional_step = self.create_conditional_step(evaluation_step, training_step)
        
        # Create pipeline
        pipeline = Pipeline(
            name=self.pipeline_name,
            parameters=[
                self.model_type,
                self.use_smote,
                self.train_test_split_ratio
            ],
            steps=[self.processing_step, training_step, evaluation_step, conditional_step],
            sagemaker_session=self.pipeline_session
        )
        
        return pipeline
    
    def run_pipeline(self, wait: bool = True) -> dict:
        """
        Run the pipeline
        
        Args:
            wait: Whether to wait for completion
            
        Returns:
            Execution response
        """
        pipeline = self.create_pipeline()
        
        # Create/update pipeline
        pipeline.upsert(role_arn=self.role)
        
        # Start execution
        execution = pipeline.start()
        
        logger.info(f"Pipeline execution started: {execution.arn}")
        
        if wait:
            logger.info("Waiting for pipeline completion...")
            execution.wait()
            
        return execution.describe()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Crime Hotspot ML Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost", "catboost", "extra_trees"],
        help="Model type to train"
    )
    parser.add_argument(
        "--use-smote",
        action="store_true",
        help="Use SMOTE for class balancing"
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for pipeline completion"
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = CrimeHotspotPipeline(args.config)
    
    # Override parameters if provided
    if args.model_type:
        pipeline.model_type.default_value = args.model_type
    if args.use_smote:
        pipeline.use_smote.default_value = "true"
    
    # Run pipeline
    result = pipeline.run_pipeline(wait=not args.no_wait)
    
    logger.info("Pipeline execution completed")
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()