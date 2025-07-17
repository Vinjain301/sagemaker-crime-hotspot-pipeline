import sys
import os
import subprocess
print("Step 1: Starting script")

# Install requirements.txt before importing anything
requirements_path = "/opt/ml/processing/input/requirements/requirements.txt"

try:
    if os.path.exists(requirements_path):
        print("Step 2: Installing packages from requirements.txt")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_path
        ])
        print("Step 3: Requirements installed successfully")
    else:
        print("requirements.txt not found at expected path!")
except Exception as e:
    print(f"Failed to install requirements: {e}")
    raise

# Now do imports
print("Step 4: Importing modules")

try:
    import argparse
    import logging
    import json
    from pathlib import Path
    import boto3
    from sagemaker.model_card import (
        ModelCard,
        ModelCardStatusEnum,
        ModelOverview,
        IntendedUses,
        TrainingDetails,
        EvaluationJob,
        MetricGroup,
        Metric,
        AdditionalInformation
    )
    from sagemaker.session import Session
    from botocore.exceptions import ClientError
except Exception as e:
    print(f"Import failed: {e}")
    raise


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics-path', type=str, default='/opt/ml/processing/input/metrics')
    parser.add_argument('--output-path', type=str, default='/opt/ml/processing/output')
    parser.add_argument('--model-type', type=str, required=True)
    parser.add_argument('--use-smote', type=str, required=True)
    parser.add_argument('--model-card-name', type=str, default="crime-hotspot-modelcard",  help="Name of the model card to create or update")
    args = parser.parse_args()

    # Load metrics
    metrics_file = os.path.join(args.metrics_path, 'metrics.json')
    with open(metrics_file, 'r') as f:
        actual_metrics = json.load(f)

    evaluation_file = os.path.join(args.metrics_path, 'evaluation_report.json')
    with open(evaluation_file, 'r') as f:
        evaluation_report = json.load(f)

    # Instantiate Session
    boto_sess = boto3.Session(region_name=os.environ.get("AWS_REGION", "us-east-1"))
    sm_session = Session(boto_session=boto_sess)

    # Compose Model Card fields
    model_overview = ModelOverview(
        model_creator="Crime Hotspot ML Team",
        model_description="Predicts crime hotspots in Los Angeles districts on an hourly basis.",
        model_artifact=["s3://bucket/model"],  # to be updated dynamically in pipeline
    )

    intended_uses = IntendedUses(
        purpose_of_model="Predict crime hotspots in Los Angeles districts for the next hour",
        intended_uses="Support law enforcement resource allocation and patrol planning",
        factors_affecting_model_efficiency="Weather conditions, temporal patterns, and historical crime data",
        risk_rating="High" if actual_metrics['f1_score'] < 0.6 else "Medium",
        explanations_for_risk_rating=f"Model F1-score is {actual_metrics['f1_score']:.3f}"
    )

    evaluation_job = EvaluationJob(
        name="test_set_evaluation",
        evaluation_observation=f"Test set size: {evaluation_report.get('test_size', 'Unknown')}",
        datasets=["Temporal holdout test set"],
        metric_groups=[
            MetricGroup(
                name="classification_metrics",
                metric_data=[
                    Metric(name="Accuracy", type="number", value=actual_metrics["accuracy"], notes="Overall accuracy on test set"),
                    Metric(name="Precision", type="number", value=actual_metrics["precision"], notes="Precision for hotspot class"),
                    Metric(name="Recall", type="number", value=actual_metrics["recall"], notes="Recall for hotspot class"),
                    Metric(name="F1-Score", type="number", value=actual_metrics["f1_score"], notes="Primary metric for model selection"),
                    Metric(name="ROC-AUC", type="number", value=actual_metrics.get("roc_auc", 0.0), notes="Area under ROC curve"),
                ]
            )
        ]
    )

    additional_info = AdditionalInformation(
        ethical_considerations="Model should not be used as sole decision factor for resource allocation",
        caveats_and_recommendations=f"Model achieved {actual_metrics['f1_score']:.1%} F1-score. Monitor for drift in production.",
        custom_details={
            "positive_class_ratio": str(evaluation_report.get("positive_class_ratio", "Unknown")),
            "feature_count": str(evaluation_report.get("feature_count", "Unknown"))
        }
    )

    # Create ModelCard object
    model_card = ModelCard(
        name=args.model_card_name,
        status=ModelCardStatusEnum.DRAFT,
        model_overview=model_overview,
        intended_uses=intended_uses,
        evaluation_details=[evaluation_job],
        additional_information=additional_info,
        sagemaker_session=sm_session
    )

    # Try to load existing ModelCard
    try:
        existing_card = ModelCard.load(name=args.model_card_name, sagemaker_session=sm_session)
    # If load succeeds, update the existing card
        logger.info(f"Model card '{args.model_card_name}' already exists. Updating...")
        existing_card.model_overview = model_card.model_overview
        existing_card.intended_uses = model_card.intended_uses
        existing_card.evaluation_details = [evaluation_job]
        existing_card.additional_information = model_card.additional_information
        existing_card.status = model_card.status
        existing_card.update()
        logger.info(f"Model card '{args.model_card_name}' updated in SageMaker.")

    except ClientError as e:
        if "ResourceNotFound" in str(e):
            # If it does not exist, create it
            logger.info(f"Model card '{args.model_card_name}' does not exist. Creating...")
            model_card.create()
            logger.info(f"Model card '{args.model_card_name}' created in SageMaker.")
        else:
            # Any other error is re-raised
            raise


if __name__ == "__main__":
    main()
