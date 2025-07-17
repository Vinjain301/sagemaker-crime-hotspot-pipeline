# Crime Hotspot Prediction Pipeline - AWS SageMaker MLOps Implementation

An enterprise-grade ML pipeline for predicting crime hotspots in Los Angeles using AWS SageMaker, featuring modular architecture, automated model evaluation, and conditional deployment.

## Key Learnings & Technical Deep Dive

### Pipeline Architecture & Modularization
Through this project, I gained deep expertise in building production-ready ML pipelines:

- **Code Modularization**: Transformed a monolithic 1000+ line notebook into a clean, modular architecture with separate concerns for data processing, feature engineering, model training, and evaluation. Each module follows single responsibility principle with proper error handling and logging.

- **SageMaker Pipeline Steps**: Mastered the orchestration of different SageMaker components:
  - `ProcessingStep` for data preprocessing and feature engineering
  - `TrainingStep` for model training with various algorithms
  - `ConditionStep` for implementing conditional logic based on model performance
  - `RegisterModel` for automatic model registry integration

- **Inter-Step Data Flow**: Learned how to pass outputs between pipeline steps using:
  ```python
  # Using step properties to chain outputs
  training_step.properties.ModelArtifacts.S3ModelArtifacts
  evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri
  ```

### Advanced SageMaker Concepts

- **Parameterized Pipelines**: Implemented dynamic pipelines using `ParameterString` and `ParameterFloat` to enable:
  - Model type selection at runtime
  - Configurable train/test split ratios
  - Toggle for SMOTE oversampling
  - Dynamic F1 threshold for model registration

- **SDK vs Processors Understanding**:
  - `SKLearnProcessor`: Runs containerized processing jobs with automatic dependency management
  - `ScriptProcessor`: More flexible, allows custom container images
  - `Framework Processors`: Pre-built for specific ML frameworks with optimizations

- **Model Card Implementation**: Created dynamic model cards that capture:
  - Actual evaluation metrics (not placeholders)
  - Business context and intended use
  - Risk assessments based on model performance
  - Complete audit trail for model governance

### Debugging & Troubleshooting Experience

- **Module Import Issues**: Resolved Python path problems in distributed SageMaker environments by understanding how code is packaged and deployed to processing/training containers
- **Pipeline Property References**: Debugged `TypeError` with pipeline variables by learning to use `.to_string()` and `Join` functions for runtime resolution
- **Conditional Step Logic**: Implemented proper JSON path extraction using `JsonGet` with correct step references

### Production MLOps Best Practices

- **Automated Quality Gates**: F1 score threshold (0.7) prevents low-quality models from reaching production
- **Versioning Strategy**: Every pipeline run creates unique artifacts without overwriting previous results
- **Cost Optimization**: Configured appropriate instance types (ml.t3.medium for testing, ml.m5.xlarge for production)
- **Monitoring & Observability**: Integrated CloudWatch logging and Model Registry metrics

## Pipeline Overview

![Pipeline Execution Success](pipeline_execution_success.png)

The pipeline successfully processes ~800,000 crime records through multiple stages, achieving 94% accuracy with automated quality checks.

## Project Structure

```
crime_hotspot_pipeline/
├── config/                      # Configuration files
│   ├── config.yaml             # Main pipeline configuration
│   ├── feature_config.yaml     # Feature engineering settings
│   └── model_config.yaml       # Model hyperparameters
├── data/                       # Data handling modules
│   ├── __init__.py
│   ├── loader.py              # S3 data loading utilities
│   ├── cleaner.py             # Data cleaning and preprocessing
│   └── splitter.py            # Time-aware train/test splitting
├── features/                   # Feature engineering modules
│   ├── __init__.py
│   ├── engineer.py            # Main feature orchestrator
│   ├── temporal_features.py   # Time-based features
│   ├── spatial_features.py    # Location-based features
│   ├── crime_features.py      # Crime-specific features
│   ├── lag_features.py        # Temporal lag features
│   └── weather_features.py    # Weather data integration
├── model/                      # Model training and evaluation
│   ├── __init__.py
│   ├── train.py               # Multi-model training logic
│   ├── evaluate.py            # Evaluation metrics
│   ├── register.py            # Model registry integration
│   └── ensemble.py            # Ensemble methods
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── s3.py                  # S3 operations wrapper
│   ├── logger.py              # Centralized logging
│   ├── metrics.py             # Custom metrics
│   └── artifacts.py           # Artifact versioning
├── __init__.py                # Package init
├── create_model_card.py       # Dynamic model card generation
├── eda_module.py              # Exploratory data analysis
├── evaluation.py              # SageMaker evaluation script
├── inference.py               # Model inference endpoint
├── pipeline_main.py           # Main pipeline orchestration
├── processing.py              # SageMaker processing script
├── training.py                # SageMaker training script
├── requirements.txt           # Python dependencies
└── setup.py                   # Package setup
```

## Data Flow Architecture

### 1. **Raw Data Ingestion**
- Crime data (800K+ records) → S3 bucket
- Weather data → S3 bucket  
- Shapefile boundaries → S3 bucket

### 2. **Processing Pipeline**
```
S3 Raw Data
    ↓
ProcessingStep (processing.py)
    ├── DataLoader.load_csv_from_s3()
    ├── DataCleaner.clean_crime_data()
    ├── FeatureEngineer.engineer_features()
    │   ├── Temporal features (cyclical encoding)
    │   ├── Spatial features (crime rates per 1000)
    │   ├── Lag features (previous hour/day/year)
    │   └── Weather features (temperature, wind)
    └── TimeAwareSplitter.split_temporal_data()
         ↓
    S3 Processed Data (train/test splits)
```

### 3. **Training Pipeline**
```
Processed Data
    ↓
TrainingStep (training.py)
    ├── Optional SMOTE oversampling
    ├── Model selection (RF/XGBoost/CatBoost)
    ├── Training with configured hyperparameters
    └── Initial evaluation
         ↓
    S3 Model Artifacts
```

### 4. **Evaluation & Conditional Deployment**
```
Model Artifacts + Test Data
    ↓
EvaluationStep (evaluation.py)
    ├── Calculate metrics (F1, precision, recall)
    └── Generate evaluation report
         ↓
    ConditionStep (F1 ≥ 0.7?)
         ├── Yes → CreateModelCard → RegisterModel
         └── No → Pipeline ends
```

## AI-Assisted Development

This project leveraged AI to accelerate development while maintaining high code quality. Here are some example prompts that demonstrate the intersection of domain expertise and AI productivity:

### Architectural Design
```
"Given a monolithic crime prediction notebook with 1000+ lines mixing data processing, 
feature engineering, and model training, design a modular SageMaker pipeline architecture 
following MLOps best practices. Consider temporal data dependencies, class imbalance 
(90/10 split), and the need for conditional model deployment based on F1 scores."
```

### Complex Feature Engineering
```
"Implement lag features for temporal crime data that preserve causality in a 
time-series split. Include: crime_rate_last_1h using shift(1), 
crime_rate_yesterday_same_hour using datetime alignment, and 
was_hotspot_last_year_same_day with proper null handling for the first year."
```

### Pipeline Debugging
```
"Debug this SageMaker pipeline error: 'TypeError: Pipeline variables do not support 
__str__ operation'. The error occurs when creating ModelMetrics with 
evaluation_step.properties.ProcessingOutputConfig. Explain the difference between 
pipeline definition time and execution time resolution."
```

### Production Optimization
```
"Optimize this RandomForest configuration for ml.t3.medium instances with 4GB RAM 
processing 100k temporal records. Maintain model quality while preventing OOM errors. 
Consider: n_estimators, max_depth, min_samples_split, and memory-efficient alternatives."
```

## Key Features

### Automated ML Pipeline
- **Data Processing**: Handles missing values, outliers, and temporal alignment
- **Feature Engineering**: 50+ engineered features including cyclical time encoding
- **Model Training**: Supports multiple algorithms with hyperparameter tuning
- **Conditional Deployment**: Only registers models meeting quality thresholds

### Advanced Capabilities
- **Time-Aware Splitting**: Preserves temporal order for valid evaluation
- **Hotspot Detection**: Z-score based anomaly detection with rolling statistics
- **Weather Integration**: External data fusion for improved predictions
- **Model Versioning**: Complete lineage tracking without overwriting

### Production-Ready Features
- **Cost Optimization**: Configurable instance types for different workloads
- **Error Handling**: Comprehensive logging and graceful failure recovery
- **Scalability**: Processes millions of records efficiently
- **Governance**: Model cards and audit trails for compliance

## Model Performance (Top 3)

| Model | Configuration | Accuracy | Precision | Recall | F1-Score |
|-------|--------------|----------|-----------|---------|----------|
| Random Forest | SMOTE + Tuned | 94% | 76% | 74% | 75% |
| XGBoost | Scale weights | 93% | 75% | 72% | 73% |
| CatBoost | Light weights + SMOTE | 94% | 77% | 73% | 75% |

## Running the Pipeline

### Prerequisites
1. AWS Account with SageMaker access
2. S3 bucket with data uploaded
3. IAM role with appropriate permissions

### Quick Start
```bash
# Configure AWS credentials
aws configure

# Update configuration
vim config/config.yaml  # Set your S3 bucket and role ARN

# Run pipeline
python pipeline_main.py \
  --config config/config.yaml \
  --model-type random_forest \
  --use-smote
```

### Monitoring
- View pipeline progress in SageMaker Studio
- Check CloudWatch logs for detailed execution logs
- Review Model Registry for registered models

## Lessons Learned

1. **Modular Design**: Breaking down complex notebooks into focused modules improves maintainability and testing
2. **Pipeline Parameters**: Making pipelines configurable enables rapid experimentation without code changes
3. **Conditional Logic**: Quality gates prevent bad models from reaching production
4. **Time Series Considerations**: Proper temporal splitting is crucial for valid model evaluation
5. **Cost Awareness**: Instance type selection significantly impacts both cost and execution time

## Future Enhancements

- Real-time inference endpoint with auto-scaling
- A/B testing framework for model comparison
- Drift detection and automated retraining
- Integration with police dispatch systems
- Mobile app for field officers

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Los Angeles Police Department for providing crime data
- AWS SageMaker team for excellent documentation
- Open source communities behind scikit-learn, XGBoost, and CatBoost
