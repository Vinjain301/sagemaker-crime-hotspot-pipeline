# Crime Hotspot Prediction Pipeline

An AWS SageMaker ML Pipeline for predicting crime hotspots in Los Angeles using spatio-temporal analysis and machine learning.

## Overview

This project implements a modular, scalable machine learning pipeline that:
- Processes historical crime data from Los Angeles
- Engineers temporal, spatial, and weather-based features
- Trains multiple ML models (Random Forest, XGBoost, CatBoost, etc.)
- Predicts hourly crime hotspots at the district level
- Handles severe class imbalance (90% non-hotspot vs 10% hotspot)

## Project Structure

```
crime_hotspot_pipeline/
│
├── config/                      # Configuration files
│   ├── config.yaml             # Main pipeline configuration
│   ├── feature_config.yaml     # Feature engineering settings
│   └── model_config.yaml       # Model hyperparameters
│
├── data/                       # Data handling modules
│   ├── loader.py              # S3 data loading
│   ├── cleaner.py             # Data cleaning and preprocessing
│   └── splitter.py            # Time-aware train/test splitting
│
├── features/                   # Feature engineering modules
│   ├── engineer.py            # Main feature orchestrator
│   ├── temporal_features.py   # Time-based features
│   ├── spatial_features.py    # Location-based features
│   ├── crime_features.py      # Crime-specific features
│   ├── lag_features.py        # Temporal lag features
│   └── weather_features.py    # Weather data integration
│
├── model/                      # Model training and evaluation
│   ├── train.py               # Multi-model training
│   ├── evaluate.py            # Evaluation metrics
│   └── register.py            # Model registry integration
│
├── utils/                      # Utility modules
│   ├── s3.py                  # S3 operations
│   └── logger.py              # Logging configuration
│
├── pipeline/                   # SageMaker pipeline orchestration
│   └── run_pipeline.py        # Main pipeline runner
│
├── scripts/                    # SageMaker job scripts
│   ├── processing.py          # Processing job script
│   └── training.py            # Training job script
│
└── requirements.txt            # Python dependencies
```

## Features

### Data Processing
- Handles missing values intelligently based on feature type
- Removes outliers and corrects data quality issues
- Standardizes text fields and date formats
- Filters incomplete years from the dataset

### Feature Engineering
- **Temporal Features**: Hour/day/month cyclical encoding, holidays, weekends
- **Spatial Features**: Crime rates per 1000 population at various time scales
- **Crime Features**: Category mapping, severity classification
- **Lag Features**: Previous hour/day/year crime rates and hotspot indicators
- **Weather Features**: Temperature and wind speed integration
- **Hotspot Detection**: Z-score based anomaly detection with rolling statistics

### Models Supported
- Random Forest (with/without SMOTE)
- XGBoost
- CatBoost (with optional class weights)
- Extra Trees
- Balanced Random Forest

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crime-hotspot-pipeline.git
cd crime-hotspot-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Configuration

1. Update `config/config.yaml` with your AWS settings:
```yaml
s3:
  bucket: your-s3-bucket
pipeline:
  role_arn: your-sagemaker-role-arn
```

2. Adjust feature engineering settings in `config/feature_config.yaml`
3. Modify model hyperparameters in `config/model_config.yaml`

## Usage

### Running the Full Pipeline

```bash
python pipeline/run_pipeline.py \
  --config config/config.yaml \
  --model-type random_forest \
  --use-smote
```

### Running Individual Components

#### Data Processing Only
```python
from data.loader import DataLoader
from data.cleaner import DataCleaner
from features.engineer import FeatureEngineer

# Load and clean data
loader = DataLoader("your-bucket")
crime_df = loader.load_crime_data(config)
cleaner = DataCleaner(config)
cleaned_df = cleaner.clean_crime_data(crime_df)

# Engineer features
engineer = FeatureEngineer(feature_config)
featured_df = engineer.engineer_features(cleaned_df)
```

#### Model Training Only
```python
from model.train import ModelTrainer

trainer = ModelTrainer("config/model_config.yaml")
model = trainer.train_model(
    X_train, y_train,
    model_type="xgboost",
    use_smote=True
)
```

## Model Performance

Based on experiments with ~800,000 crime records:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Random Forest (SMOTE) | 94% | 76% | 74% | 75% |
| XGBoost (Tuned) | 93% | 75% | 72% | 73% |
| CatBoost (SMOTE) | 94% | 77% | 73% | 75% |

## Key Insights

1. **Temporal Patterns**: Crime hotspots show strong hourly and daily patterns
2. **Feature Importance**: Previous hour hotspot status is the strongest predictor
3. **Weather Impact**: Wind speed shows unexpected importance in predictions
4. **Class Imbalance**: SMOTE significantly improves minority class recall

## AWS SageMaker Deployment

The pipeline is designed for SageMaker deployment:

1. **Processing Jobs**: Handle data cleaning and feature engineering
2. **Training Jobs**: Train models with various algorithms
3. **Model Registry**: Track and version trained models
4. **Batch Transform**: Score new data for hotspot predictions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Los Angeles Police Department for providing the crime data
- AWS SageMaker team for the ML platform
- scikit-learn, XGBoost, and CatBoost communities
