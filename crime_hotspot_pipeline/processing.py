"""
SageMaker Processing Job Script
Handles data loading, cleaning, feature engineering, and splitting
"""
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
    import yaml
    print("yaml imported")
    import pandas as pd
    print("pandas imported")
    import numpy as np
    print("numpy imported")
    import json
    print("json imported")
    import argparse
    print("argparse imported")
    import logging
    print("logging imported")
    from pathlib import Path
    print("pathlib imported")
except Exception as e:
    print(f"Import failed: {e}")
    raise

# Add parent directory to path for imports
sys.path.append('/opt/ml/code')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main processing function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.8)
    args = parser.parse_args()
    
    logger.info("Starting data processing")
    
    # Define paths
    input_path = Path('/opt/ml/processing/input')
    output_path = Path('/opt/ml/processing/output')
    
    # Create output directories
    (output_path / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'test').mkdir(parents=True, exist_ok=True)
    (output_path / 'features').mkdir(parents=True, exist_ok=True)
    
    try:
        # Load configuration
        with open("/opt/ml/processing/input/config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        with open('/opt/ml/processing/input/config/feature_config.yaml', 'r') as f:
            feature_config = yaml.safe_load(f)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        print("sys.path:", sys.path)
        print("cwd:", os.getcwd())
        print("__file__:", __file__)

        # Import modules
        from data.data_loader import DataLoader
        from data.data_cleaner import DataCleaner
        from data.data_splitter import TimeAwareSplitter
        from features.feature_engineer import FeatureEngineer
        
        # Load data
        logger.info("Loading data...")
        loader = DataLoader(config['s3']['bucket'])
        
        # Load crime data
        crime_file = list((input_path / 'raw').glob('*.csv'))[0]
        crime_df = pd.read_csv(crime_file)
        logger.info(f"Loaded crime data: {crime_df.shape}")
        # Convert date column for sorting
        crime_df['DATE OCC'] = pd.to_datetime(crime_df['DATE OCC'])
        
        # Load weather data
        weather_file = input_path / 'weather' / 'weather_data_final.csv'
        weather_df = pd.read_csv(weather_file) if weather_file.exists() else None
        logger.info(f"Loaded weather data: {weather_df.shape if weather_df is not None else 'None'}")
        
        # Clean data
        logger.info("Cleaning data...")
        cleaner = DataCleaner(feature_config)
        crime_df = cleaner.clean_crime_data(crime_df)
        
        if weather_df is not None:
            weather_df = cleaner.clean_weather_data(weather_df)
        # Take the most recent 100k records to preserve temporal continuity
        crime_df = crime_df.sort_values('DATE OCC').tail(100000)
        logger.info(f"Selected most recent 100k records for testing: {crime_df.shape}")
        logger.info(f"Date range: {crime_df['DATE OCC'].min()} to {crime_df['DATE OCC'].max()}")
        # Engineer features
        logger.info("Engineering features...")
        engineer = FeatureEngineer(feature_config)
        featured_df = engineer.engineer_features(crime_df, weather_df)
        
        # Split data
        logger.info("Splitting data...")
        splitter = TimeAwareSplitter()
        splits = splitter.split_temporal_data(
            featured_df,
            train_ratio=args.train_test_split_ratio,
            target_column='is_hourly_hotspot'
        )
        
        train_df = splits['train']
        test_df = splits['test']
        
        # Save processed data
        logger.info("Saving processed data...")
        train_df.to_parquet(output_path / 'train' / 'train.parquet', index=False)
        test_df.to_parquet(output_path / 'test' / 'test.parquet', index=False)
        
        # Save feature metadata
        feature_metadata = {
            'feature_columns': [col for col in train_df.columns if col != 'is_hourly_hotspot'],
            'target_column': 'is_hourly_hotspot',
            'train_shape': train_df.shape,
            'test_shape': test_df.shape,
            'class_distribution': {
                'train': train_df['is_hourly_hotspot'].value_counts().to_dict(),
                'test': test_df['is_hourly_hotspot'].value_counts().to_dict()
            }
        }
        
        with open(output_path / 'features' / 'feature_metadata.json', 'w') as f:
            json.dump(feature_metadata, f, indent=4)
        
        logger.info("Data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()