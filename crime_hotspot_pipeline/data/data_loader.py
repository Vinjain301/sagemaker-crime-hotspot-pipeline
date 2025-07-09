"""
Data Loader Module
Handles loading raw data from S3 and initial data ingestion
"""
import pandas as pd
import geopandas as gpd
import boto3
import tempfile
import os
from typing import Optional, Dict, Any
import logging
from botocore.exceptions import NoCredentialsError, ClientError

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading from various sources"""
    
    def __init__(self, bucket_name: str):
        """
        Initialize DataLoader with S3 bucket name
        
        Args:
            bucket_name: Name of the S3 bucket
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        
    def load_csv_from_s3(self, file_key: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV file from S3
        
        Args:
            file_key: S3 key for the file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            s3_uri = f's3://{self.bucket_name}/{file_key}'
            logger.info(f"Loading CSV from {s3_uri}")
            
            df = pd.read_csv(s3_uri, **kwargs)
            logger.info(f"Successfully loaded {len(df)} rows from {file_key}")
            
            return df
            
        except FileNotFoundError:
            logger.error(f"File {file_key} not found in bucket {self.bucket_name}")
            raise
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"AWS Client Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading data: {e}")
            raise
    
    def load_parquet_from_s3(self, file_key: str, **kwargs) -> pd.DataFrame:
        """
        Load Parquet file from S3
        
        Args:
            file_key: S3 key for the file
            **kwargs: Additional arguments for pd.read_parquet
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            s3_uri = f's3://{self.bucket_name}/{file_key}'
            logger.info(f"Loading Parquet from {s3_uri}")
            
            df = pd.read_parquet(s3_uri, **kwargs)
            logger.info(f"Successfully loaded {len(df)} rows from {file_key}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading parquet file: {e}")
            raise
    
    def load_shapefile_from_s3(self, prefix: str, extensions: list = None) -> gpd.GeoDataFrame:
        """
        Load shapefile components from S3 and return GeoDataFrame
        
        Args:
            prefix: S3 prefix for shapefile components (without extension)
            extensions: List of shapefile extensions to download
            
        Returns:
            gpd.GeoDataFrame: Loaded geodataframe
        """
        if extensions is None:
            extensions = ['shp', 'shx', 'dbf', 'prj', 'cpg']
            
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Download all shapefile components
                for ext in extensions:
                    key = f"{prefix}.{ext}"
                    local_path = os.path.join(tmpdir, f"{os.path.basename(prefix)}.{ext}")
                    
                    logger.info(f"Downloading {key} to {local_path}")
                    self.s3_client.download_file(self.bucket_name, key, local_path)
                
                # Load shapefile
                shp_path = os.path.join(tmpdir, f"{os.path.basename(prefix)}.shp")
                geo_df = gpd.read_file(shp_path)
                logger.info(f"Successfully loaded shapefile with {len(geo_df)} features")
                
                return geo_df
                
        except Exception as e:
            logger.error(f"Error loading shapefile: {e}")
            raise
    
    def load_crime_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Load main crime dataset based on configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            pd.DataFrame: Crime data
        """
        file_key = f"{config['s3']['raw_data_prefix']}/{config['data']['raw_file']}"
        return self.load_csv_from_s3(file_key)
    
    def load_weather_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Load weather data based on configuration
        
        Args:
            config: Configuration dictionary
            
        Returns:
            pd.DataFrame: Weather data
        """
        file_key = config['data']['weather_file']
        return self.load_csv_from_s3(file_key)
    
    def load_district_boundaries(self, config: Dict[str, Any]) -> gpd.GeoDataFrame:
        """
        Load district boundary shapefile
        
        Args:
            config: Configuration dictionary
            
        Returns:
            gpd.GeoDataFrame: District boundaries
        """
        prefix = config['data']['shapefile_prefix']
        return self.load_shapefile_from_s3(prefix)


def load_all_data(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load all required datasets
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict containing all loaded datasets
    """
    loader = DataLoader(config['s3']['bucket'])
    
    data = {
        'crime': loader.load_crime_data(config),
        'weather': loader.load_weather_data(config),
        'districts': loader.load_district_boundaries(config)
    }
    
    logger.info("All datasets loaded successfully")
    return data