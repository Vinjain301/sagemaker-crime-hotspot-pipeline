"""
S3 Utilities Module
Handles all S3 operations for the pipeline
"""
import boto3
import pandas as pd
import json
import pickle
import io
import os
from typing import Any, Dict, Optional
import logging
from botocore.exceptions import NoCredentialsError, ClientError

logger = logging.getLogger(__name__)


class S3Handler:
    """Handles all S3 operations"""
    
    def __init__(self, bucket_name: str):
        """
        Initialize S3Handler
        
        Args:
            bucket_name: Name of the S3 bucket
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')
        
    def upload_dataframe(self, df: pd.DataFrame, key: str, format: str = 'csv') -> bool:
        """
        Upload dataframe to S3
        
        Args:
            df: Dataframe to upload
            key: S3 key (path)
            format: Format to save ('csv', 'parquet', 'json')
            
        Returns:
            bool: Success status
        """
        try:
            buffer = io.BytesIO() if format == 'parquet' else io.StringIO()
            
            if format == 'csv':
                df.to_csv(buffer, index=False)
            elif format == 'parquet':
                df.to_parquet(buffer, index=False)
            elif format == 'json':
                df.to_json(buffer, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            buffer.seek(0)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue()
            )
            
            logger.info(f"Successfully uploaded dataframe to s3://{self.bucket_name}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading dataframe: {e}")
            return False
    
    def download_dataframe(self, key: str, format: str = 'csv') -> Optional[pd.DataFrame]:
        """
        Download dataframe from S3
        
        Args:
            key: S3 key (path)
            format: Format to read ('csv', 'parquet', 'json')
            
        Returns:
            pd.DataFrame or None if error
        """
        try:
            s3_uri = f's3://{self.bucket_name}/{key}'
            
            if format == 'csv':
                df = pd.read_csv(s3_uri)
            elif format == 'parquet':
                df = pd.read_parquet(s3_uri)
            elif format == 'json':
                df = pd.read_json(s3_uri)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Successfully downloaded dataframe from {s3_uri}")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading dataframe: {e}")
            return None
    
    def upload_json(self, data: Dict[str, Any], key: str) -> bool:
        """
        Upload JSON data to S3
        
        Args:
            data: Dictionary to save as JSON
            key: S3 key (path)
            
        Returns:
            bool: Success status
        """
        try:
            json_string = json.dumps(data, indent=4)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json_string
            )
            logger.info(f"Successfully uploaded JSON to s3://{self.bucket_name}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading JSON: {e}")
            return False
    
    def download_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Download JSON data from S3
        
        Args:
            key: S3 key (path)
            
        Returns:
            Dict or None if error
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            logger.info(f"Successfully downloaded JSON from s3://{self.bucket_name}/{key}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading JSON: {e}")
            return None
    
    def upload_pickle(self, obj: Any, key: str) -> bool:
        """
        Upload pickled object to S3
        
        Args:
            obj: Object to pickle and upload
            key: S3 key (path)
            
        Returns:
            bool: Success status
        """
        try:
            buffer = io.BytesIO()
            pickle.dump(obj, buffer)
            buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue()
            )
            
            logger.info(f"Successfully uploaded pickle to s3://{self.bucket_name}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading pickle: {e}")
            return False
    
    def download_pickle(self, key: str) -> Optional[Any]:
        """
        Download pickled object from S3
        
        Args:
            key: S3 key (path)
            
        Returns:
            Unpickled object or None if error
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            obj = pickle.loads(response['Body'].read())
            logger.info(f"Successfully downloaded pickle from s3://{self.bucket_name}/{key}")
            return obj
            
        except Exception as e:
            logger.error(f"Error downloading pickle: {e}")
            return None
    
    def upload_file(self, local_path: str, key: str) -> bool:
        """
        Upload local file to S3
        
        Args:
            local_path: Path to local file
            key: S3 key (path)
            
        Returns:
            bool: Success status
        """
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, key)
            logger.info(f"Successfully uploaded file to s3://{self.bucket_name}/{key}")
            return True
            
        except FileNotFoundError:
            logger.error(f"File not found: {local_path}")
            return False
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return False
    
    def download_file(self, key: str, local_path: str) -> bool:
        """
        Download file from S3 to local path
        
        Args:
            key: S3 key (path)
            local_path: Local path to save file
            
        Returns:
            bool: Success status
        """
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, key, local_path)
            logger.info(f"Successfully downloaded file to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return False
    
    def list_objects(self, prefix: str = "") -> list:
        """
        List objects in S3 bucket with given prefix
        
        Args:
            prefix: S3 prefix to filter objects
            
        Returns:
            List of object keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error listing objects: {e}")
            return []
    
    def object_exists(self, key: str) -> bool:
        """
        Check if object exists in S3
        
        Args:
            key: S3 key (path)
            
        Returns:
            bool: True if exists
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                logger.error(f"Error checking object existence: {e}")
                return False