"""
Weather Features Engineering Module
Handles weather data integration and feature creation
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class WeatherFeatureEngineer:
    """Integrates weather data with crime data"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with weather feature configuration
        
        Args:
            config: Weather features configuration
        """
        self.config = config
        self.enable_weather = config.get('enable', True)
        self.weather_features = config.get('features', ['Temp', 'Wspeed'])
        
    def merge_weather_data(self, crime_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge weather data with crime data
        
        Args:
            crime_df: Crime dataframe with temporal features
            weather_df: Weather dataframe
            
        Returns:
            pd.DataFrame: Merged dataframe with weather features
        """
        if not self.enable_weather:
            logger.info("Weather features disabled in configuration")
            return crime_df
            
        logger.info("Merging weather data with crime data")
        
        # Prepare weather data
        weather_df = self._prepare_weather_data(weather_df)
        
        # Prepare crime data for merge
        crime_df = self._prepare_crime_data_for_merge(crime_df)
        
        # Merge on area, date, and hour
        merge_keys = ['DR_NO']
        
        # Select only needed weather columns
        weather_cols = merge_keys + self.weather_features
        weather_subset = weather_df[weather_cols].drop_duplicates()
        
        # Perform merge
        merged_df = crime_df.merge(
            weather_subset,
            on=merge_keys,
            how='left'
        )
        
        # Handle missing weather values
        merged_df = self._handle_missing_weather(merged_df)
        
        # Drop duplicates based on DR_NO if it exists
        if 'DR_NO' in merged_df.columns:
            initial_rows = len(merged_df)
            merged_df = merged_df.drop_duplicates(subset=['DR_NO'])
            logger.info(f"Dropped {initial_rows - len(merged_df)} duplicate rows after weather merge")
        
        logger.info(f"Weather merge complete. Added features: {self.weather_features}")
        return merged_df
    
    def _prepare_weather_data(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare weather data for merging"""
        # Remove duplicates
        weather_df = weather_df.drop_duplicates()
        
        # Convert DATE OCC to datetime
        if 'DATE OCC' in weather_df.columns:
            weather_df['DATE OCC'] = pd.to_datetime(weather_df['DATE OCC'], errors='coerce')
        
        # Ensure numeric weather columns
        for col in self.weather_features:
            if col in weather_df.columns:
                weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
        
        logger.info(f"Prepared weather data with {len(weather_df)} records")
        return weather_df
    
    def _prepare_crime_data_for_merge(self, crime_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare crime data for weather merge"""
        # Ensure DATE OCC is datetime
        if 'DATE OCC' in crime_df.columns:
            crime_df['DATE OCC'] = pd.to_datetime(crime_df['DATE OCC'], errors='coerce')
        
        return crime_df
    
    def _handle_missing_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing weather values after merge"""
        for feature in self.weather_features:
            if feature in df.columns:
                # Fill with median for numeric features
                median_val = df[feature].median()
                if pd.notna(median_val):
                    df[feature] = df[feature].fillna(median_val)
                    logger.info(f"Filled missing {feature} values with median: {median_val:.2f}")
        
        return df
    
    def create_weather_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between weather and time
        
        Args:
            df: Dataframe with weather and temporal features
            
        Returns:
            pd.DataFrame: Dataframe with weather interaction features
        """
        # Temperature during night hours (8 PM - 6 AM)
        if 'Temp' in df.columns and 'Hour' in df.columns:
            df['Temp_Night'] = df['Temp'] * ((df['Hour'] >= 20) | (df['Hour'] <= 6)).astype(int)
        
        # High wind indicator
        if 'Wspeed' in df.columns:
            wind_threshold = df['Wspeed'].quantile(0.75)
            df['High_Wind'] = (df['Wspeed'] > wind_threshold).astype(int)
        
        # Temperature categories
        if 'Temp' in df.columns:
            df['Temp_Category'] = pd.cut(
                df['Temp'],
                bins=[-np.inf, 50, 70, 85, np.inf],
                labels=['Cold', 'Mild', 'Warm', 'Hot']
            )
        
        logger.info("Created weather interaction features")
        return df
    
    def get_weather_features_list(self) -> list:
        """Get list of all weather features"""
        base_features = self.weather_features.copy()
        
        # Add potential interaction features
        interaction_features = ['Temp_Night', 'High_Wind', 'Temp_Category']
        
        return base_features + interaction_features