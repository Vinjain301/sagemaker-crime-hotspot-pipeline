"""
Main Feature Engineering Module
Orchestrates all feature engineering operations
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from .temporal_features import TemporalFeatureEngineer
from .spatial_features import SpatialFeatureEngineer
from .crime_features import CrimeFeatureEngineer
from .lag_features import LagFeatureEngineer
from .weather_features import WeatherFeatureEngineer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Main feature engineering orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureEngineer with configuration
        
        Args:
            config: Feature engineering configuration
        """
        self.config = config
        
        # Initialize sub-engineers
        self.temporal_engineer = TemporalFeatureEngineer(config.get('temporal_features', {}))
        self.spatial_engineer = SpatialFeatureEngineer(config.get('spatial_features', {}))
        self.crime_engineer = CrimeFeatureEngineer(config.get('crime_features', {}))
        self.lag_engineer = LagFeatureEngineer(config.get('lag_features', {}))
        self.weather_engineer = WeatherFeatureEngineer(config.get('weather_features', {}))
        
        self.features_to_drop = config.get('highly_correlated_features', [])
        
    def engineer_features(self, 
                         crime_df: pd.DataFrame, 
                         weather_df: Optional[pd.DataFrame] = None,
                         geo_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            crime_df: Crime dataframe
            weather_df: Weather dataframe (optional)
            geo_df: Geographic boundaries dataframe (optional)
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        logger.info("Starting feature engineering pipeline")
        df = crime_df.copy()
        
        # 1. Temporal Features
        logger.info("Engineering temporal features")
        df = self.temporal_engineer.create_temporal_features(df)
        
        # 2. Crime-related Features
        logger.info("Engineering crime features")
        df = self.crime_engineer.create_crime_features(df)
        
        # 3. Spatial Features (if population data available)
        logger.info("Engineering spatial features")
        df = self.spatial_engineer.create_spatial_features(df)
        
        # 4. Hotspot Detection Features
        logger.info("Creating hotspot features")
        df = self.crime_engineer.create_hotspot_features(df)
        
        # 5. Lag Features
        logger.info("Engineering lag features")
        df = self.lag_engineer.create_lag_features(df)
        
        # 6. Weather Features (if weather data provided)
        if weather_df is not None:
            logger.info("Merging weather features")
            df = self.weather_engineer.merge_weather_data(df, weather_df)
        
        # 7. Drop highly correlated features
        df = self._drop_correlated_features(df)
        
        # 8. Create final model dataset
        df = self._prepare_model_dataset(df)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df
    
    def _drop_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop highly correlated features"""
        features_to_remove = [col for col in self.features_to_drop if col in df.columns]
        
        if features_to_remove:
            logger.info(f"Dropping correlated features: {features_to_remove}")
            df = df.drop(columns=features_to_remove)
        
        return df
    
    def _prepare_model_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare final dataset for modeling"""
        # Select relevant columns for modeling
        selected_cols = self._get_selected_columns(df)
        
        # Filter columns that exist in dataframe
        existing_cols = [col for col in selected_cols if col in df.columns]
        
        # Create model dataset
        model_df = df[existing_cols].copy()
        
        # Fill any remaining NaN values
        model_df = model_df.fillna(0)
        
        # Drop duplicates
        initial_rows = len(model_df)
        model_df = model_df.drop_duplicates()
        logger.info(f"Dropped {initial_rows - len(model_df)} duplicate rows")
        
        return model_df
    
    def _get_selected_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of columns to include in final dataset"""
        selected_cols = [
            # Time features
            "Datetime_Key", "Hour_sin", "Hour_cos", "DayOfWeek_sin", "DayOfWeek_cos",
            "Month_sin", "Month_cos", "Year", "IsHoliday", "IsWeekendorHoliday", "Weekend",
            
            # Location
            "AREA NAME",
            
            # Crime rate indicators
            "Hourly_Crime_Rate_per_1000", "Daily_Crime_Rate_per_1000",
            "Monthly_Crime_Rate_per_1000", "Yearly_Crime_Rate_per_1000",
            
            # Lag features
            "crime_rate_last_1h", "crime_rate_last_3h_avg", "crime_rate_last_6h_sum",
            "crime_rate_yesterday_same_hour", "was_hourly_hotspot_last_hour",
            "was_daily_hotspot_yesterday", "was_hotspot_last_year_same_day",
            
            # Weather
            "Temp", "Wspeed",
            
            # Target
            "is_hourly_hotspot"
        ]
        
        return selected_cols
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groups for importance analysis
        
        Returns:
            Dictionary mapping feature group names to feature lists
        """
        return {
            'temporal': ['Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 
                        'Month_sin', 'Month_cos', 'Year'],
            'holiday': ['IsHoliday', 'IsWeekendorHoliday', 'Weekend'],
            'crime_rates': ['Hourly_Crime_Rate_per_1000', 'Daily_Crime_Rate_per_1000',
                           'Monthly_Crime_Rate_per_1000'],
            'lag_features': ['crime_rate_last_1h', 'crime_rate_last_3h_avg', 
                           'crime_rate_last_6h_sum', 'crime_rate_yesterday_same_hour'],
            'hotspot_history': ['was_hourly_hotspot_last_hour', 'was_daily_hotspot_yesterday',
                              'was_hotspot_last_year_same_day'],
            'weather': ['Temp', 'Wspeed']
        }