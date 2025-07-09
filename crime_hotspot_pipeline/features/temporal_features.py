"""
Temporal Features Engineering Module
Handles time-based feature creation
"""
import pandas as pd
import numpy as np
import holidays
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """Creates temporal features from datetime columns"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with temporal feature configuration
        
        Args:
            config: Temporal features configuration
        """
        self.config = config
        self.enable_cyclical = config.get('enable_cyclical_encoding', True)
        self.cyclical_features = config.get('cyclical_features', ['Hour', 'DayOfWeek', 'Month'])
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all temporal features
        
        Args:
            df: Input dataframe with DATE OCC column
            
        Returns:
            pd.DataFrame: Dataframe with temporal features added
        """
        # Ensure DATE OCC is datetime
        df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
        df['Date'] = df['DATE OCC'].dt.date
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract basic temporal features
        df = self._extract_basic_features(df)
        
        # Create cyclical encodings
        if self.enable_cyclical:
            df = self._create_cyclical_features(df)
        
        # Create holiday features
        df = self._create_holiday_features(df)
        
        # Create weekend features
        df = self._create_weekend_features(df)
        
        # Create datetime key for joining
        df = self._create_datetime_keys(df)
        
        logger.info("Temporal features created successfully")
        return df
    
    def _extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic time components"""
        # Year, Month, Day
        df['Year'] = df['DATE OCC'].dt.year
        df['Month'] = df['DATE OCC'].dt.month
        df['Day'] = df['DATE OCC'].dt.day
        
        # Day of week (0 = Monday, 6 = Sunday)
        df['DayOfWeek'] = df['DATE OCC'].dt.dayofweek
        
        # Quarter
        df['Quarter'] = df['DATE OCC'].dt.quarter
        
        # Hour from TIME OCC
        if 'TIME OCC' in df.columns:
            # Ensure TIME OCC is 4 digits
            df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)
            df['Hour'] = df['TIME OCC'].str[:2].astype(int)
        
        logger.info("Basic temporal features extracted")
        return df
    
    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical encodings for temporal features"""
        
        if 'Hour' in self.cyclical_features and 'Hour' in df.columns:
            df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            logger.info("Created cyclical hour features")
        
        if 'DayOfWeek' in self.cyclical_features and 'DayOfWeek' in df.columns:
            df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
            logger.info("Created cyclical day of week features")
        
        if 'Month' in self.cyclical_features and 'Month' in df.columns:
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            logger.info("Created cyclical month features")
        
        return df
    
    def _create_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create holiday indicator features"""
        # Get unique years in dataset
        years = df['Year'].unique().tolist()
        
        # Create US holidays calendar
        us_holidays = holidays.US(years=years)
        
        # Check if date is holiday
        df['IsHoliday'] = df['Date'].isin(us_holidays.keys())
        
        logger.info(f"Created holiday features for years: {years}")
        return df
    
    def _create_weekend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weekend-related features"""
        # Weekend indicator (Friday, Saturday, Sunday)
        df['Weekend'] = df['DayOfWeek'].apply(lambda x: 'Weekend' if x in [4, 5, 6] else 'Weekday')
        
        # Binary weekend indicator
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Weekend or Holiday
        df['IsWeekendorHoliday'] = (df['IsWeekend'] == 1) | (df['IsHoliday'] == True)
        
        # Week type for grouping
        df['WeekType'] = df['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
        
        logger.info("Created weekend features")
        return df
    
    def _create_datetime_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create datetime keys for various aggregations"""
        # Datetime key (date + hour)
        df['Datetime_Key'] = df['DATE OCC'].dt.floor('d') + pd.to_timedelta(df['Hour'], unit='h')
        
        # Year-Month key
        df['YearMonth_Key'] = df['Year'].astype(str) + '_' + df['Month'].astype(str)
        
        # Date-Hour key
        df['Datetime_Hour_Key'] = df['Date'].astype(str) + '_' + df['Hour'].astype(str)
        
        logger.info("Created datetime keys")
        return df
    
    def get_time_features_list(self) -> List[str]:
        """Get list of all temporal features created"""
        features = ['Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'Quarter',
                   'IsHoliday', 'IsWeekend', 'IsWeekendorHoliday', 'Weekend', 'WeekType']
        
        if self.enable_cyclical:
            features.extend(['Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 
                           'DayOfWeek_cos', 'Month_sin', 'Month_cos'])
        
        return features