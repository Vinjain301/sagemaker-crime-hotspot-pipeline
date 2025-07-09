"""
Lag Features Engineering Module
Creates temporal lag features for time series analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class LagFeatureEngineer:
    """Creates lag features for temporal patterns"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with lag feature configuration
        
        Args:
            config: Lag features configuration
        """
        self.config = config
        
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all lag features
        
        Args:
            df: Input dataframe with temporal and crime rate features
            
        Returns:
            pd.DataFrame: Dataframe with lag features added
        """
        # Sort by area and datetime for proper lag calculation
        df = df.sort_values(['AREA NAME', 'Datetime_Key'])
        
        # Create hourly lag features
        df = self._create_hourly_lags(df)
        
        # Create daily lag features
        df = self._create_daily_lags(df)
        
        # Create yearly lag features
        df = self._create_yearly_lags(df)
        
        logger.info("Created all lag features")
        return df
    
    def _create_hourly_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create hour-based lag features"""
        
        # Crime rate last 1 hour
        df['crime_rate_last_1h'] = df.groupby('AREA NAME')['Hourly_Crime_Rate_per_1000'].shift(1)
        
        # Crime rate last 3-hour rolling average
        df['crime_rate_last_3h_avg'] = (
            df.groupby('AREA NAME')['Hourly_Crime_Rate_per_1000']
            .rolling(3, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        
        # Crime rate last 6-hour rolling sum
        df['crime_rate_last_6h_sum'] = (
            df.groupby('AREA NAME')['Hourly_Crime_Rate_per_1000']
            .rolling(6, min_periods=1)
            .sum()
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        
        # Was hourly hotspot last hour
        if 'is_hourly_hotspot' in df.columns:
            df['was_hourly_hotspot_last_hour'] = df.groupby('AREA NAME')['is_hourly_hotspot'].shift(1)
        
        logger.info("Created hourly lag features")
        return df
    
    def _create_daily_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create day-based lag features"""
        
        # Crime rate yesterday same hour
        df['Datetime_Prev_Day'] = df['Datetime_Key'] - pd.Timedelta(days=1)
        df['Prev_Day_Key'] = df['AREA NAME'] + '_' + df['Datetime_Prev_Day'].dt.strftime('%Y-%m-%d_%H')
        
        # Create lookup dictionary
        lookup_df = df[['AREA NAME', 'Datetime_Key', 'Hourly_Crime_Rate_per_1000']].copy()
        lookup_df['Lookup_Key'] = lookup_df['AREA NAME'] + '_' + lookup_df['Datetime_Key'].dt.strftime('%Y-%m-%d_%H')
        lookup_dict = lookup_df.set_index('Lookup_Key')['Hourly_Crime_Rate_per_1000'].to_dict()
        
        df['crime_rate_yesterday_same_hour'] = df['Prev_Day_Key'].map(lookup_dict)
        
        # Was daily hotspot yesterday
        if 'is_daily_hotspot' in df.columns:
            # Create daily hotspot lookup
            daily_lookup = (
                df.drop_duplicates(['AREA NAME', 'Date'])
                [['AREA NAME', 'Date', 'is_daily_hotspot']]
                .copy()
            )
            daily_lookup['Date_Yesterday'] = daily_lookup['Date'] + pd.Timedelta(days=1)
            daily_lookup['join_key'] = daily_lookup['AREA NAME'].astype(str) + '_' + daily_lookup['Date_Yesterday'].astype(str)
            hotspot_dict = dict(zip(daily_lookup['join_key'], daily_lookup['is_daily_hotspot']))
            
            df['join_key'] = df['AREA NAME'].astype(str) + '_' + df['Date'].astype(str)
            df['was_daily_hotspot_yesterday'] = df['join_key'].map(hotspot_dict).fillna(0).astype(int)
            df = df.drop(columns=['join_key'])
        
        # Clean up temporary columns
        df = df.drop(columns=['Datetime_Prev_Day', 'Prev_Day_Key'])
        
        logger.info("Created daily lag features")
        return df
    
    def _create_yearly_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create year-based lag features"""
        
        # Was hotspot same day last year
        if 'is_daily_hotspot' in df.columns:
            df['Date_Last_Year'] = df['Datetime_Key'] - pd.DateOffset(years=1)
            df['Last_Year_Key'] = df['AREA NAME'] + '_' + df['Date_Last_Year'].dt.strftime('%Y-%m-%d')
            
            # Create yearly lookup
            year_lookup = df.drop_duplicates(['AREA NAME', 'Date'])[['AREA NAME', 'Date', 'is_daily_hotspot']].copy()
            year_lookup['Year_Key'] = year_lookup['AREA NAME'] + '_' + pd.to_datetime(year_lookup['Date']).dt.strftime('%Y-%m-%d')
            year_lookup_dict = year_lookup.set_index('Year_Key')['is_daily_hotspot'].to_dict()
            
            df['was_hotspot_last_year_same_day'] = df['Last_Year_Key'].map(year_lookup_dict)
            
            # Clean up
            df = df.drop(columns=['Date_Last_Year', 'Last_Year_Key'])
        
        logger.info("Created yearly lag features")
        return df
    
    def get_lag_features_list(self) -> List[str]:
        """Get list of all lag features created"""
        return [
            'crime_rate_last_1h',
            'crime_rate_last_3h_avg',
            'crime_rate_last_6h_sum',
            'crime_rate_yesterday_same_hour',
            'was_hourly_hotspot_last_hour',
            'was_daily_hotspot_yesterday',
            'was_hotspot_last_year_same_day'
        ]