"""
Data Cleaner Module
Handles data cleaning and preprocessing operations
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles all data cleaning operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataCleaner with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.features_to_drop = config.get('features_to_drop', [])
        
    def clean_crime_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean crime dataset based on identified issues
        
        Args:
            df: Raw crime dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info(f"Starting data cleaning. Initial shape: {df.shape}")
        
        # Create a copy to avoid modifying original
        data = df.copy()
        
        # Drop columns with extreme missing values
        columns_to_drop = ['Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4', 'Cross Street']
        data = self._drop_columns(data, columns_to_drop)
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Convert date columns
        data = self._convert_dates(data)
        
        # Fix data quality issues
        data = self._fix_victim_age(data)
        data = self._fix_victim_sex(data)
        
        # Remove 2024 data (incomplete year)
        data = self._filter_years(data)
        
        # Standardize text fields
        data = self._standardize_text_fields(data)
        
        logger.info(f"Data cleaning complete. Final shape: {data.shape}")
        return data
    
    def _drop_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Drop specified columns if they exist"""
        existing_cols = [col for col in columns if col in df.columns]
        if existing_cols:
            logger.info(f"Dropping columns: {existing_cols}")
            df = df.drop(columns=existing_cols)
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column type and missing rate"""
        
        # Encode missing weapon info
        df['Weapon Used Cd'] = df['Weapon Used Cd'].fillna('Weapon Usage Not Known')
        df['Weapon Desc'] = df['Weapon Desc'].fillna('Unknown')
        
        # Encode missing descent and premise
        df['Vict Descent'] = df['Vict Descent'].fillna('X')
        df['Premis Desc'] = df['Premis Desc'].fillna('Unknown')
        
        # Impute Mocodes with mode
        if 'Mocodes' in df.columns:
            mode_value = df['Mocodes'].mode()
            if len(mode_value) > 0:
                df['Mocodes'] = df['Mocodes'].fillna(mode_value[0])
        
        # Handle victim sex
        df['Vict Sex'] = df['Vict Sex'].fillna('X')
        df = df[df['Vict Sex'] != '-']  # Remove invalid entries
        
        # Drop rows with missing critical values
        critical_columns = ['Premis Cd', 'Status']
        df = df.dropna(subset=critical_columns)
        
        return df
    
    def _convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date columns to datetime"""
        date_columns = ['DATE OCC', 'Date Rptd']
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"Converted {col} to datetime")
        
        return df
    
    def _fix_victim_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix negative and extreme victim ages"""
        if 'Vict Age' in df.columns:
            # Calculate mean age for valid ages
            valid_ages = df['Vict Age'][df['Vict Age'] > 0]
            mean_age = valid_ages.mean() if len(valid_ages) > 0 else 30
            
            # Replace negative ages with mean
            df['Vict Age'] = df['Vict Age'].apply(lambda x: mean_age if x < 0 else x)
            
            # Cap age at 100
            df['Vict Age'] = df['Vict Age'].clip(lower=0, upper=100)
            
            logger.info("Fixed victim age values")
        
        return df
    
    def _fix_victim_sex(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix invalid victim sex values"""
        if 'Vict Sex' in df.columns:
            valid_values = ['M', 'F', 'X']
            df.loc[~df['Vict Sex'].isin(valid_values), 'Vict Sex'] = 'X'
            logger.info("Fixed victim sex values")
        
        return df
    
    def _filter_years(self, df: pd.DataFrame, max_year: int = 2023) -> pd.DataFrame:
        """Remove incomplete years from dataset"""
        if 'DATE OCC' in df.columns:
            df['Year'] = df['DATE OCC'].dt.year
            initial_count = len(df)
            df = df[df['Year'] <= max_year]
            removed_count = initial_count - len(df)
            logger.info(f"Removed {removed_count} records from years after {max_year}")
        
        return df
    
    def _standardize_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize text fields to uppercase"""
        text_fields = ['AREA NAME', 'Crm Cd Desc', 'Status', 'Status Desc']
        
        for field in text_fields:
            if field in df.columns and df[field].dtype == 'object':
                df[field] = df[field].str.upper()
                logger.info(f"Standardized {field} to uppercase")
        
        return df
    
    def clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean weather dataset
        
        Args:
            df: Raw weather dataframe
            
        Returns:
            pd.DataFrame: Cleaned weather dataframe
        """
        logger.info("Cleaning weather data")
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows from weather data")
        
        # Convert date column
        if 'DATE OCC' in df.columns:
            df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
        
        # Handle missing weather values
        weather_columns = ['Temp', 'Wspeed', 'Prec', 'Sun_time']
        for col in weather_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def validate_cleaned_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate cleaned data and return summary statistics
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'date_range': None,
            'unique_areas': None
        }
        
        if 'DATE OCC' in df.columns:
            validation_results['date_range'] = {
                'min': df['DATE OCC'].min(),
                'max': df['DATE OCC'].max()
            }
        
        if 'AREA NAME' in df.columns:
            validation_results['unique_areas'] = df['AREA NAME'].nunique()
        
        logger.info("Data validation complete")
        return validation_results