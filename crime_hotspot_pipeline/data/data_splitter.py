"""
Data Splitter Module
Handles time-aware train/test splitting for temporal data
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class TimeAwareSplitter:
    """Handles time-aware data splitting for temporal datasets"""
    
    def __init__(self, datetime_column: str = 'Datetime_Key'):
        """
        Initialize splitter with datetime column name
        
        Args:
            datetime_column: Name of the datetime column to use for sorting
        """
        self.datetime_column = datetime_column
        
    def split_temporal_data(self, 
                          df: pd.DataFrame, 
                          train_ratio: float = 0.8,
                          target_column: str = 'is_hourly_hotspot') -> Dict[str, pd.DataFrame]:
        """
        Split temporal data maintaining time order
        
        Args:
            df: DataFrame to split
            train_ratio: Proportion of data for training
            target_column: Name of target column
            
        Returns:
            Dictionary with train and test dataframes
        """
        logger.info(f"Performing time-aware split with ratio {train_ratio}")
        
        # Sort by datetime
        df_sorted = df.sort_values(self.datetime_column).reset_index(drop=True)
        
        # Calculate split index
        split_index = int(len(df_sorted) * train_ratio)
        
        # Split data
        train_df = df_sorted.iloc[:split_index].copy()
        test_df = df_sorted.iloc[split_index:].copy()
        
        # Log split information
        self._log_split_info(train_df, test_df, target_column)
        
        return {
            'train': train_df,
            'test': test_df
        }
    
    def split_features_target(self, 
                            df: pd.DataFrame,
                            target_column: str,
                            features_to_exclude: list) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split dataframe into features and target
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            features_to_exclude: List of columns to exclude from features
            
        Returns:
            Tuple of (features, target)
        """
        # Get target
        y = df[target_column]
        
        # Get features
        exclude_cols = features_to_exclude + [target_column]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Select only numeric features
        X = df[feature_cols].select_dtypes(include=[np.number])
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y
    
    def create_validation_split(self,
                              train_df: pd.DataFrame,
                              val_ratio: float = 0.2,
                              target_column: str = 'is_hourly_hotspot') -> Dict[str, pd.DataFrame]:
        """
        Create validation split from training data
        
        Args:
            train_df: Training dataframe
            val_ratio: Proportion of training data for validation
            target_column: Name of target column
            
        Returns:
            Dictionary with train and validation dataframes
        """
        # Sort by datetime
        train_sorted = train_df.sort_values(self.datetime_column).reset_index(drop=True)
        
        # Calculate split index
        split_index = int(len(train_sorted) * (1 - val_ratio))
        
        # Split data
        new_train = train_sorted.iloc[:split_index].copy()
        validation = train_sorted.iloc[split_index:].copy()
        
        logger.info(f"Created validation split: Train={len(new_train)}, Val={len(validation)}")
        
        return {
            'train': new_train,
            'validation': validation
        }
    
    def stratified_temporal_split(self,
                                df: pd.DataFrame,
                                train_ratio: float = 0.8,
                                target_column: str = 'is_hourly_hotspot',
                                stratify_column: str = 'AREA NAME') -> Dict[str, pd.DataFrame]:
        """
        Perform time-aware split while maintaining area distribution
        
        Args:
            df: DataFrame to split
            train_ratio: Proportion of data for training
            target_column: Name of target column
            stratify_column: Column to stratify by (e.g., area)
            
        Returns:
            Dictionary with train and test dataframes
        """
        logger.info(f"Performing stratified temporal split by {stratify_column}")
        
        train_dfs = []
        test_dfs = []
        
        # Split each group separately
        for group_name, group_df in df.groupby(stratify_column):
            group_sorted = group_df.sort_values(self.datetime_column).reset_index(drop=True)
            split_index = int(len(group_sorted) * train_ratio)
            
            train_dfs.append(group_sorted.iloc[:split_index])
            test_dfs.append(group_sorted.iloc[split_index:])
        
        # Combine all groups
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        # Sort by datetime again
        train_df = train_df.sort_values(self.datetime_column).reset_index(drop=True)
        test_df = test_df.sort_values(self.datetime_column).reset_index(drop=True)
        
        self._log_split_info(train_df, test_df, target_column)
        
        return {
            'train': train_df,
            'test': test_df
        }
    
    def _log_split_info(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str):
        """Log information about the split"""
        logger.info(f"Train set size: {len(train_df)}")
        logger.info(f"Test set size: {len(test_df)}")
        
        if target_column in train_df.columns:
            train_pos = train_df[target_column].sum()
            test_pos = test_df[target_column].sum()
            
            logger.info(f"Train positive class: {train_pos} ({train_pos/len(train_df)*100:.2f}%)")
            logger.info(f"Test positive class: {test_pos} ({test_pos/len(test_df)*100:.2f}%)")
        
        if self.datetime_column in train_df.columns:
            logger.info(f"Train date range: {train_df[self.datetime_column].min()} to {train_df[self.datetime_column].max()}")
            logger.info(f"Test date range: {test_df[self.datetime_column].min()} to {test_df[self.datetime_column].max()}")