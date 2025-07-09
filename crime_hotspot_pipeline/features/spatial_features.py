"""
Spatial Features Engineering Module
Handles location-based feature creation
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SpatialFeatureEngineer:
    """Creates spatial/location-based features"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with spatial feature configuration
        
        Args:
            config: Spatial features configuration
        """
        self.config = config
        self.population_data = self._load_population_data()
        
    def _load_population_data(self) -> pd.DataFrame:
        """Load and format population data from config"""
        population_by_year = {
            2020: {
                'WILSHIRE': 143600, 'CENTRAL': 180700, 'SOUTHWEST': 118100, 'VAN NUYS': 163500,
                'HOLLENBECK': 136900, 'RAMPART': 139400, 'NEWTON': 108000, 'NORTHEAST': 94500,
                '77TH STREET': 105800, 'HOLLYWOOD': 130800, 'HARBOR': 135300, 'WEST VALLEY': 160400,
                'WEST LA': 130900, 'N HOLLYWOOD': 154800, 'PACIFIC': 68700, 'DEVONSHIRE': 124500,
                'MISSION': 158500, 'SOUTHEAST': 123700, 'OLYMPIC': 82000, 'FOOTHILL': 88500, 'TOPANGA': 127100
            },
            2021: {
                'WILSHIRE': 144080, 'CENTRAL': 181300, 'SOUTHWEST': 118400, 'VAN NUYS': 164300,
                'HOLLENBECK': 137400, 'RAMPART': 139900, 'NEWTON': 108500, 'NORTHEAST': 95000,
                '77TH STREET': 106400, 'HOLLYWOOD': 131500, 'HARBOR': 135800, 'WEST VALLEY': 161100,
                'WEST LA': 131500, 'N HOLLYWOOD': 155700, 'PACIFIC': 69000, 'DEVONSHIRE': 125100,
                'MISSION': 159300, 'SOUTHEAST': 124100, 'OLYMPIC': 82500, 'FOOTHILL': 89000, 'TOPANGA': 127700
            },
            2022: {
                'WILSHIRE': 144570, 'CENTRAL': 182150, 'SOUTHWEST': 118750, 'VAN NUYS': 165100,
                'HOLLENBECK': 137880, 'RAMPART': 140500, 'NEWTON': 109000, 'NORTHEAST': 95500,
                '77TH STREET': 106900, 'HOLLYWOOD': 132200, 'HARBOR': 136500, 'WEST VALLEY': 161900,
                'WEST LA': 132100, 'N HOLLYWOOD': 156500, 'PACIFIC': 69300, 'DEVONSHIRE': 125700,
                'MISSION': 160000, 'SOUTHEAST': 124500, 'OLYMPIC': 83000, 'FOOTHILL': 89500, 'TOPANGA': 128300
            },
            2023: {
                'WILSHIRE': 145320, 'CENTRAL': 182765, 'SOUTHWEST': 118940, 'VAN NUYS': 165732,
                'HOLLENBECK': 138210, 'RAMPART': 140875, 'NEWTON': 109432, 'NORTHEAST': 95678,
                '77TH STREET': 107215, 'HOLLYWOOD': 132540, 'HARBOR': 136890, 'WEST VALLEY': 162305,
                'WEST LA': 132450, 'N HOLLYWOOD': 156780, 'PACIFIC': 69540, 'DEVONSHIRE': 125890,
                'MISSION': 160432, 'SOUTHEAST': 124765, 'OLYMPIC': 83210, 'FOOTHILL': 89765, 'TOPANGA': 128540
            }
        }
        
        # Convert to dataframe
        population_records = []
        for year, districts in population_by_year.items():
            for district, population in districts.items():
                population_records.append({
                    'AREA NAME': district,
                    'Year': year,
                    'Population': population
                })
        
        return pd.DataFrame(population_records)
    
    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create spatial features including crime rates per population
        
        Args:
            df: Input dataframe with location information
            
        Returns:
            pd.DataFrame: Dataframe with spatial features added
        """
        # Ensure AREA NAME is uppercase
        if 'AREA NAME' in df.columns:
            df['AREA NAME'] = df['AREA NAME'].str.upper()
        
        # Create crime rate features
        df = self._create_crime_rate_features(df)
        
        # Add district centroids if lat/lon available
        if 'LAT' in df.columns and 'LON' in df.columns:
            df = self._add_district_centroids(df)
        
        logger.info("Spatial features created successfully")
        return df
    
    def _create_crime_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create crime rate per 1000 population features"""
        
        # Monthly crime rate
        monthly = df.groupby(['AREA NAME', 'Year', 'Month']).size().reset_index(name='Monthly_Crime_Count')
        monthly = monthly.merge(self.population_data, on=['AREA NAME', 'Year'], how='left')
        monthly['Monthly_Crime_Rate_per_1000'] = (monthly['Monthly_Crime_Count'] / monthly['Population']) * 1000
        df = df.merge(monthly[['AREA NAME', 'Year', 'Month', 'Monthly_Crime_Rate_per_1000']],
                      on=['AREA NAME', 'Year', 'Month'], how='left')
        
        # Yearly crime rate
        yearly = df.groupby(['AREA NAME', 'Year']).size().reset_index(name='Yearly_Crime_Count')
        yearly = yearly.merge(self.population_data, on=['AREA NAME', 'Year'], how='left')
        yearly['Yearly_Crime_Rate_per_1000'] = (yearly['Yearly_Crime_Count'] / yearly['Population']) * 1000
        df = df.merge(yearly[['AREA NAME', 'Year', 'Yearly_Crime_Rate_per_1000']],
                      on=['AREA NAME', 'Year'], how='left')
        
        # Daily crime rate
        daily = df.groupby(['AREA NAME', 'Date']).size().reset_index(name='Daily_Crime_Count')
        daily['Year'] = pd.to_datetime(daily['Date']).dt.year
        daily = daily.merge(self.population_data, on=['AREA NAME', 'Year'], how='left')
        daily['Daily_Crime_Rate_per_1000'] = (daily['Daily_Crime_Count'] / daily['Population']) * 1000
        df = df.merge(daily[['AREA NAME', 'Date', 'Daily_Crime_Rate_per_1000']],
                      on=['AREA NAME', 'Date'], how='left')
        
        # Hourly crime rate
        hourly = df.groupby(['AREA NAME', 'Date', 'Hour']).size().reset_index(name='Hourly_Crime_Count')
        hourly['Year'] = pd.to_datetime(hourly['Date']).dt.year
        hourly = hourly.merge(self.population_data, on=['AREA NAME', 'Year'], how='left')
        hourly['Hourly_Crime_Rate_per_1000'] = (hourly['Hourly_Crime_Count'] / hourly['Population']) * 1000
        df = df.merge(hourly[['AREA NAME', 'Date', 'Hour', 'Hourly_Crime_Rate_per_1000']],
                      on=['AREA NAME', 'Date', 'Hour'], how='left')
        
        logger.info("Created crime rate features")
        return df
    
    def _add_district_centroids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add district centroid coordinates"""
        # Calculate district centroids
        district_centroids = df.groupby('AREA NAME')[['LAT', 'LON']].mean().reset_index()
        district_centroids.columns = ['AREA NAME', 'Center_LAT', 'Center_LON']
        
        # Merge back to main dataframe
        df = df.merge(district_centroids, on='AREA NAME', how='left')
        
        logger.info("Added district centroid coordinates")
        return df
    
    def get_spatial_features_list(self) -> list:
        """Get list of all spatial features created"""
        return [
            'Monthly_Crime_Rate_per_1000',
            'Yearly_Crime_Rate_per_1000',
            'Daily_Crime_Rate_per_1000',
            'Hourly_Crime_Rate_per_1000',
            'Center_LAT',
            'Center_LON'
        ]