"""
Exploratory Data Analysis Module
Generates comprehensive EDA reports and visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy import stats
import geopandas as gpd

logger = logging.getLogger(__name__)


class CrimeEDA:
    """Handles exploratory data analysis for crime data"""
    
    def __init__(self, s3_handler=None):
        """
        Initialize EDA module
        
        Args:
            s3_handler: Optional S3Handler for saving plots
        """
        self.s3_handler = s3_handler
        self.figures = {}
        
    def run_full_eda(self, 
                     crime_df: pd.DataFrame,
                     geo_df: Optional[gpd.GeoDataFrame] = None,
                     save_plots: bool = True) -> Dict[str, Any]:
        """
        Run complete EDA pipeline
        
        Args:
            crime_df: Crime dataframe
            geo_df: Optional geographic boundaries
            save_plots: Whether to save plots to S3
            
        Returns:
            Dictionary containing EDA results
        """
        logger.info("Starting comprehensive EDA")
        
        results = {
            'basic_stats': self._basic_statistics(crime_df),
            'temporal_analysis': self._temporal_analysis(crime_df),
            'spatial_analysis': self._spatial_analysis(crime_df, geo_df),
            'crime_type_analysis': self._crime_type_analysis(crime_df),
            'victim_analysis': self._victim_demographics_analysis(crime_df),
            'statistical_tests': self._statistical_tests(crime_df)
        }
        
        if save_plots and self.s3_handler:
            self._save_all_plots()
        
        return results
    
    def _basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic dataset statistics"""
        stats = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'date_range': {
                'min': df['DATE OCC'].min() if 'DATE OCC' in df.columns else None,
                'max': df['DATE OCC'].max() if 'DATE OCC' in df.columns else None
            }
        }
        return stats
    
    def _temporal_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform temporal analysis"""
        results = {}
        
        # Crime trends by year
        fig, ax = plt.subplots(figsize=(12, 6))
        yearly_crimes = df.groupby('Year').size()
        yearly_crimes.plot(kind='bar', ax=ax)
        ax.set_title('Number of Crimes by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Crimes')
        self.figures['crimes_by_year'] = fig
        plt.close()
        
        # Monthly patterns
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_avg = df.groupby('Month').size() / df['Year'].nunique()
        monthly_avg.plot(kind='bar', color='green', ax=ax)
        ax.set_title('Average Number of Crimes per Month')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Number of Crimes')
        self.figures['average_crimes_by_month'] = fig
        plt.close()
        
        # Day of week and hour heatmap
        if 'Hour' in df.columns and 'DayOfWeek' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            pivot_table = df.pivot_table(
                index='DayOfWeek', 
                columns='Hour', 
                values='DATE OCC', 
                aggfunc='count'
            )
            sns.heatmap(pivot_table, cmap='coolwarm', ax=ax)
            ax.set_title('Crime Distribution by Day of Week and Hour')
            self.figures['crime_heatmap_dow_hour'] = fig
            plt.close()
        
        # Weekday vs Weekend comparison
        weekend_crimes = df[df['Weekend'] == 'Weekend'].shape[0] if 'Weekend' in df.columns else 0
        weekday_crimes = df[df['Weekend'] == 'Weekday'].shape[0] if 'Weekend' in df.columns else 0
        
        results['yearly_trend'] = yearly_crimes.to_dict()
        results['monthly_average'] = monthly_avg.to_dict()
        results['weekend_vs_weekday'] = {
            'weekend': weekend_crimes,
            'weekday': weekday_crimes
        }
        
        return results
    
    def _spatial_analysis(self, df: pd.DataFrame, geo_df: Optional[gpd.GeoDataFrame]) -> Dict[str, Any]:
        """Perform spatial analysis"""
        results = {}
        
        # Crimes by area
        area_crimes = df['AREA NAME'].value_counts()
        results['crimes_by_area'] = area_crimes.to_dict()
        
        # Top 10 crime areas
        fig, ax = plt.subplots(figsize=(12, 8))
        area_crimes.head(10).plot(kind='barh', ax=ax)
        ax.set_title('Top 10 Areas by Crime Count')
        ax.set_xlabel('Number of Crimes')
        self.figures['top_10_crime_areas'] = fig
        plt.close()
        
        # Geographic distribution if geo data available
        if geo_df is not None and 'AREA NAME' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 10))
            crime_counts = df['AREA NAME'].value_counts().reset_index()
            crime_counts.columns = ['AREA NAME', 'Crime Count']
            
            merged = geo_df.merge(crime_counts, left_on='STATION', right_on='AREA NAME', how='left')
            merged.plot(column='Crime Count', ax=ax, legend=True, cmap='Reds',
                       legend_kwds={'label': 'Crime Count by District'})
            ax.set_title('Crime Count Distribution by District')
            self.figures['crime_map'] = fig
            plt.close()
        
        return results
    
    def _crime_type_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze crime types and categories"""
        results = {}
        
        # Top crime types
        if 'Crm Cd Desc' in df.columns:
            top_crimes = df['Crm Cd Desc'].value_counts().head(15)
            results['top_15_crimes'] = top_crimes.to_dict()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            top_crimes.plot(kind='barh', ax=ax)
            ax.set_title('Top 15 Most Reported Crime Types')
            ax.set_xlabel('Number of Reports')
            self.figures['top_15_crime_types'] = fig
            plt.close()
        
        # Crime categories if available
        if 'Crime Category' in df.columns:
            category_counts = df['Crime Category'].value_counts()
            results['crime_categories'] = category_counts.to_dict()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            category_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
            ax.set_title('Distribution of Crime Categories')
            self.figures['crime_categories_pie'] = fig
            plt.close()
        
        # Crime severity
        if 'Crime Severity' in df.columns:
            severity_counts = df['Crime Severity'].value_counts()
            results['crime_severity'] = severity_counts.to_dict()
        
        return results
    
    def _victim_demographics_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze victim demographics"""
        results = {}
        
        # Age distribution
        if 'Vict Age' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram
            df['Vict Age'].hist(bins=50, ax=ax1)
            ax1.set_title('Victim Age Distribution')
            ax1.set_xlabel('Age')
            ax1.set_ylabel('Count')
            
            # Box plot by crime severity if available
            if 'Crime Severity' in df.columns:
                df.boxplot(column='Vict Age', by='Crime Severity', ax=ax2)
                ax2.set_title('Victim Age by Crime Severity')
            
            self.figures['victim_age_analysis'] = fig
            plt.close()
            
            results['age_stats'] = {
                'mean': df['Vict Age'].mean(),
                'median': df['Vict Age'].median(),
                'std': df['Vict Age'].std()
            }
        
        # Sex distribution
        if 'Vict Sex' in df.columns:
            sex_counts = df['Vict Sex'].value_counts()
            results['victim_sex'] = sex_counts.to_dict()
        
        # Descent distribution
        if 'Simplified Vict Descent' in df.columns:
            descent_counts = df['Simplified Vict Descent'].value_counts()
            results['victim_descent'] = descent_counts.to_dict()
            
            # Stacked bar chart by crime category
            if 'Crime Category' in df.columns:
                fig, ax = plt.subplots(figsize=(12, 8))
                contingency = pd.crosstab(df['Simplified Vict Descent'], df['Crime Category'])
                contingency.plot(kind='bar', stacked=True, ax=ax)
                ax.set_title('Crime Categories by Victim Descent')
                ax.set_xlabel('Victim Descent')
                ax.set_ylabel('Number of Crimes')
                ax.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                self.figures['crime_by_descent'] = fig
                plt.close()
        
        return results
    
    def _statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical tests"""
        results = {}
        
        # Test 1: Weekend vs Weekday crime rates
        if 'Weekend' in df.columns:
            weekend_data = df[df['Weekend'] == 'Weekend']['Crm Cd'].value_counts()
            weekday_data = df[df['Weekend'] == 'Weekday']['Crm Cd'].value_counts()
            
            # Mann-Whitney U test
            if len(weekend_data) > 0 and len(weekday_data) > 0:
                stat, pval = stats.mannwhitneyu(weekend_data, weekday_data)
                results['weekend_vs_weekday_test'] = {
                    'statistic': stat,
                    'p_value': pval,
                    'significant': pval < 0.05
                }
        
        # Test 2: Crime severity by victim age
        if 'Crime Severity' in df.columns and 'Vict Age' in df.columns:
            serious_ages = df[df['Crime Severity'] == 'Serious']['Vict Age'].dropna()
            less_serious_ages = df[df['Crime Severity'] == 'Less Serious']['Vict Age'].dropna()
            
            if len(serious_ages) > 0 and len(less_serious_ages) > 0:
                stat, pval = stats.mannwhitneyu(serious_ages, less_serious_ages)
                results['severity_by_age_test'] = {
                    'statistic': stat,
                    'p_value': pval,
                    'significant': pval < 0.05
                }
        
        # Test 3: Chi-square test for crime category and victim descent
        if 'Crime Category' in df.columns and 'Simplified Vict Descent' in df.columns:
            contingency = pd.crosstab(df['Crime Category'], df['Simplified Vict Descent'])
            chi2, pval, dof, expected = stats.chi2_contingency(contingency)
            results['category_descent_association'] = {
                'chi2': chi2,
                'p_value': pval,
                'degrees_of_freedom': dof,
                'significant': pval < 0.05
            }
        
        return results
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a text summary of EDA results"""
        report = []
        report.append("Crime Hotspot EDA Summary Report")
        report.append("=" * 50)
        
        # Basic statistics
        if 'basic_stats' in results:
            stats = results['basic_stats']
            report.append(f"\nDataset Shape: {stats['shape']}")
            report.append(f"Date Range: {stats['date_range']['min']} to {stats['date_range']['max']}")
        
        # Temporal insights
        if 'temporal_analysis' in results:
            temp = results['temporal_analysis']
            if 'weekend_vs_weekday' in temp:
                total = temp['weekend_vs_weekday']['weekend'] + temp['weekend_vs_weekday']['weekday']
                weekend_pct = (temp['weekend_vs_weekday']['weekend'] / total) * 100
                report.append(f"\nWeekend crimes: {weekend_pct:.1f}% of total")
        
        # Spatial insights
        if 'spatial_analysis' in results:
            spatial = results['spatial_analysis']
            if 'crimes_by_area' in spatial:
                top_area = list(spatial['crimes_by_area'].keys())[0]
                top_count = list(spatial['crimes_by_area'].values())[0]
                report.append(f"\nHighest crime area: {top_area} ({top_count} crimes)")
        
        # Crime type insights
        if 'crime_type_analysis' in results:
            crime_types = results['crime_type_analysis']
            if 'top_15_crimes' in crime_types:
                top_crime = list(crime_types['top_15_crimes'].keys())[0]
                report.append(f"\nMost common crime: {top_crime}")
        
        # Statistical test results
        if 'statistical_tests' in results:
            tests = results['statistical_tests']
            report.append("\nStatistical Test Results:")
            for test_name, test_result in tests.items():
                if 'significant' in test_result:
                    sig = "Significant" if test_result['significant'] else "Not significant"
                    report.append(f"  {test_name}: {sig} (p={test_result['p_value']:.4f})")
        
        return "\n".join(report)
    
    def _save_all_plots(self):
        """Save all generated plots to S3"""
        if not self.s3_handler:
            return
        
        for name, fig in self.figures.items():
            key = f"Crime_Hotspot_Prediction/EDA/{name}.png"
            try:
                # Save to temporary file
                temp_path = f"/tmp/{name}.png"
                fig.savefig(temp_path, bbox_inches='tight', dpi=150)
                
                # Upload to S3
                self.s3_handler.upload_file(temp_path, key)
                logger.info(f"Saved plot: {name}")
                
                # Clean up
                plt.close(fig)
            except Exception as e:
                logger.error(f"Error saving plot {name}: {e}")