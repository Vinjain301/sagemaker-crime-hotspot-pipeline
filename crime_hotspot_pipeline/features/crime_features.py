"""
Crime Features Engineering Module
Handles crime-specific feature creation including categories, severity, and hotspot detection
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class CrimeFeatureEngineer:
    """Creates crime-specific features"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with crime feature configuration
        
        Args:
            config: Crime features configuration
        """
        self.config = config
        self.crime_category_map = self._load_crime_category_mapping()
        self.severity_mapping = config.get('severity_mapping', {1: 'Serious', 2: 'Less Serious'})
        
    def _load_crime_category_mapping(self) -> Dict[str, str]:
        """Load crime category mapping from config"""
        crime_category_map = {
            # VIOLENT CRIMES
            'ARSON': 'VIOLENT CRIMES',
            'ASSAULT WITH DEADLY WEAPON ON POLICE OFFICER': 'VIOLENT CRIMES',
            'ASSAULT WITH DEADLY WEAPON': 'VIOLENT CRIMES',
            'AGGRAVATED ASSAULT': 'VIOLENT CRIMES',
            'ATTEMPTED ROBBERY': 'VIOLENT CRIMES',
            'BATTERY - SIMPLE ASSAULT': 'VIOLENT CRIMES',
            'BATTERY ON A FIREFIGHTER': 'VIOLENT CRIMES',
            'BATTERY POLICE (SIMPLE)': 'VIOLENT CRIMES',
            'BRANDISH WEAPON': 'VIOLENT CRIMES',
            'CRIMINAL HOMICIDE': 'VIOLENT CRIMES',
            'CRIMINAL THREATS - NO WEAPON DISPLAYED': 'VIOLENT CRIMES',
            'DISCHARGE FIREARMS/SHOTS FIRED': 'VIOLENT CRIMES',
            'DRUNK ROLL': 'VIOLENT CRIMES',
            'EXTORTION': 'VIOLENT CRIMES',
            'FALSE IMPRISONMENT': 'VIOLENT CRIMES',
            'HUMAN TRAFFICKING - COMMERCIAL SEX ACTS': 'VIOLENT CRIMES',
            'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE': 'VIOLENT CRIMES',
            'INTIMATE PARTNER - AGGRAVATED ASSAULT': 'VIOLENT CRIMES',
            'INTIMATE PARTNER - SIMPLE ASSAULT': 'VIOLENT CRIMES',
            'KIDNAPPING': 'VIOLENT CRIMES',
            'KIDNAPPING - GRAND ATTEMPT': 'VIOLENT CRIMES',
            'LYNCHING': 'VIOLENT CRIMES',
            'LYNCHING - ATTEMPTED': 'VIOLENT CRIMES',
            'MANSLAUGHTER, NEGLIGENT': 'VIOLENT CRIMES',
            'OTHER ASSAULT': 'VIOLENT CRIMES',
            'ROBBERY': 'VIOLENT CRIMES',
            'SHOTS FIRED AT INHABITED DWELLING': 'VIOLENT CRIMES',
            'SHOTS FIRED AT MOVING VEHICLE, TRAIN OR AIRCRAFT': 'VIOLENT CRIMES',
            'STALKING': 'VIOLENT CRIMES',
            'THREATENING PHONE CALLS/LETTERS': 'VIOLENT CRIMES',
            'THROWING OBJECT AT MOVING VEHICLE': 'VIOLENT CRIMES',
            'WEAPONS POSSESSION/BOMBING': 'VIOLENT CRIMES',
            
            # PROPERTY CRIMES
            'BIKE - ATTEMPTED STOLEN': 'PROPERTY CRIMES',
            'BIKE - STOLEN': 'PROPERTY CRIMES',
            'BOAT - STOLEN': 'PROPERTY CRIMES',
            'BURGLARY': 'PROPERTY CRIMES',
            'BURGLARY FROM VEHICLE': 'PROPERTY CRIMES',
            'BURGLARY FROM VEHICLE, ATTEMPTED': 'PROPERTY CRIMES',
            'BURGLARY, ATTEMPTED': 'PROPERTY CRIMES',
            'DEFRAUDING INNKEEPER/THEFT OF SERVICES, $950 & UNDER': 'PROPERTY CRIMES',
            'DEFRAUDING INNKEEPER/THEFT OF SERVICES, OVER $950.01': 'PROPERTY CRIMES',
            'DISHONEST EMPLOYEE - GRAND THEFT': 'PROPERTY CRIMES',
            'DISHONEST EMPLOYEE - PETTY THEFT': 'PROPERTY CRIMES',
            'DISHONEST EMPLOYEE ATTEMPTED THEFT': 'PROPERTY CRIMES',
            'DRIVING WITHOUT OWNER CONSENT (DWOC)': 'PROPERTY CRIMES',
            'GRAND THEFT / AUTO REPAIR': 'PROPERTY CRIMES',
            'PETTY THEFT - AUTO REPAIR': 'PROPERTY CRIMES',
            'PICKPOCKET': 'PROPERTY CRIMES',
            'PICKPOCKET, ATTEMPT': 'PROPERTY CRIMES',
            'PURSE SNATCHING': 'PROPERTY CRIMES',
            'PURSE SNATCHING - ATTEMPT': 'PROPERTY CRIMES',
            'SHOPLIFTING - ATTEMPT': 'PROPERTY CRIMES',
            'SHOPLIFTING - PETTY THEFT ($950 & UNDER)': 'PROPERTY CRIMES',
            'SHOPLIFTING-GRAND THEFT ($950.01 & OVER)': 'PROPERTY CRIMES',
            'TELEPHONE PROPERTY - DAMAGE': 'PROPERTY CRIMES',
            'THEFT FROM MOTOR VEHICLE - ATTEMPT': 'PROPERTY CRIMES',
            'THEFT FROM MOTOR VEHICLE - GRAND ($950.01 AND OVER)': 'PROPERTY CRIMES',
            'THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)': 'PROPERTY CRIMES',
            'THEFT FROM PERSON - ATTEMPT': 'PROPERTY CRIMES',
            'THEFT PLAIN - ATTEMPT': 'PROPERTY CRIMES',
            'THEFT PLAIN - PETTY ($950 & UNDER)': 'PROPERTY CRIMES',
            'THEFT-GRAND ($950.01 & OVER)EXCPT,GUNS,FOWL,LIVESTK,PROD': 'PROPERTY CRIMES',
            'THEFT, COIN MACHINE - ATTEMPT': 'PROPERTY CRIMES',
            'THEFT, COIN MACHINE - GRAND ($950.01 & OVER)': 'PROPERTY CRIMES',
            'THEFT, COIN MACHINE - PETTY ($950 & UNDER)': 'PROPERTY CRIMES',
            'THEFT, PERSON': 'PROPERTY CRIMES',
            'TILL TAP - GRAND THEFT ($950.01 & OVER)': 'PROPERTY CRIMES',
            'TILL TAP - PETTY ($950 & UNDER)': 'PROPERTY CRIMES',
            'TRESPASSING': 'PROPERTY CRIMES',
            'VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)': 'PROPERTY CRIMES',
            'VANDALISM - MISDEAMEANOR ($399 OR UNDER)': 'PROPERTY CRIMES',
            'VEHICLE - ATTEMPT STOLEN': 'PROPERTY CRIMES',
            'VEHICLE - STOLEN': 'PROPERTY CRIMES',
            'VEHICLE, STOLEN - OTHER (MOTORIZED SCOOTERS, BIKES, ETC)': 'PROPERTY CRIMES',
            
            # SEX CRIMES
            'BATTERY WITH SEXUAL CONTACT': 'SEX CRIMES',
            'BEASTIALITY, CRIME AGAINST NATURE SEXUAL ASSLT WITH ANIM': 'SEX CRIMES',
            'INDECENT EXPOSURE': 'SEX CRIMES',
            'LEWD CONDUCT': 'SEX CRIMES',
            'ORAL COPULATION': 'SEX CRIMES',
            'PANDERING': 'SEX CRIMES',
            'PEEPING TOM': 'SEX CRIMES',
            'PIMPING': 'SEX CRIMES',
            'RAPE, ATTEMPTED': 'SEX CRIMES',
            'RAPE, FORCIBLE': 'SEX CRIMES',
            'SEX OFFENDER REGISTRANT OUT OF COMPLIANCE': 'SEX CRIMES',
            'SEX,UNLAWFUL(INC MUTUAL CONSENT, PENETRATION W/ FRGN OBJ)': 'SEX CRIMES',
            'SEXUAL PENETRATION W/FOREIGN OBJECT': 'SEX CRIMES',
            'SODOMY/SEXUAL CONTACT B/W PENIS OF ONE PERS TO ANUS OTH': 'SEX CRIMES',
            
            # CRIMES AGAINST CHILDREN
            'CHILD ABANDONMENT': 'CRIMES AGAINST CHILDREN',
            'CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT': 'CRIMES AGAINST CHILDREN',
            'CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT': 'CRIMES AGAINST CHILDREN',
            'CHILD ANNOYING (17YRS & UNDER)': 'CRIMES AGAINST CHILDREN',
            'CHILD NEGLECT (SEE 300 W.I.C.)': 'CRIMES AGAINST CHILDREN',
            'CHILD PORNOGRAPHY': 'CRIMES AGAINST CHILDREN',
            'CHILD STEALING': 'CRIMES AGAINST CHILDREN',
            'CONTRIBUTING': 'CRIMES AGAINST CHILDREN',
            'CRM AGNST CHLD (13 OR UNDER) (14-15 & SUSP 10 YRS OLDER)': 'CRIMES AGAINST CHILDREN',
            'DRUGS, TO A MINOR': 'CRIMES AGAINST CHILDREN',
            'INCEST (SEXUAL ACTS BETWEEN BLOOD RELATIVES)': 'CRIMES AGAINST CHILDREN',
            'LEWD/LASCIVIOUS ACTS WITH CHILD': 'CRIMES AGAINST CHILDREN',
            'LETTERS, LEWD - TELEPHONE CALLS, LEWD': 'CRIMES AGAINST CHILDREN',
            
            # FRAUD & FINANCIAL CRIMES
            'BUNCO, ATTEMPT': 'FRAUD & FINANCIAL CRIMES',
            'BUNCO, GRAND THEFT': 'FRAUD & FINANCIAL CRIMES',
            'BUNCO, PETTY THEFT': 'FRAUD & FINANCIAL CRIMES',
            'COUNTERFEIT': 'FRAUD & FINANCIAL CRIMES',
            'CREDIT CARDS, FRAUD USE ($950 & UNDER': 'FRAUD & FINANCIAL CRIMES',
            'CREDIT CARDS, FRAUD USE ($950.01 & OVER)': 'FRAUD & FINANCIAL CRIMES',
            'DOCUMENT FORGERY / STOLEN FELONY': 'FRAUD & FINANCIAL CRIMES',
            'DOCUMENT WORTHLESS ($200 & UNDER)': 'FRAUD & FINANCIAL CRIMES',
            'DOCUMENT WORTHLESS ($200.01 & OVER)': 'FRAUD & FINANCIAL CRIMES',
            'EMBEZZLEMENT, GRAND THEFT ($950.01 & OVER)': 'FRAUD & FINANCIAL CRIMES',
            'EMBEZZLEMENT, PETTY THEFT ($950 & UNDER)': 'FRAUD & FINANCIAL CRIMES',
            'GRAND THEFT / INSURANCE FRAUD': 'FRAUD & FINANCIAL CRIMES',
            'THEFT OF IDENTITY': 'FRAUD & FINANCIAL CRIMES',
            'UNAUTHORIZED COMPUTER ACCESS': 'FRAUD & FINANCIAL CRIMES',
            
            # CRIMES AGAINST PUBLIC ORDER & SAFETY
            'BIGAMY': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            'BLOCKING DOOR INDUCTION CENTER': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            'BOMB SCARE': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            'CRUELTY TO ANIMALS': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            'DISRUPT SCHOOL': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            'DISTURBING THE PEACE': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            'FAILURE TO DISPERSE': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            'FAILURE TO YIELD': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            'ILLEGAL DUMPING': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            'INCITING A RIOT': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            'OTHER MISCELLANEOUS CRIME': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            'PROWLER': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            'RECKLESS DRIVING': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            'REPLICA FIREARMS(SALE,DISPLAY,MANUFACTURE OR DISTRIBUTE)': 'CRIMES AGAINST PUBLIC ORDER & SAFETY',
            
            # CRIMES AGAINST JUSTICE & GOVERNMENT
            'BRIBERY': 'CRIMES AGAINST JUSTICE & GOVERNMENT',
            'CONSPIRACY': 'CRIMES AGAINST JUSTICE & GOVERNMENT',
            'CONTEMPT OF COURT': 'CRIMES AGAINST JUSTICE & GOVERNMENT',
            'FALSE POLICE REPORT': 'CRIMES AGAINST JUSTICE & GOVERNMENT',
            'FIREARMS EMERGENCY PROTECTIVE ORDER (FIREARMS EPO)': 'CRIMES AGAINST JUSTICE & GOVERNMENT',
            'FIREARMS RESTRAINING ORDER (FIREARMS RO)': 'CRIMES AGAINST JUSTICE & GOVERNMENT',
            'RESISTING ARREST': 'CRIMES AGAINST JUSTICE & GOVERNMENT',
            'VIOLATION OF COURT ORDER': 'CRIMES AGAINST JUSTICE & GOVERNMENT',
            'VIOLATION OF RESTRAINING ORDER': 'CRIMES AGAINST JUSTICE & GOVERNMENT',
            'VIOLATION OF TEMPORARY RESTRAINING ORDER': 'CRIMES AGAINST JUSTICE & GOVERNMENT',
        }
        
        return crime_category_map
    
    def create_crime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create crime-specific features
        
        Args:
            df: Input dataframe with crime information
            
        Returns:
            pd.DataFrame: Dataframe with crime features added
        """
        # Create crime category
        if 'Crm Cd Desc' in df.columns:
            df['Crime Category'] = df['Crm Cd Desc'].map(self.crime_category_map)
            logger.info("Created crime category feature")
        
        # Create crime severity
        if 'Part 1-2' in df.columns:
            df['Crime Severity'] = df['Part 1-2'].map(self.severity_mapping)
            logger.info("Created crime severity feature")
        
        # Create victim descent simplification
        if 'Vict Descent' in df.columns:
            df['Simplified Vict Descent'] = df['Vict Descent'].apply(self._map_descent)
            logger.info("Created simplified victim descent")
        
        return df
    
    def _map_descent(self, desc: str) -> str:
        """Map detailed descent codes to broader categories"""
        if desc == 'W':
            return 'White'
        elif desc == 'B':
            return 'Black'
        elif desc == 'H':
            return 'Hispanic'
        elif desc in ['A', 'C', 'D', 'F', 'G', 'I', 'J', 'K', 'L', 'P', 'S', 'U', 'V', 'Z']:
            return 'Asian'
        else:
            return 'Other'
    
    def create_hotspot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create hotspot detection features
        
        Args:
            df: Input dataframe with crime rate features
            
        Returns:
            pd.DataFrame: Dataframe with hotspot features added
        """
        # Get configuration
        z_threshold = self.config.get('hotspot_detection', {}).get('z_score_threshold', 1.5)
        quantile_threshold = self.config.get('hotspot_detection', {}).get('quantile_threshold', 0.75)
        rolling_window = self.config.get('hotspot_detection', {}).get('rolling_window_hours', 24)
        min_periods = self.config.get('hotspot_detection', {}).get('min_periods', 12)
        
        # Sort by area and datetime
        df = df.sort_values(['AREA NAME', 'Datetime_Key'])
        
        # Daily hotspot (top 25% of districts each day)
        df['is_daily_hotspot'] = df.groupby('Date')['Daily_Crime_Rate_per_1000'].transform(
            lambda x: (x > x.quantile(quantile_threshold)).astype(int)
        )
        
        # Rolling statistics for hourly hotspot detection
        df['rolling_mean'] = (
            df.groupby('AREA NAME')['Hourly_Crime_Rate_per_1000']
            .transform(lambda x: x.rolling(window=rolling_window, min_periods=min_periods).mean())
        )
        df['rolling_std'] = (
            df.groupby('AREA NAME')['Hourly_Crime_Rate_per_1000']
            .transform(lambda x: x.rolling(window=rolling_window, min_periods=min_periods).std())
        )
        
        # Z-score calculation
        df['z_score'] = (
            (df['Hourly_Crime_Rate_per_1000'] - df['rolling_mean']) / df['rolling_std']
        )
        
        # Hourly hotspot based on z-score
        df['is_hourly_hotspot'] = (df['z_score'] > z_threshold).astype(int)
        
        # Monthly hotspot
        df['is_monthly_hotspot'] = df.groupby('YearMonth_Key')['Monthly_Crime_Rate_per_1000'].transform(
            lambda x: (x > x.quantile(quantile_threshold)).astype(int)
        )
        
        # Yearly hotspot
        df['is_yearly_hotspot'] = df.groupby('Year')['Yearly_Crime_Rate_per_1000'].transform(
            lambda x: (x > x.quantile(quantile_threshold)).astype(int)
        )
        
        logger.info("Created hotspot detection features")
        return df
    
    def get_crime_features_list(self) -> List[str]:
        """Get list of all crime features created"""
        return [
            'Crime Category',
            'Crime Severity',
            'Simplified Vict Descent',
            'is_daily_hotspot',
            'is_hourly_hotspot',
            'is_monthly_hotspot',
            'is_yearly_hotspot',
            'rolling_mean',
            'rolling_std',
            'z_score'
        ]