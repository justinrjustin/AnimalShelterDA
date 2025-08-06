"""
Data Cleaner Module
Handles data cleaning, standardization, and preprocessing for pet adoption datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class DataCleaner:
    """Handles data cleaning and standardization for pet adoption datasets."""
    
    def __init__(self):
        """Initialize DataCleaner."""
        self.king_clean = None
        self.montgomery_clean = None
        self.combined_df = None
    
    def _extract_age_numeric(self, age_series: pd.Series) -> pd.Series:
        """
        Safely extract numeric age values from age strings.
        
        Args:
            age_series (pd.Series): Series containing age strings
            
        Returns:
            pd.Series: Series with numeric age values
        """
        age_extracted = age_series.str.extract(r'(\d+)')[0]
        return pd.to_numeric(age_extracted, errors='coerce')
    
    def _standardize_gender_data(self, gender_series: pd.Series) -> pd.Series:
        """
        Standardize gender data by mapping single-letter codes to full descriptions.
        
        Args:
            gender_series (pd.Series): Series containing gender data
            
        Returns:
            pd.Series: Standardized gender data
        """
        # First convert to string and title case
        gender_series = gender_series.astype(str).str.title()
        
        # Create mapping for single-letter codes to full descriptions
        gender_mapping = {
            'S': 'Spayed Female',
            'N': 'Neutered Male', 
            'M': 'Male',
            'F': 'Female',
            'U': 'Unknown'
        }
        
        # Apply the mapping
        gender_series = gender_series.replace(gender_mapping)
        
        return gender_series
    
    def _apply_common_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply common cleaning steps to both datasets.
        
        Args:
            df (pd.DataFrame): Dataset to clean
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        # Clean animal_type column
        df['animal_type'] = df['animal_type'].str.title()
        
        # Clean age column - extract numeric values
        df['age_clean'] = self._extract_age_numeric(df['age'])
        
        # Create age categories
        df['age_category'] = df['age_clean'].apply(self._categorize_age)
        
        # Clean and standardize gender column
        df['animal_gender'] = self._standardize_gender_data(df['animal_gender'])
        
        # Standardize breed names
        df['animal_breed'] = df['animal_breed'].str.title()
        df['animal_breed'] = df['animal_breed'].str.replace('Rex', 'Rex Rabbit', case=False)
        
        # Create status column based on record_type
        df['status'] = df['record_type'].str.title()
        
        return df
    
    def clean_king_county_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize King County dataset.
        
        Args:
            df (pd.DataFrame): Raw King County dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
        df_clean['location'] = 'King County, WA'
        
        # Apply common cleaning steps
        df_clean = self._apply_common_cleaning(df_clean)
        
        # Parse dates
        df_clean['date_parsed'] = pd.to_datetime(df_clean['date'], errors='coerce')
        
        return df_clean
    
    def clean_montgomery_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize Montgomery County dataset.
        
        Args:
            df (pd.DataFrame): Raw Montgomery County dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
        df_clean['location'] = 'Montgomery County, MD'
        
        # Map column names to match King County
        column_mapping = {
            'animal_id': 'animal_id',
            'pet_name': 'animal_name',
            'animal_type': 'animal_type',
            'pet_age': 'age',
            'color': 'animal_color',
            'breed': 'animal_breed',
            'sex': 'animal_gender',
            'intake_type': 'record_type'
        }
        df_clean = df_clean.rename(columns=column_mapping)
        
        # Apply common cleaning steps
        df_clean = self._apply_common_cleaning(df_clean)
        
        # Parse dates
        df_clean['date_parsed'] = pd.to_datetime(df_clean['in_date'], errors='coerce')
        
        return df_clean
    
    def _categorize_age(self, age_val: float) -> str:
        """
        Categorize age values into age groups.
        
        Args:
            age_val (float): Age value
            
        Returns:
            str: Age category
        """
        if pd.isna(age_val):
            return 'Unknown'
        elif age_val < 1:
            return 'Under 1 year'
        elif age_val < 3:
            return '1-3 years'
        elif age_val < 7:
            return '3-7 years'
        else:
            return '7+ years'
    
    def clean_all_datasets(self, king_df: pd.DataFrame, montgomery_df: pd.DataFrame):
        """
        Clean both datasets and combine them.
        
        Args:
            king_df (pd.DataFrame): Raw King County dataset
            montgomery_df (pd.DataFrame): Raw Montgomery County dataset
            
        Returns:
            tuple: (king_clean, montgomery_clean, combined_df)
        """
        # Clean individual datasets
        self.king_clean = self.clean_king_county_data(king_df)
        self.montgomery_clean = self.clean_montgomery_data(montgomery_df)
        
        # Combine datasets
        self.combined_df = pd.concat([self.king_clean, self.montgomery_clean], ignore_index=True)
        
        return self.king_clean, self.montgomery_clean, self.combined_df
    
    def assess_data_quality(self, df: pd.DataFrame, location_name: str):
        """
        Assess data quality for a given dataset.
        
        Args:
            df (pd.DataFrame): Dataset to assess
            location_name (str): Name of the location
            
        Returns:
            tuple: (missing_data, missing_percent)
        """
        print(f"\nData Quality Assessment: {location_name}")
        print("=" * 50)
        
        # Missing values analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        print("\nMissing Values:")
        for col, missing, percent in zip(df.columns, missing_data, missing_percent):
            if missing > 0:
                print(f"  {col}: {missing:,} ({percent:.1f}%)")
        
        # Key column analysis
        print(f"\nKey Column Analysis:")
        print(f"  Animal Types: {df['animal_type'].nunique()} unique values")
        print(f"  Breeds: {df['animal_breed'].nunique()} unique values")
        print(f"  Statuses: {df['status'].nunique()} unique values")
        
        return missing_data, missing_percent
    
    def get_cleaned_data(self):
        """
        Get cleaned datasets.
        
        Returns:
            tuple: (king_clean, montgomery_clean, combined_df)
        """
        if self.combined_df is None:
            raise ValueError("Data not cleaned yet. Call clean_all_datasets() first.")
        
        return self.king_clean, self.montgomery_clean, self.combined_df 