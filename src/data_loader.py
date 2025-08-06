"""
Data Loader Module
Handles loading and initial exploration of pet adoption datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Handles loading and initial exploration of pet adoption datasets."""
    
    def __init__(self, data_path: str = "data"):
        """
        Initialize DataLoader.
        
        Args:
            data_path (str): Path to data directory
        """
        self.data_path = Path(data_path)
        self.king_county_df = None
        self.montgomery_df = None
    
    def _log_loading_results(self):
        """Log dataset loading results for debugging."""
        king_count = len(self.king_county_df) if self.king_county_df is not None else 0
        montgomery_count = len(self.montgomery_df) if self.montgomery_df is not None else 0
        total_count = king_count + montgomery_count
        
        # Store results for potential logging/debugging
        self._loading_stats = {
            'king_county_records': king_count,
            'montgomery_records': montgomery_count,
            'total_records': total_count
        }
        
    def load_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both datasets from CSV files.
        
        Returns:
            tuple: (king_county_df, montgomery_df)
        """
        try:
            # Load King County dataset
            king_county_path = self.data_path / "KingCountyWA-AdoptablePets.csv"
            self.king_county_df = pd.read_csv(king_county_path)
            
            # Load Montgomery County dataset
            montgomery_path = self.data_path / "MontgomeryMD-AdoptablePets.csv"
            self.montgomery_df = pd.read_csv(montgomery_path)
            
            # Log dataset loading results
            self._log_loading_results()
            
            return self.king_county_df, self.montgomery_df
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading datasets: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")
    
    def get_dataset_info(self) -> dict:
        """
        Get basic information about the datasets.
        
        Returns:
            dict: Dictionary with dataset information
        """
        if self.king_county_df is None or self.montgomery_df is None:
            self.load_datasets()
        
        info = {
            'king_county': {
                'records': len(self.king_county_df),
                'columns': list(self.king_county_df.columns),
                'missing_values': self.king_county_df.isnull().sum().to_dict(),
                'data_types': self.king_county_df.dtypes.to_dict()
            },
            'montgomery': {
                'records': len(self.montgomery_df),
                'columns': list(self.montgomery_df.columns),
                'missing_values': self.montgomery_df.isnull().sum().to_dict(),
                'data_types': self.montgomery_df.dtypes.to_dict()
            }
        }
        
        return info
    
    def print_dataset_info(self):
        """Print detailed information about the datasets."""
        info = self.get_dataset_info()
        
        print("Dataset Information")
        print("=" * 50)
        
        for dataset_name, data in info.items():
            print(f"\n{dataset_name.replace('_', ' ').title()}:")
            print(f"  Records: {data['records']:,}")
            print(f"  Columns: {len(data['columns'])}")
            print(f"  Missing Values: {sum(data['missing_values'].values())}")
            
            print(f"  Column Names:")
            for col in data['columns']:
                print(f"    - {col}") 