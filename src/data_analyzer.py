"""
Data Analyzer Module
Handles statistical analysis, pivot tables, and business metrics calculations.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


class DataAnalyzer:
    """Handles statistical analysis and business metrics calculations."""
    
    def __init__(self, combined_df: pd.DataFrame):
        """
        Initialize DataAnalyzer.
        
        Args:
            combined_df (pd.DataFrame): Combined cleaned dataset
        """
        self.combined_df = combined_df
        self.king_metrics = None
        self.montgomery_metrics = None
        self.combined_metrics = None
    
    def create_pivot_tables(self):
        """
        Create pivot tables for analysis.
        
        Returns:
            dict: Dictionary containing all pivot tables
        """
        print("Creating pivot tables...")
        
        # Pivot Table 1: Pet Type by Location and Status
        pivot_pet_type = pd.pivot_table(
            self.combined_df,
            values='animal_id',
            index=['location', 'animal_type'],
            columns='status',
            aggfunc='count',
            fill_value=0
        )
        
        # Pivot Table 2: Age Distribution by Location
        pivot_age = pd.pivot_table(
            self.combined_df,
            values='animal_id',
            index='location',
            columns='age_category',
            aggfunc='count',
            fill_value=0
        )
        
        # Pivot Table 3: Gender Analysis by Pet Type
        pivot_gender = pd.pivot_table(
            self.combined_df,
            values='animal_id',
            index='animal_type',
            columns='animal_gender',
            aggfunc='count',
            fill_value=0
        )
        
        pivot_tables = {
            'pet_type_by_location_status': pivot_pet_type,
            'age_distribution_by_location': pivot_age,
            'gender_analysis_by_pet_type': pivot_gender
        }
        
        print("Pivot tables created successfully!")
        return pivot_tables
    
    def calculate_key_metrics(self, df: pd.DataFrame) -> dict:
        """
        Calculate key business metrics for a dataset.
        
        Args:
            df (pd.DataFrame): Dataset to analyze
            
        Returns:
            dict: Dictionary containing key metrics
        """
        metrics = {}
        
        # Total records
        metrics['total_records'] = len(df)
        
        # Pet type distribution
        pet_type_counts = df['animal_type'].value_counts()
        metrics['pet_type_distribution'] = pet_type_counts
        
        # Status distribution
        status_counts = df['status'].value_counts()
        metrics['status_distribution'] = status_counts
        
        # Recovery rate (if applicable)
        if 'Found' in status_counts and 'Lost' in status_counts:
            recovery_rate = (status_counts['Found'] / (status_counts['Found'] + status_counts['Lost'])) * 100
            metrics['recovery_rate'] = recovery_rate
        
        # Age distribution
        age_counts = df['age_category'].value_counts()
        metrics['age_distribution'] = age_counts
        
        # Gender distribution
        gender_counts = df['animal_gender'].value_counts()
        metrics['gender_distribution'] = gender_counts
        
        return metrics
    
    def calculate_all_metrics(self, king_clean: pd.DataFrame, montgomery_clean: pd.DataFrame):
        """
        Calculate metrics for all datasets.
        
        Args:
            king_clean (pd.DataFrame): Cleaned King County data
            montgomery_clean (pd.DataFrame): Cleaned Montgomery County data
            
        Returns:
            dict: Dictionary containing all metrics
        """
        print("Calculating key metrics...")
        
        self.king_metrics = self.calculate_key_metrics(king_clean)
        self.montgomery_metrics = self.calculate_key_metrics(montgomery_clean)
        self.combined_metrics = self.calculate_key_metrics(self.combined_df)
        
        all_metrics = {
            'king_county': self.king_metrics,
            'montgomery_county': self.montgomery_metrics,
            'combined': self.combined_metrics
        }
        
        print("Metrics calculated successfully!")
        return all_metrics
    
    def perform_statistical_analysis(self):
        """
        Perform statistical analysis on the data.
        
        Returns:
            dict: Dictionary containing statistical results
        """
        print("Performing statistical analysis...")
        
        results = {}
        
        # 1. Chi-square test for independence between location and pet type
        contingency_table = pd.crosstab(self.combined_df['location'], self.combined_df['animal_type'])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        results['chi_square_test'] = {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant_difference': p_value < 0.05
        }
        
        # 2. Correlation analysis for numeric variables
        numeric_df = self.combined_df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            correlation_matrix = numeric_df.corr()
            results['correlation_analysis'] = correlation_matrix
        
        # 3. Summary statistics by location
        location_summary = self.combined_df.groupby('location').agg({
            'animal_id': 'count',
            'age_clean': ['mean', 'median', 'std'],
            'animal_type': 'nunique',
            'animal_breed': 'nunique'
        }).round(2)
        
        results['location_summary'] = location_summary
        
        print("Statistical analysis completed!")
        return results
    
    def perform_clustering_analysis(self, n_clusters: int = 3):
        """
        Perform clustering analysis on the data.
        
        Args:
            n_clusters (int): Number of clusters to create
            
        Returns:
            dict: Dictionary containing clustering results
        """
        print("Performing clustering analysis...")
        
        try:
            # Prepare data for clustering
            features = ['animal_type', 'animal_gender', 'age_category', 'location']
            encoded_df = self.combined_df[features].copy()
            
            # Encode categorical variables
            le = LabelEncoder()
            for col in features:
                encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(encoded_df)
            
            # Add clusters to original data
            self.combined_df['cluster'] = clusters
            
            # Analyze cluster characteristics
            cluster_analysis = {}
            for cluster in range(n_clusters):
                cluster_data = self.combined_df[self.combined_df['cluster'] == cluster]
                cluster_analysis[f'cluster_{cluster}'] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(self.combined_df) * 100,
                    'top_pet_type': cluster_data['animal_type'].mode().iloc[0],
                    'top_location': cluster_data['location'].mode().iloc[0]
                }
            
            results = {
                'n_clusters': n_clusters,
                'cluster_distribution': self.combined_df['cluster'].value_counts().sort_index().to_dict(),
                'cluster_characteristics': cluster_analysis
            }
            
            print("Clustering analysis completed!")
            return results
            
        except Exception as e:
            print(f"Clustering analysis could not be completed: {str(e)}")
            return None
    
    def generate_executive_summary(self):
        """
        Generate executive summary of findings.
        
        Returns:
            dict: Dictionary containing executive summary
        """
        print("Generating executive summary...")
        
        summary = {
            'total_records': len(self.combined_df),
            'geographic_coverage': self.combined_df['location'].nunique(),
            'pet_types_represented': self.combined_df['animal_type'].nunique(),
            'breed_diversity': self.combined_df['animal_breed'].nunique(),
            'most_common_pet_type': self.combined_df['animal_type'].mode().iloc[0],
            'most_common_breed': self.combined_df['animal_breed'].mode().iloc[0],
            'gender_distribution': self.combined_df['animal_gender'].value_counts().to_dict()
        }
        
        # Regional differences
        regional_differences = {}
        for location in self.combined_df['location'].unique():
            location_data = self.combined_df[self.combined_df['location'] == location]
            regional_differences[location] = {
                'records': len(location_data),
                'top_pet_type': location_data['animal_type'].mode().iloc[0],
                'top_breed': location_data['animal_breed'].mode().iloc[0]
            }
        
        summary['regional_differences'] = regional_differences
        
        print("Executive summary generated!")
        return summary
    
    def get_analysis_results(self):
        """
        Get all analysis results.
        
        Returns:
            dict: Dictionary containing all analysis results
        """
        if self.combined_metrics is None:
            raise ValueError("Analysis not performed yet. Call calculate_all_metrics() first.")
        
        return {
            'metrics': {
                'king_county': self.king_metrics,
                'montgomery_county': self.montgomery_metrics,
                'combined': self.combined_metrics
            },
            'pivot_tables': self.create_pivot_tables(),
            'statistical_analysis': self.perform_statistical_analysis(),
            'clustering_analysis': self.perform_clustering_analysis(),
            'executive_summary': self.generate_executive_summary()
        } 