"""
Visualizer Module
Handles all plotting and visualization logic with robust error handling.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re


class Visualizer:
    """Handles all visualization logic with robust error handling."""
    
    def __init__(self, combined_df: pd.DataFrame):
        """
        Initialize Visualizer.
        
        Args:
            combined_df (pd.DataFrame): Combined cleaned dataset
        """
        self.combined_df = combined_df
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup plotting style for consistent visualizations."""
        plt.style.use('default')
        sns.set_palette("husl")
    
    
    
    def create_plotly_dashboard(self):
        """
        Create interactive dashboard with Plotly (excluding Pet Type Distribution).
        
        Returns:
            plotly.graph_objects.Figure: The dashboard figure
        """
        try:
            from plotly.subplots import make_subplots
            
            # Create subplots - 3 rows, 2 columns for better spacing
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Status by Location', 'Age Distribution', 
                              'Top 10 Breeds', 'Gender Distribution',
                              'Records by Month', 'Breed Analysis'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                     [{"type": "bar"}, {"type": "pie"}],
                     [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # 1. Status Distribution by Location (Bar Chart) - Row 1, Col 1
            status_by_location = self.combined_df.groupby(['location', 'status']).size().unstack(fill_value=0)
            for status in status_by_location.columns:
                fig.add_trace(
                    go.Bar(
                        x=status_by_location.index,
                        y=status_by_location[status],
                        name=status,
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # 2. Age Distribution (Bar Chart) - Row 1, Col 2
            age_counts = self.combined_df['age_category'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=age_counts.index,
                    y=age_counts.values,
                    name="Age Distribution",
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
            
            # 3. Top 10 Breeds (Horizontal Bar Chart) - Row 2, Col 1
            breed_counts = self.combined_df['animal_breed'].value_counts().head(10)
            fig.add_trace(
                go.Bar(
                    x=breed_counts.values,
                    y=breed_counts.index,
                    orientation='h',
                    name="Top Breeds",
                    marker_color='lightgreen'
                ),
                row=2, col=1
            )
            
            # 4. Gender Distribution (Pie Chart) - Row 2, Col 2
            gender_counts = self.combined_df['animal_gender'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=gender_counts.index,
                    values=gender_counts.values,
                    textinfo='label+percent',
                    textposition='outside',
                    name="Gender"
                ),
                row=2, col=2
            )
            
            # 5. Records by Month (Line Chart) - Row 3, Col 1
            monthly_counts = self.combined_df['date_parsed'].dt.to_period('M').value_counts().sort_index()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(monthly_counts))),
                    y=monthly_counts.values,
                    mode='lines+markers',
                    name="Monthly Records",
                    line=dict(color='red', width=2),
                    marker=dict(size=8)
                ),
                row=3, col=1
            )
            
            # 6. Breed Analysis by Location (Bar Chart) - Row 3, Col 2
            breed_by_location = self.combined_df.groupby(['location', 'animal_breed']).size().reset_index(name='count')
            top_breeds = self.combined_df['animal_breed'].value_counts().head(5).index
            breed_by_location_filtered = breed_by_location[breed_by_location['animal_breed'].isin(top_breeds)]
            
            for breed in top_breeds:
                breed_data = breed_by_location_filtered[breed_by_location_filtered['animal_breed'] == breed]
                fig.add_trace(
                    go.Bar(
                        x=breed_data['location'],
                        y=breed_data['count'],
                        name=breed,
                        showlegend=True
                    ),
                    row=3, col=2
                )
            
            # Update layout
            fig.update_layout(
                title_text="Animal Analysis Dashboard",
                title_x=0.5,
                title_font_size=20,
                height=1000,  # Increased height for 3 rows
                showlegend=True
            )
            
            # Update axes labels for 3x2 layout (removed redundant "Location" labels)
            fig.update_xaxes(title_text="Age Category", row=1, col=2)
            fig.update_xaxes(title_text="Count", row=2, col=1)  # Updated for Top 10 Breeds
            fig.update_xaxes(title_text="Month", row=3, col=1)
            fig.update_xaxes(title_text="", row=3, col=2)  # No title for breed analysis
            
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=2, col=1)  # Updated for Top 10 Breeds
            fig.update_yaxes(title_text="Count", row=3, col=1)
            fig.update_yaxes(title_text="Count", row=3, col=2)
            
            return fig
            
        except Exception as e:
            print(f"Error creating Plotly dashboard: {str(e)}")
            return None
    
    def create_pet_type_distribution_chart(self):
        """
        Create a dedicated Pet Type Distribution chart with proper spacing.
        
        Returns:
            plotly.graph_objects.Figure: The pet type distribution figure
        """
        try:
            pet_type_counts = self.combined_df['animal_type'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=pet_type_counts.index,
                values=pet_type_counts.values,
                textinfo='label+percent',
                textposition='outside',
                hole=0.3,  # Add a hole in the middle for better appearance
                marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])
            )])
            
            fig.update_layout(
                title={
                    'text': 'Pet Type Distribution',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                height=600,
                width=800,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating pet type distribution chart: {str(e)}")
            return None
    
    def create_interactive_pet_type_sunburst_chart(self):
        """
        Create interactive pet type analysis with Plotly.
        
        Returns:
            plotly.graph_objects.Figure: The interactive figure
        """
        try:
            # Alternative approach: Create data manually to avoid pandas compatibility issues
            # Get unique combinations and their counts
            location_animal_status = []
            for _, row in self.combined_df.iterrows():
                location_animal_status.append((row['location'], row['animal_type'], row['status']))
            
            # Count occurrences manually
            from collections import Counter
            counts = Counter(location_animal_status)
            
            # Convert to DataFrame format that plotly expects
            data_for_plot = []
            for (location, animal_type, status), count in counts.items():
                data_for_plot.append({
                    'location': location,
                    'animal_type': animal_type,
                    'status': status,
                    'count': count
                })
            
            pet_type_data = pd.DataFrame(data_for_plot)
            
            if pet_type_data.empty:
                raise ValueError("No data available for sunburst chart")
            
            fig = px.sunburst(
                pet_type_data,
                path=['location', 'animal_type', 'status'],
                values='count',
                title='Interactive Pet Type Analysis by Location and Status - Hierarchical breakdown of pet distribution'
            )
            return fig
            
        except Exception as e:
            print(f"Sunburst chart failed: {str(e)}")
            # Alternative: Bar chart with explicit DataFrame conversion
            bar_data = self.combined_df.groupby(['location', 'animal_type']).size().reset_index(name='count')
            bar_data = pd.DataFrame(bar_data)
            bar_data = bar_data.dropna()
            
            fig = px.bar(
                bar_data,
                x='location',
                y='count',
                color='animal_type',
                title='Pet Type Distribution by Location'
            )
            return fig
    
    def create_interactive_age_gender_scatter_plot(self):
        """
        Create interactive age and gender analysis with Plotly.
        
        Returns:
            plotly.graph_objects.Figure: The interactive figure
        """
        try:
            # Filter out rows with missing age data
            age_data = self.combined_df.dropna(subset=['age_clean'])
            
            # Create size mapping for better visualization
            # Group by age, animal_type, and gender to get counts
            size_data = age_data.groupby(['age_clean', 'animal_type', 'animal_gender']).size().reset_index(name='count')
            
            fig = px.scatter(
                size_data,
                x='age_clean',
                y='animal_type',
                color='animal_gender',
                size='count',
                hover_data=['count', 'animal_gender'],
                title='Age Distribution by Pet Type and Gender',
                labels={'count': 'Frequency', 'age_clean': 'Age (years)', 'animal_type': 'Animal Type'}
            )
            
            # Add size legend
            fig.update_layout(
                title_x=0.5,
                title_font_size=16,
                height=600,
                showlegend=True
            )
            
            # Add size legend annotation
            fig.add_annotation(
                text="Circle Size = Number of Animals | Frequency is the number of animals in each age category",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
            
            return fig
            
        except Exception as e:
            print(f"Scatter plot error: {str(e)}")
            print("Creating alternative visualization...")
            
            # Alternative: Box plot
            fig = px.box(
                age_data,
                x='animal_type',
                y='age_clean',
                color='animal_gender',
                title='Age Distribution by Pet Type and Gender'
            )
            return fig
    
    def create_interactive_breed_treemap_chart(self):
        """
        Create interactive breed analysis with Plotly.
        
        Returns:
            plotly.graph_objects.Figure: The interactive figure
        """
        try:
            # Alternative approach: Create breed data manually to avoid pandas compatibility issues
            # Get unique breed-location combinations and their counts
            breed_location = []
            for _, row in self.combined_df.iterrows():
                breed_location.append((row['animal_breed'], row['location']))
            
            # Count occurrences manually
            from collections import Counter
            counts = Counter(breed_location)
            
            # Convert to DataFrame format that plotly expects
            data_for_plot = []
            for (breed, location), count in counts.items():
                data_for_plot.append({
                    'animal_breed': breed,
                    'location': location,
                    'count': count
                })
            
            breed_analysis = pd.DataFrame(data_for_plot)
            
            # Filter to top breeds for better visualization
            breed_counts = {}
            for _, row in self.combined_df.iterrows():
                breed = row['animal_breed']
                breed_counts[breed] = breed_counts.get(breed, 0) + 1
            
            top_breeds = sorted(breed_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            top_breed_names = [breed for breed, _ in top_breeds]
            
            breed_analysis_filtered = breed_analysis[breed_analysis['animal_breed'].isin(top_breed_names)]
            
            fig = px.treemap(
                breed_analysis_filtered,
                path=['location', 'animal_breed'],
                values='count',
                title='Top 20 Breed Distribution by Location - Hierarchical view of most common breeds'
            )
            return fig
            
        except Exception as e:
            # Alternative: Bar chart for top breeds with explicit DataFrame conversion
            top_breeds = self.combined_df['animal_breed'].value_counts().head(20).index
            top_breeds_data = self.combined_df[self.combined_df['animal_breed'].isin(top_breeds)]
            bar_data = top_breeds_data.groupby(['animal_breed', 'location']).size().reset_index(name='count')
            bar_data = pd.DataFrame(bar_data)
            bar_data = bar_data.dropna()
            
            fig = px.bar(
                bar_data,
                x='animal_breed',
                y='count',
                color='location',
                title='Top 20 Breeds by Location'
            )
            fig.update_xaxes(tickangle=45)
            return fig
    
    def create_all_interactive_charts(self):
        """
        Create all interactive visualizations.
        
        Returns:
            dict: Dictionary containing all interactive figures
        """
        
        visualizations = {
            'pet_type_analysis': self.create_interactive_pet_type_analysis(),
            'age_gender_analysis': self.create_interactive_age_gender_analysis(),
            'breed_analysis': self.create_interactive_breed_analysis()
        }
        
        return visualizations
    
    def create_king_county_recovery_and_pet_type_charts(self, metrics: dict):
        """
        Create custom visualizations based on metrics.
        
        Args:
            metrics (dict): Dictionary containing analysis metrics
            
        Returns:
            dict: Dictionary containing custom figures
        """        
        custom_figs = {}
        
        try:
            # Recovery rate visualization based on King County record_type data
            if 'combined' in metrics and 'recovery_rate' in metrics['combined']:
                # Get King County data specifically
                king_county_data = self.combined_df[self.combined_df['location'] == 'King County, WA']
                
                # Count pets by record_type
                record_type_counts = king_county_data['record_type'].value_counts()
                
                # Get specific counts
                lost_count = record_type_counts.get('LOST', 0)
                found_count = record_type_counts.get('FOUND', 0)
                adoptable_count = record_type_counts.get('ADOPTABLE', 0)
                
                # Calculate total and percentages
                total_pets = lost_count + found_count + adoptable_count
                lost_percentage = (lost_count / total_pets) * 100 if total_pets > 0 else 0
                found_percentage = (found_count / total_pets) * 100 if total_pets > 0 else 0
                adoptable_percentage = (adoptable_count / total_pets) * 100 if total_pets > 0 else 0
                
                # Create pie chart
                fig = go.Figure(data=[go.Pie(
                    labels=['LOST', 'FOUND', 'ADOPTABLE'],
                    values=[lost_count, found_count, adoptable_count],
                    hole=0.3,
                    marker_colors=['red', 'green', 'blue'],
                    textinfo='label+value',
                    textposition='outside'
                )])
                
                fig.update_layout(
                    title={
                        'text': f"King County, WA - Pet Status Distribution<br>LOST: {lost_percentage:.1f}% | FOUND: {found_percentage:.1f}% | ADOPTABLE: {adoptable_percentage:.1f}%",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 14}
                    },
                    annotations=[
                        dict(
                            text=f"Total Pets: {total_pets}<br>LOST: {lost_count} ({lost_percentage:.1f}%)<br>FOUND: {found_count} ({found_percentage:.1f}%)<br>ADOPTABLE: {adoptable_count} ({adoptable_percentage:.1f}%)",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.02, y=0.02,
                            xanchor='left', yanchor='bottom',
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="black",
                            borderwidth=1
                        )
                    ]
                )
                
                custom_figs['king_county_recovery_analysis'] = fig
            

                        
        except Exception as e:
            print(f"Error creating custom visualizations: {str(e)}")
        
        return custom_figs
    
    def display_all_charts_and_dashboards(self, metrics: dict = None):
        """
        Display all visualizations.
        
        Args:
            metrics (dict): Optional metrics for custom visualizations
        """        
        # Static dashboard
        static_fig = self.create_static_dashboard()
        if static_fig:
            plt.show()
        
        # Interactive visualizations
        interactive_figs = self.create_all_interactive_charts()
        for name, fig in interactive_figs.items():
            if fig:
                fig.show()
        
        # Custom visualizations
        if metrics:
            custom_figs = self.create_king_county_recovery_and_pet_type_charts(metrics)
            for name, fig in custom_figs.items():
                if fig:
                    fig.show()
    
    def save_all_charts_to_files(self, output_path: str = "output"):
        """
        Save all visualizations to files.
        
        Args:
            output_path (str): Path to save visualizations
        """
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Save static dashboard
        static_fig = self.create_static_overview_dashboard()
        if static_fig:
            static_fig.savefig(f"{output_path}/static_dashboard.png", dpi=300, bbox_inches='tight')
        
        # Save interactive visualizations
        interactive_figs = self.create_all_interactive_charts()
        for name, fig in interactive_figs.items():
            if fig:
                fig.write_html(f"{output_path}/{name}.html")
        
    def create_king_county_adoption_fee_bar_chart(self):
        """
        Create adoption fee analysis for King County data.
        
        Returns:
            plotly.graph_objects.Figure: The adoption fee analysis figure
        """
        # Get King County data
        king_county_data = self.combined_df[self.combined_df['location'] == 'King County, WA']
        
        # Extract adoption fees from Memo column
        adoption_fee_data = []
        
        # Check available columns for debugging
        available_columns = list(king_county_data.columns)
        
        for idx, row in king_county_data.iterrows():
            # Check Memo column for adoption fee information
            memo = str(row.get('memo', '')).lower()
            if 'adoption fee' in memo:
                # Extract animal type
                animal_type = row.get('animal_type', 'Unknown')
                
                # Try to extract fee amount (look for $XX or XX dollars)
                fee_match = re.search(r'\$(\d+)', memo)
                if fee_match:
                    fee_amount = int(fee_match.group(1))
                    adoption_fee_data.append({
                        'animal_type': animal_type,
                        'fee': fee_amount,
                        'description': memo[:100] + '...' if len(memo) > 100 else memo
                    })
        
        animals_with_fees = len(adoption_fee_data)
        
        if adoption_fee_data:
            # Create DataFrame for analysis
            fee_df = pd.DataFrame(adoption_fee_data)
            
            # Calculate statistics
            avg_fee = fee_df['fee'].mean()
            fee_by_type = fee_df.groupby('animal_type')['fee'].agg(['mean', 'count']).round(2)
            
            # Create bar chart of fees by animal type
            fig_fee = go.Figure(data=[
                go.Bar(
                    x=fee_by_type.index,
                    y=fee_by_type['mean'],
                    text=fee_by_type['mean'].round(2),
                    textposition='auto',
                    marker_color=['red', 'green', 'blue', 'orange', 'purple'][:min(len(fee_by_type), 5)]
                )
            ])
            
            fig_fee.update_layout(
                title={
                    'text': f"King County, WA Adoption Fees by Animal Type<br>Average Fee: ${avg_fee:.2f}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 14}
                },
                xaxis_title="Animal Type",
                yaxis_title="Average Adoption Fee ($)",
                annotations=[
                    dict(
                        text=f"Total Animals with Fees: {len(fee_df)}<br>Overall Average: ${avg_fee:.2f}<br>Fee Range: ${fee_df['fee'].min()}-${fee_df['fee'].max()}",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.02, y=0.02,
                        xanchor='left', yanchor='bottom',
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="black",
                        borderwidth=1
                    )
                ]
            )
            
            return fig_fee
        else:
            return None 