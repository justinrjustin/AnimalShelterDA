# Adoptable Pets Data Analysis Project

```
AdoptablePetsDA/
├── data/
│   ├── KingCountyWA-AdoptablePets.csv
│   └── MontgomeryMD-AdoptablePets.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading and exploration
│   ├── data_cleaner.py     # Data cleaning and preprocessing
│   ├── data_analyzer.py    # Statistical analysis and metrics
│   └── visualizer.py       # Visualization and plotting
├── notebooks/
│   └── AdoptablePets_Analysis.ipynb  # Main analysis notebook
├── requirements.txt
└── README.md
```

### Analysis Objectives

1. **Pet Recovery Analysis**: King County LOST/FOUND/ADOPTABLE status tracking
2. **Pet Type Distribution**: Most common animals in shelter systems
3. **Adoption Fee Analysis**: Real adoption costs by animal type
4. **Geographic Patterns**: Regional differences between WA and MD
5. **Age Demographics**: Age distribution of pets
6. **Gender Analysis**: Male vs female pet statistics

## Data Sources

This project analyzes pet adoption data from official government sources:

- **King County, WA**: `data/KingCountyWA-AdoptablePets.csv` (563 records)
  - Source: [Lost, found, adoptable pets](https://catalog.data.gov/dataset/lost-found-adoptable-pets) from King County, Washington
  - Publisher: data.kingcounty.gov
  - Update Frequency: Real-time animal shelter data

- **Montgomery County, MD**: `data/MontgomeryMD-AdoptablePets.csv` (95 records)
  - Source: [Adoptable Pets](https://catalog.data.gov/dataset/adoptable-pets) from Montgomery County, Maryland
  - Publisher: data.montgomerycountymd.gov
  - Update Frequency: Every two hours

The datasets contain anonymized pet records with no personal information. All data is publicly available through [Data.gov](https://data.gov) and used for educational/analytical purposes.

## Technical Stack

- **Python 3.8+**
- **Modular Architecture**: Clean separation of concerns
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive dashboards
- **Scikit-learn**: Machine learning insights
- **Jupyter Notebook**: Interactive analysis environment

## Professional Features

### Modular Architecture
- **Separation of Concerns**: Logic separated from visualization
- **Reusable Components**: Each module can be used independently
- **Clean Code**: Professional documentation and error handling
- **Scalable Design**: Easy to extend and modify

### Data Processing
- **Automated data cleaning** and standardization
- **Missing value handling** with robust strategies
- **Date parsing** and validation
- **Categorical variable encoding**

### Analysis Capabilities
- **Pivot table creation** (Excel-like functionality)
- **Statistical hypothesis testing**
- **Correlation analysis**
- **Clustering analysis**

### Visualization Dashboard
- **Interactive charts** with Plotly
- **Professional static visualizations**
- **Geographic analysis**
- **Trend analysis**
- **Robust error handling**

### Key Insights
- **Pet recovery analysis** with real data
- **Adoption fee analysis** from actual records
- **Pet type distribution** analysis
- **Statistical insights** and recommendations

## Sample Outputs

### Key Metrics
- **Total Records**: 658
- **Pet Types**: Dog, Cat, Bird, Rabbit, etc.
- **Geographic Coverage**: King County, WA & Montgomery County, MD

### Key Findings
- **Most Common Pet**: Cats (382 total)
- **Recovery Rate**: Analysis of LOST vs FOUND pets
- **Adoption Fees**: Real cost analysis by animal type
- **Regional Differences**: WA vs MD pet demographics

## Contributing

This is a portfolio project demonstrating data analysis skills. Feel free to:
- Fork the repository
- Submit issues for bugs or improvements
- Create pull requests for enhancements

## License

This project is for educational and portfolio purposes. The data sources are publicly available.




