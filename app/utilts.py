import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def format_feature_name(feature_name):
    """
    Format feature names for display
    """
    # Remove prefixes from preprocessor feature names
    if '__' in feature_name:
        feature_name = feature_name.split('__')[1]
    
    # Replace underscores with spaces and capitalize
    feature_name = feature_name.replace('_', ' ').title()
    
    # Special case handling
    replacements = {
        'Has ': 'Has ',
        'Is ': 'Is ',
        'Log Area': 'Area (log)',
        'Num ': 'Number of '
    }
    
    for old, new in replacements.items():
        feature_name = feature_name.replace(old, new)
        
    return feature_name

def create_price_distribution_chart(data, region=None):
    """
    Create a price distribution chart for a specific region or all regions
    """
    plt.figure(figsize=(10, 6))
    
    if region:
        region_data = data[data['region_standardized'] == region]
        sns.histplot(region_data['price'], kde=True)
        plt.title(f'Price Distribution in {region}')
    else:
        sns.histplot(data['price'], kde=True)
        plt.title('Price Distribution Across Italy')
    
    plt.xlabel('Monthly Rent (€)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    return plt