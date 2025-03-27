import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import os
import sys
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from prediction import RentalPricePredictor

# Page config
st.set_page_config(
    page_title="Italian Rental Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
    }
    .info-text {
        font-size: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Region mapping
region_mapping = {
    'lombardia': 'Lombardia',
    'piemonte': 'Piemonte',
    'veneto': 'Veneto',
    'toscana': 'Toscana',
    'friuli-venezia-giulia': 'Friuli-Venezia Giulia',
    'umbria': 'Umbria',
    'emilia-romagna': 'Emilia-Romagna',
    'emilia-Romagna': 'Emilia-Romagna',
    'liguria': 'Liguria',
    'Liguria': 'Liguria',
    'trentino-alto-adige': 'Trentino-Alto Adige/Südtirol',
    'calabria': 'Calabria',
    'lazio': 'Lazio',
    'puglia': 'Puglia',
    'campania': 'Campania',
    'sicilia': 'Sicilia',
    'marche': 'Marche',
    'abruzzo': 'Abruzzo',
    'molise': 'Molise',
    'basilicata': 'Basilicata',
    'sardegna': 'Sardegna',
    'valle-d-aosta': "Valle d'Aosta/Vallée d'Aoste",
    
    # Cities mapped to their regions
    'milano': 'Lombardia',
    'torino': 'Piemonte',
    'genova': 'Liguria',
    'napoli': 'Campania',
    'palermo': 'Sicilia',
    'bari': 'Puglia',
    'ancona': 'Marche',
    'catanzaro': 'Calabria',
    'l-aquila': 'Abruzzo',
    'trieste': 'Friuli-Venezia Giulia',
    'venezia': 'Veneto'
}

# Initialize the predictor
@st.cache_resource
def load_predictor():
    model_path = os.path.join(project_root, 'models', 'best_xgb_model.pkl')
    preprocessor_path = os.path.join(project_root, 'models', 'preprocessor.pkl')
    return RentalPricePredictor(model_path, preprocessor_path)

# Load confidence metrics
@st.cache_data
def load_confidence_metrics():
    confidence_path = os.path.join(project_root, 'models', 'confidence_metrics.pkl')
    try:
        with open(confidence_path, 'rb') as f:
            confidence_metrics = pickle.load(f)
        return confidence_metrics
    except:
        return {'confidence_percentage': 82.3}  # Default value based on the model's performance

# Load the regional data for the map
@st.cache_data
def load_map_data():
    regions_gdf = gpd.read_file(os.path.join(project_root, 'data', 'limits_IT_regions.geojson'))
    rental_data = pd.read_csv(os.path.join(project_root, 'data', 'italian_rental_processed.csv'))
    
    rental_data['region_standardized'] = rental_data['region'].map(region_mapping)
    
    region_stats = rental_data.groupby('region_standardized')['price'].agg(['mean', 'median', 'count']).reset_index()
    region_stats.columns = ['reg_name', 'mean_price', 'median_price', 'property_count']
    
    merged_gdf = regions_gdf.merge(
        region_stats, 
        left_on='reg_name',
        right_on='reg_name'
    )
    
    return merged_gdf, rental_data

# Create the interactive map
def create_map(merged_gdf):
    center = [41.8719, 12.5674]  # Center of Italy
    m = folium.Map(location=center, zoom_start=6, tiles='CartoDB positron')
    
    # Add choropleth layer
    folium.Choropleth(
        geo_data=merged_gdf,
        name='Regional Prices',
        data=merged_gdf,
        columns=['reg_name', 'mean_price'],
        key_on='feature.properties.reg_name',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Average Rental Price (€)'
    ).add_to(m)
    
    # Add tooltips for regions
    tooltip = folium.features.GeoJson(
        merged_gdf,
        style_function=lambda x: {'fillColor': '#ffffff', 'color': '#000000', 'fillOpacity': 0.1, 'weight': 0.1},
        control=False,
        highlight_function=lambda x: {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.50, 'weight': 0.1},
        tooltip=folium.features.GeoJsonTooltip(
            fields=['reg_name', 'mean_price', 'median_price', 'property_count'],
            aliases=['Region', 'Average Price (€)', 'Median Price (€)', 'Number of Properties'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    )
    m.add_child(tooltip)
    
    return m

# Make prediction and display results
def make_prediction(features, predictor, confidence_metrics):
    try:
        # Use predict_with_confidence to get prediction with bounds
        predicted_price, lower_bound, upper_bound = predictor.predict_with_confidence(features)
        
        # Display prediction with bounds check
        if predicted_price < 200 or predicted_price > 10000:
            st.warning(f"Predicted price (€{predicted_price:.2f}) is outside the typical range (€200-€10,000). Take this prediction with caution.")
        
        # Create prediction box
        st.markdown('<div class="prediction-box" style="background-color: #E3F2FD;">', unsafe_allow_html=True)
        st.markdown(f"### Estimated Monthly Rent: €{predicted_price:.2f}")
        
        # Display confidence bounds
        st.markdown(f"#### Price Range: €{lower_bound:.2f} - €{upper_bound:.2f}")
        
        # Confidence percentage
        confidence_percentage = confidence_metrics.get('confidence_percentage', 83.64)
        st.markdown(f"#### Prediction Confidence: {confidence_percentage:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(f"Details: {str(e)}")

# Main app 
def main():
    # Load the model and confidence metrics
    try:
        predictor = load_predictor()
        confidence_metrics = load_confidence_metrics()
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False
        predictor = None
        confidence_metrics = None

    # App title and description
    st.markdown('<div class="main-header">🏠 Rentelligence AI</div>', unsafe_allow_html=True)
    st.markdown("""
    Looking for the perfect rental in Italy? We've got you covered! Our smart AI analyzes thousands of properties 
    to help you find fair rental prices anywhere in the country.
    
    Take a tour of the interactive map to see what people are paying in different regions, then fill out the form 
    with your dream home details to get a personalized price estimate.
    """)

    # Layout
    col1, col2 = st.columns([3, 2])

    # Load map data once for both columns
    try:
        merged_gdf, rental_data = load_map_data()
        data_loaded = True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        data_loaded = False
        merged_gdf, rental_data = None, None

    # Map column
    with col1:
        st.markdown('<div class="sub-header">Regional Price Map</div>', unsafe_allow_html=True)
        
        if data_loaded:
            try:
                m = create_map(merged_gdf)
                folium_static(m)
                
                # Map instructions
                st.info("👆 Click on a region to see average rental prices and property counts.")
            except Exception as e:
                st.error(f"Error creating map: {e}")
        else:
            st.info("Please ensure the required data files are available.")

    # Prediction column
    with col2:
        st.markdown('<div class="sub-header">Property Details</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-text">Enter the details of the property to get a price prediction.</div>', unsafe_allow_html=True)
        
        # Get unique regions and cities from the data for the dropdown
        if data_loaded:
            regions = sorted(rental_data['region_standardized'].dropna().unique())
            cities = sorted(rental_data['city'].dropna().unique())
        else:
            regions = ["Lombardia", "Lazio", "Toscana", "Veneto", "Piemonte", "Emilia-Romagna", "Campania", "Sicilia"]
            cities = ["Milano", "Roma", "Firenze", "Venezia", "Torino", "Bologna", "Napoli", "Palermo"]
        
        # Create input form
        with st.form("prediction_form"):
            # Basic property features
            st.markdown("#### Basic Information")
            
            area = st.number_input("Area (m²)", min_value=15, max_value=1000, value=80)
            log_area = np.log1p(area)  # Calculate log_area
            
            num_bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=2)
            bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=1)
            
            # Location
            st.markdown("#### Location")
            region = st.selectbox("Region", regions)
            city = st.selectbox("City", cities)
            
            # Building details
            st.markdown("#### Building Details")
            current_floor = st.number_input("Current Floor", min_value=0, max_value=50, value=2)
            total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=50, value=5)
            
            # Calculate floor_to_height_ratio
            floor_to_height_ratio = current_floor / total_floors if total_floors > 0 else 0
            
            # Parking
            parking_spaces = st.number_input("Parking Spaces", min_value=0, max_value=10, value=0)
            
            # Amenities
            st.markdown("#### Amenities")
            col_a, col_b = st.columns(2)
            with col_a:
                has_elevator = st.checkbox("Elevator")
                has_doorman = st.checkbox("Doorman")
                has_balcony = st.checkbox("Balcony")
            with col_b:
                has_external_exposure = st.checkbox("External Exposure")
                has_furnished = st.checkbox("Furnished")
            
            # Luxury score calculation
            luxury_features = [has_elevator, has_doorman, has_balcony, has_external_exposure, has_furnished]
            luxury_score = sum(luxury_features) / len(luxury_features)
            
            # Submit button
            submitted = st.form_submit_button("Predict Rental Price")
        
        # Make prediction when form is submitted
        if submitted and model_loaded:
            # Prepare features dictionary with all required features
            features = {
                'log_area': log_area,
                'bathrooms': bathrooms,
                'num_bedrooms': num_bedrooms,
                'floor_to_height_ratio': floor_to_height_ratio,
                'total_floors': total_floors,
                'parking_spaces': parking_spaces,
                'has_elevator': int(has_elevator),
                'has_doorman': int(has_doorman),
                'has_balcony': int(has_balcony),
                'has_external_exposure': int(has_external_exposure),
                'has_furnished': int(has_furnished),
                'region_standardized': region,
                'city': city,
                'luxury_score': luxury_score
            }
            
            # Get prediction
            make_prediction(features, predictor, confidence_metrics)

    # Add model information section
    with st.expander("About this Model"):
        st.markdown("""
        ### Meet Your Rental Price Assistant
        
        Behind Rentelligence AI is a smart brain that's learned from thousands of real Italian rental listings. 
        Think of it as a local real estate expert who's analyzed the Italian market inside and out!
        
        Our AI doesn't cut corners, we've designed it to understand the full spectrum of the rental market, 
        from cozy studios to luxury penthouses. This means you get reliable estimates whether you're looking 
        for a budget-friendly apartment or a premium villa with a view.
        
        ### How to Get the Best Results
        
        For the most accurate predictions:
        - Be as precise as possible with measurements
        - Double-check your location details
        - Don't forget to tick all the amenities your dream home has
        
        The more details you provide, the smarter our prediction will be!
        """)

    # Add footer
    st.markdown("---")
    st.markdown("© 2025 Rentelligence AI")

if __name__ == "__main__":
    main()