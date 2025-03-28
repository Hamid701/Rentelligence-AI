# 🏠 Rentelligence AI: Predicting Italian Rental Prices

<table>
  <tr>
    <td style="vertical-align: top; width: 60%;">
      A machine learning application that predicts rental prices for properties across Italy based on location, property size, amenities, and other key features. Rentelligence AI helps users make informed decisions when navigating the Italian rental market. Rentelligence AI was born from a personal need to understand fair rental prices when moving to Italy. What started as a personal quest evolved into a comprehensive analysis of the Italian rental market, revealing fascinating patterns about how location, timing, and property features influence rental prices across the country.
    </td>
    <td style="vertical-align: top; width: 40%; text-align: center;">
      <img src="./images/header.jpg" alt="Italian Rental Market" width="2500px" style="border-radius: 10px; max-width: 100%;">
    </td>
  </tr>
</table>

## 📚 Project Blog

Check out our detailed [project blog](./blog.md) for an in-depth journey through the development process, data insights, and technical challenges of building Rentelligence AI.

## 🚀 Features

- Interactive map showing average rental prices by region in Italy
- Personalized price predictions based on detailed property features
- Geographic data visualization of Italy's rental landscape
- User-friendly interface built with Streamlit for easy exploration

## 💻 Tech Stack

- Python (pandas, numpy, scikit-learn, XGBoost)
- Streamlit for the web application
- GeoPandas and Folium for geographic visualization
- Docker for containerization

## 📊 Data Overview

The project is built on a comprehensive dataset of over 12,6267 Italian rental listings, featuring:
- Location data (region, city, neighborhood)
- Property features (area, bedrooms, bathrooms)
- Building characteristics (floor, elevator access)
- Amenities (balcony, parking, furnishings)
- Price information

## 🛠️ How to Run Locally

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app/app.py`

## 🌐 Deployment

This app is deployed on Streamlit Cloud and available at [https://rentelligence-ai.streamlit.app](https://rentelligence-ai.streamlit.app)

## 🔮 Future Directions

- Expanding the dataset with more recent listings
- Adding time-based predictions for seasonal price variations
- Incorporating more detailed neighborhood-level data
- Developing comparative analysis between cities


