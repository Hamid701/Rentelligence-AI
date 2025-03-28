import os
import pickle
import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RentalPricePredictor:
    def __init__(self, model_path, preprocessor_path):
        """
        Initialize the predictor with model and preprocessor paths.
        """
        try:
            with open(model_path, 'rb') as model_file:
                self.model = pickle.load(model_file)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ValueError(f"Failed to load model: {str(e)}")
            
        try:
            with open(preprocessor_path, 'rb') as preprocessor_file:
                self.preprocessor = pickle.load(preprocessor_file)
            logger.info(f"Preprocessor loaded successfully from {preprocessor_path}")
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise ValueError(f"Failed to load preprocessor: {str(e)}")
        
        # Define feature groups based on training
        self.numeric_features = [
            'log_area', 'bathrooms', 'floor_to_height_ratio', 
            'total_floors', 'parking_spaces', 'luxury_score'
        ]
        self.binary_features = [
            'has_elevator', 'has_doorman', 'has_balcony', 
            'has_external_exposure', 'has_furnished'
        ]
        self.categorical_features = ['region_standardized', 'city']
        
        # Define rare amenities for luxury score calculation
        self.rare_amenities = [
            'has_terrace', 'has_garden', 'has_air_conditioning', 
            'has_storage_room', 'has_cellar'
        ]

    def preprocess_features(self, features):
        """
        Prepare features for prediction by ensuring they match the expected format.
        """
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Calculate luxury score as sum of rare amenities
        luxury_score = 0
        for amenity in self.rare_amenities:
            if amenity in features_df.columns and features_df[amenity].iloc[0] == 1:
                luxury_score += 1
        
        features_df['luxury_score'] = luxury_score
        logger.info(f"Calculated luxury_score: {luxury_score}")
        
        # Keep only the features used in training
        required_features = self.numeric_features + self.binary_features + self.categorical_features
        
        # Ensure all required features are present
        for feature in required_features:
            if feature not in features_df.columns:
                logger.warning(f"Missing feature: {feature}, setting to default value")
                if feature in self.numeric_features:
                    features_df[feature] = 0
                elif feature in self.binary_features:
                    features_df[feature] = 0
                else:
                    features_df[feature] = "unknown"
        
        # Select only the required features in the correct order
        features_df = features_df[required_features]
        
        logger.info(f"Preprocessed features: {features_df.to_dict(orient='records')[0]}")
        return features_df

    def predict(self, features):
        """
        Make a prediction based on input features.
        """
        try:
            # Preprocess features
            features_df = self.preprocess_features(features)
            
            # Apply the column transformer
            X_processed = self.preprocessor.transform(features_df)
            
            # Make prediction (model was trained on log_price)
            log_prediction = self.model.predict(X_processed)[0]
            
            # Convert from log scale back to original scale
            prediction = np.exp(log_prediction)
            
            logger.info(f"Final prediction: {prediction}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
            
    def predict_with_confidence(self, features):
        """
        Make a prediction with confidence bounds.
        """
        predicted_price = self.predict(features)
        
        # Calculate confidence bounds (10% range)
        lower_bound = predicted_price * 0.9
        upper_bound = predicted_price * 1.1
        
        return predicted_price, lower_bound, upper_bound