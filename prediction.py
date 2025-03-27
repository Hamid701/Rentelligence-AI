import pickle
import numpy as np
import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RentalPricePredictor:
    def __init__(self, model_path, preprocessor_path):
        """
        Initialize the predictor with model and preprocessor paths.
        
        Args:
            model_path (str): Path to the pickled XGBoost model
            preprocessor_path (str): Path to the pickled preprocessor
        """
        self.model = None
        self.preprocessor = None
        
        # Load the model
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise ValueError(f"Failed to load model: {str(e)}")
        
        # Load the preprocessor
        try:
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            logger.info(f"Preprocessor loaded successfully from {preprocessor_path}")
            
            # Verify the preprocessor is not a string
            if isinstance(self.preprocessor, str):
                logger.error("Preprocessor was loaded as a string instead of a transformer object")
                raise TypeError("Preprocessor is a string, not a transformer object")
        except Exception as e:
            logger.error(f"Error loading preprocessor from {preprocessor_path}: {str(e)}")
            raise ValueError(f"Failed to load preprocessor: {str(e)}")
    
    def predict(self, features):
        """
        Make a prediction based on input features.
        
        Args:
            features (dict): Dictionary containing feature names and values
        
        Returns:
            float: Predicted rental price
        """
        if not self.model or not self.preprocessor:
            raise ValueError("Model or preprocessor not loaded properly")
        
        try:
            # Convert features dictionary to DataFrame with a single row
            features_df = pd.DataFrame([features])
            
            # Ensure categorical features are strings
            for col in features_df.columns:
                if col in ['region', 'city', 'region_standardized']:
                    features_df[col] = features_df[col].astype(str)
            
            # Convert boolean features to integers (0/1)
            bool_cols = [col for col in features_df.columns if col.startswith('has_') or col.startswith('is_')]
            for col in bool_cols:
                if col in features_df.columns:
                    features_df[col] = features_df[col].astype(int)
            
            # Apply preprocessing
            X_processed = self.preprocessor.transform(features_df)
            
            # Make prediction
            prediction = self.model.predict(X_processed)[0]
            
            return prediction
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Re-raise with more context
            raise RuntimeError(f"Prediction failed: {str(e)}")