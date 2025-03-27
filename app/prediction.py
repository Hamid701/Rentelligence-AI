import os
import pickle
import numpy as np
import pandas as pd

class RentalPricePredictor:
    """
    A class to handle rental price predictions using a trained XGBoost model.
    This class loads the model and preprocessor, and provides methods for making predictions.
    """
    
    def __init__(self, model_path, preprocessor_path):
        
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self._load_model()
        self._load_preprocessor()
    
    def _load_model(self):
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise Exception(f"Failed to load model from {self.model_path}: {str(e)}")
    
    def _load_preprocessor(self):
        
        try:
            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
        except Exception as e:
            raise Exception(f"Failed to load preprocessor from {self.preprocessor_path}: {str(e)}")
    
    def preprocess_features(self, features):
        
        try:
            # Convert the features dictionary to a DataFrame
            features_df = pd.DataFrame([features])
            
            # Apply the preprocessor
            X_processed = self.preprocessor.transform(features_df)
            
            return X_processed
        except Exception as e:
            raise Exception(f"Error during feature preprocessing: {str(e)}")
    
    def predict(self, features):

        try:
            # Preprocess the features
            X_processed = self.preprocess_features(features)
            
            # Make prediction (this returns log-transformed price)
            log_prediction = self.model.predict(X_processed)[0]
            
            # Apply inverse log transformation to get actual price
            prediction = np.expm1(log_prediction)
            
            # Ensure prediction is within reasonable bounds
            prediction = max(200, min(10000, prediction))
            
            return prediction
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")
    
    def predict_with_confidence(self, features):

        try:
            # Get the base prediction
            prediction = self.predict(features)
            
            # Apply a simple confidence interval (±15%)
            # In a production system, this would be more sophisticated
            lower_bound = prediction * 0.85
            upper_bound = prediction * 1.15
            
            return prediction, lower_bound, upper_bound
        except Exception as e:
            raise Exception(f"Error calculating prediction with confidence: {str(e)}")