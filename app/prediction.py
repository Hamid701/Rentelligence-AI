import pickle
import pandas as pd
import numpy as np
import os

class RentalPricePredictor:
    def __init__(self, model_path, preprocessor_path):
        try:
            # Validate paths
            if not isinstance(model_path, str) or not isinstance(preprocessor_path, str):
                raise TypeError(f"Paths must be strings. Model path type: {type(model_path)}, Preprocessor path type: {type(preprocessor_path)}")
                
            print(f"Loading model from: {model_path}")
            print(f"Loading preprocessor from: {preprocessor_path}")
            
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                print(f"Model loaded successfully. Type: {type(self.model)}")
            
            # Load preprocessor with additional error handling
            try:
                # Print current working directory for debugging
                print(f"Current working directory: {os.getcwd()}")
                print(f"Checking if preprocessor path exists: {os.path.exists(preprocessor_path)}")
                
                with open(preprocessor_path, 'rb') as f:
                    self.preprocessor = pickle.load(f)
                    print(f"Preprocessor loaded successfully. Type: {type(self.preprocessor)}")
                    
                # Double check that preprocessor is not a string
                if isinstance(self.preprocessor, str):
                    print(f"Warning: Preprocessor loaded as string. Attempting to fix...")
                    # Try to load the string as a path
                    with open(self.preprocessor, 'rb') as f:
                        self.preprocessor = pickle.load(f)
                        print(f"Preprocessor reloaded successfully. Type: {type(self.preprocessor)}")
            except Exception as e:
                print(f"Error loading preprocessor: {str(e)}")
                # Try multiple alternative paths for Streamlit Cloud
                possible_paths = [
                    os.path.join('models', 'preprocessor.pkl'), 
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'preprocessor.pkl'),  
                    os.path.abspath(os.path.join('.', 'models', 'preprocessor.pkl')),  #
                    os.path.join(os.path.dirname(model_path), 'preprocessor.pkl')  
                ]
                
                for alt_path in possible_paths:
                    try:
                        print(f"Trying alternative path: {alt_path}")
                        print(f"Path exists: {os.path.exists(alt_path)}")
                        with open(alt_path, 'rb') as f:
                            self.preprocessor = pickle.load(f)
                            print(f"Preprocessor loaded successfully from {alt_path}. Type: {type(self.preprocessor)}")
                            break
                    except Exception as inner_e:
                        print(f"Failed to load preprocessor from {alt_path}: {str(inner_e)}")
                
                # If we still don't have a preprocessor, raise the original error
                if not hasattr(self, 'preprocessor'):
                    print("All alternative paths failed. Raising original error.")
                    raise e
                
            # Verify preprocessor has transform method
            if not hasattr(self.preprocessor, 'transform'):
                raise TypeError(f"Loaded preprocessor does not have 'transform' method. Type: {type(self.preprocessor)}")
                
        except Exception as e:
            print(f"Error initializing predictor: {str(e)}")
            print(f"Model path: {model_path}")
            print(f"Preprocessor path: {preprocessor_path}")
            raise
    
    def predict(self, features):
        try:
            # Ensure features is a dictionary and not a string
            if isinstance(features, str):
                raise TypeError("Features must be a dictionary, not a string")
                
            # Create DataFrame from features
            input_df = pd.DataFrame([features])
            print(f"Input DataFrame columns: {input_df.columns.tolist()}")
            
            # Ensure preprocessor is properly loaded
            if not hasattr(self.preprocessor, 'transform'):
                raise TypeError("Preprocessor is not properly loaded or initialized")
                
            # Check if preprocessor has feature_names_in_ attribute
            if hasattr(self.preprocessor, 'feature_names_in_'):
                print(f"Preprocessor expected columns: {self.preprocessor.feature_names_in_.tolist()}")
                
                # Check for missing columns
                missing_cols = [col for col in self.preprocessor.feature_names_in_ if col not in input_df.columns]
                if missing_cols:
                    print(f"Warning: Missing columns in input data: {missing_cols}")
                    
                # Ensure columns are in the right order
                if hasattr(self.preprocessor, 'feature_names_in_'):
                    input_df = input_df.reindex(columns=self.preprocessor.feature_names_in_, fill_value=0)
            
            # Transform the input data
            input_processed = self.preprocessor.transform(input_df)
            
            # Make prediction
            log_price_pred = self.model.predict(input_processed)[0]
            price_pred = np.expm1(log_price_pred)
            
            return price_pred
        except Exception as e:
            # Add more detailed error information
            error_msg = f"Prediction error: {str(e)}\nFeatures type: {type(features)}\nPreprocessor type: {type(self.preprocessor)}\nInput DataFrame columns: {input_df.columns.tolist() if 'input_df' in locals() else 'N/A'}"
            print(error_msg)
            raise Exception(error_msg)