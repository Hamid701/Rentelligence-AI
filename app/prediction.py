import pickle
import pandas as pd
import numpy as np

class RentalPricePredictor:
    def __init__(self, model_path, preprocessor_path):
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
    
    def predict(self, features):
     
        input_df = pd.DataFrame([features])
        input_processed = self.preprocessor.transform(input_df)
        log_price_pred = self.model.predict(input_processed)[0]
        price_pred = np.expm1(log_price_pred)
        
        return price_pred