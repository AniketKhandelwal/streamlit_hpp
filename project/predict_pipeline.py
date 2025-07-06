import pandas as pd
import joblib
from preprocessing import categorical_encoding
import os

base_dir = os.path.dirname(__file__)

model_path = os.path.join(base_dir,"model","linear_model.pkl")
scaler_path = os.path.join(base_dir, "scaler", "scaler.pkl")
features_list_path = os.path.join(base_dir, "model", "features_list.pkl")


model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
features_list = joblib.load(features_list_path)

 

def predict_price(input_df):
   
    encoded_input = categorical_encoding(input_df)

  
    for col in features_list:
        if col not in encoded_input.columns:
            encoded_input[col] = 0

    
    encoded_input = encoded_input[features_list]

    scaled_input = scaler.transform(encoded_input)
    prediction = model.predict(scaled_input)

    return prediction[0]
