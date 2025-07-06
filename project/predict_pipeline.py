import joblib
import pandas as pd


model = joblib.load("model/linear_model.pkl")
scaler = joblib.load("scaler/scaler.pkl")

def predict_price(input_df):
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    return prediction[0]
