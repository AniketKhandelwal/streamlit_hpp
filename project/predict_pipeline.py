import pandas as pd
import joblib
from preprocessing import categorical_encoding

# Load the model and scaler
model = joblib.load("model/linear_model.pkl")
scaler = joblib.load("scaler/scaler.pkl")
features_list = joblib.load("model/features_list.pkl")  # List of features used during training

def predict_price(input_df):
    # 1. Apply encoding (ordinal + one-hot)
    encoded_input = categorical_encoding(input_df)

    # 2. Add any missing columns
    for col in features_list:
        if col not in encoded_input.columns:
            encoded_input[col] = 0

    # 3. Ensure correct column order
    encoded_input = encoded_input[features_list]

    # 4. Scale and predict
    scaled_input = scaler.transform(encoded_input)
    prediction = model.predict(scaled_input)

    return prediction[0]
