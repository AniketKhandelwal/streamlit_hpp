import streamlit as st
import pandas as pd
from predict_pipeline import predict_price
from preprocessing import input_features

st.title("🏠 House Price Prediction App")
st.markdown("Enter the house details below:")


user_data = {}
for feature in input_features:
    user_data[feature] = st.number_input(label=feature, value=0.0)


if st.button("Predict Price"):
    input_df = pd.DataFrame([user_data])
    price = predict_price(input_df)
    st.success(f"🏷️ Estimated House Price: ₹ {price:,.0f}")
