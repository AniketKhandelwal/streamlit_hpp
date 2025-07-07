import streamlit as st
import pandas as pd
import plotly.express as px
from predict_pipeline import predict_price
from feature_description import feature_info

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Prediction App")

st.markdown("Enter the property features below to predict the price:")

user_input = {}

for feature, meta in feature_info.items():
    label = f"{feature}"
    help_text = meta.get("description", "")
    categories = meta.get("categories", None)

    if categories:
        options = list(categories.keys())
        option_labels = [f"{key} - {val}" for key, val in categories.items()]
        selection = st.selectbox(label, option_labels, help=help_text)
        selected_key = selection.split(" - ")[0]  
        try:
            user_input[feature] = int(selected_key)
        except ValueError:
            user_input[feature] = selected_key
    else:
        user_input[feature] = st.number_input(label, value=0.0, help=help_text)


input_df = pd.DataFrame([user_input])

if st.button("Predict Price"):
    price = predict_price(input_df)
    st.success(f"Estimated Sale Price: $ {int(price):,}")


st.markdown("---")
st.subheader("📊 Lot Area vs Sale Price")


 
df = pd.read_csv("project/dataset/train.csv") 

if 'LotArea' in df.columns and 'SalePrice' in df.columns:
    fig = px.scatter(
        df,
        x='LotArea',
        y='SalePrice',
        title="Lot Area vs Sale Price",
        labels={'LotArea': 'Lot Area (sq ft)', 'SalePrice': 'Sale Price'},
        color_discrete_sequence=["#00CC96"]
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Required columns 'LotArea' and 'SalePrice' not found in the dataset.")

