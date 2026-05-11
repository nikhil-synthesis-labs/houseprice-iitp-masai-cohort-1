"""House Price Predictor — Streamlit demo app.

This app loads a pre-trained linear regression model and predicts house prices
from four user inputs: square footage, bedrooms, age of property, location score.

Used as the anchor project for AIM Session 13.1 (Git, Version Control & Code Quality).
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = "model/model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_price(model, sqft, bedrooms, age, location_score):
    features = np.array([[sqft, bedrooms, age, location_score]])
    prediction = model.predict(features)
    return prediction[0]


# ----------- UI -----------

st.set_page_config(page_title="House Price Predictor", page_icon=":house:")

# BUG #1 — "Hosue" typo. Filed as Issue #1 in ISSUES.md.
st.title("House Price Predictor")
st.write("Estimate a house price from four basic inputs. Built on a linear regression model.")

with st.sidebar:
    st.header("Inputs")
    sqft = st.slider("Square footage", min_value=0, max_value=5000, value=1500, step=50)
    bedrooms = st.slider("Bedrooms", min_value=0, max_value=8, value=3, step=1)
    age = st.slider("Age of property (years)", min_value=0, max_value=100, value=10, step=1)
    location_score = st.slider("Location score (0=rural, 10=prime)", min_value=0.0, max_value=10.0, value=6.0, step=0.5)

model = load_model()
predicted_price = predict_price(model, sqft, bedrooms, age, location_score)

# BUG #2 — Raw float displayed. Should be Indian-formatted currency. Filed as Issue #2.
st.metric("Predicted Price", f"{predicted_price}")

# BUG #3 — ZeroDivisionError when sqft = 0. Filed as Issue #3.
price_per_sqft = predicted_price / sqft
st.write(f"Price per sqft: {price_per_sqft:.2f}")

st.divider()
st.subheader("Sample comparable houses")

sample_data = pd.read_csv("data/sample_houses.csv")
st.dataframe(sample_data, use_container_width=True)

st.caption("Source: Synthetic demo dataset. Not real housing data.")
