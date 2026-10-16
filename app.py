import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="AC Price Predictor",
    page_icon="❄️",
    layout="centered"
)

st.title("❄️ AC Price Predictor")
st.write("Predict the estimated AC price using the trained Machine Learning model.")

MODEL_PATH = Path(__file__).with_name("Ac_price_model.pkl")

@st.cache_resource
def load_model():
    # The supplied model appears to have been saved with scikit-learn 1.6.1.
    # Use the matching version listed in requirements.txt to avoid compatibility errors.
    return joblib.load(MODEL_PATH)

st.subheader("Enter AC Details")

brand = st.selectbox(
    "Select AC Brand",
    ["LG", "Samsung", "Voltas", "Daikin", "Blue Star", "Hitachi", "Other"]
)

quantity = st.number_input(
    "Quantity (number of ACs)",
    min_value=1,
    max_value=1000,
    value=1,
    step=1
)

if st.button("Predict AC Price", type="primary"):
    if not MODEL_PATH.exists():
        st.error("Model file not found. Keep Ac_price_model.pkl in the same folder as app.py.")
    else:
        try:
            model = load_model()
            # The uploaded pipeline was inspected and its input columns are Brand and Quantity.
            input_data = pd.DataFrame({
                "Brand": [brand],
                "Quantity": [quantity]
            })

            prediction = float(model.predict(input_data)[0])
            st.success(f"Predicted value: ₹{prediction:,.2f}")
            st.caption(
                "This estimate is based on the model's training data. If the model was trained "
                "to predict per-unit price rather than total price, multiply by quantity only "
                "after confirming that target definition."
            )
            with st.expander("View input sent to the model"):
                st.dataframe(input_data, use_container_width=True)
        except Exception as error:
            st.error("The model could not be loaded or prediction failed.")
            st.code(str(error))
            st.info(
                "This model was saved with scikit-learn 1.6.1. Install the versions in "
                "requirements.txt, then run the app again. Also ensure the model file is trusted."
            )

st.divider()
st.subheader("Model Information")
st.write("Model file: `Ac_price_model.pkl`")
st.write("Input features: `Brand`, `Quantity`")
st.caption(
    "This app uses your saved model; it does not retrain it. Predictions are estimates, "
    "not guaranteed market prices."
)
