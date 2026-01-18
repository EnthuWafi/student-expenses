import streamlit as st
import pandas as pd
import joblib

# --- LOAD MODEL ---
try:
    model_data = joblib.load("student_spending_model.pkl")
    model = model_data["model"]
    feature_names = model_data["features"]
except FileNotFoundError:
    st.error("Model file not found. Please train the model first!")
    st.stop()

st.set_page_config(page_title="Student Expense Predictor", page_icon="💸")

st.title("💸 Student Expense Predictor")
st.write("Predicts expenses based on these selected features!")

# --- SIDEBAR / INPUTS ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        # 1. Allowance
        allowance = st.number_input("Monthly Allowance from Parents/Guardians (RM)", min_value=0, value=300, step=50)

        # Living Arrangement
        living_options = ["With Family", "On-Campus Hostel", "Off-Campus Rental"]
        living = st.selectbox("Living Arrangement", living_options)

        # Shopping Frequency
        shop_map = {"Rarely": 0, "Occasionally": 1, "Frequently": 2}
        shop = st.selectbox("Shopping Frequency for Non-Essential", list(shop_map.keys()))
        shop_val = shop_map[shop]

        # Eating Out
        eat_map = {"Rarely": 0, "Sometimes": 1, "Often (daily)": 2}
        eat = st.selectbox("Eating Out Frequency", list(eat_map.keys()))
        eat_val = eat_map[eat]

    with col2:


        # Laptop
        laptop = st.radio("Do you own a laptop?", ["No", "Yes"])
        laptop_val = 1 if laptop == "Yes" else 0

        # Tracking
        track = st.radio("Do you track expenses?", ["No", "Yes"])
        track_val = 1 if track == "Yes" else 0

        # Consciousness
        conscious = st.slider("Financial Discipline (1-5)", 1, 5, 3)

    submit = st.form_submit_button("Predict Expenses")

# --- PREDICTION ---
if submit:
    is_hostel = 1 if living == "On-Campus Hostel" else 0
    is_rental = 1 if living == "Off-Campus Rental" else 0

    # Create Data Row
    input_data = {
        'ShopNonEssential': shop_val,
        'Allowance': allowance,
        'Hostel': is_hostel,
        'Rental': is_rental,
        'EatOut': eat_val,
        'Laptop': laptop_val,
        'TrackExpenses': track_val,
        'Conscious': conscious
    }

    # Convert to DF
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names] # Ensure order

    # Predict
    pred = model.predict(input_df)[0]

    # Result
    st.divider()
    st.success(f"💰 Predicted Monthly Expenses: **RM {pred:,.2f}**")

