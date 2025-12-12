import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# --------------------------
# Load trained model from Hugging Face
# --------------------------
model_repo_id = "Disha252001/tourism-best-model"
model_file = "best_model.pkl"

local_model_path = hf_hub_download(repo_id=model_repo_id, filename=model_file)
model = joblib.load(local_model_path)

# --------------------------
# Input form
# --------------------------
with st.form("input_form"):

    Age = st.number_input("Age", min_value=0, max_value=120, value=35)
    TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    CityTier = st.selectbox("City Tier", [1, 2, 3], index=1)
    Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Business", "Other"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=0, value=2)
    PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=7, value=5)
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    NumberOfTrips = st.number_input("Number Of Trips (annual)", min_value=0, value=2)
    Passport = st.selectbox("Passport (0=No,1=Yes)", [0,1], index=1)
    OwnCar = st.selectbox("Own Car (0=No,1=Yes)", [0,1], index=1)
    NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting (below 5)", min_value=0, value=0)
    Designation = st.text_input("Designation", value="Manager")
    MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=50000)
    PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (1-10)", min_value=0, max_value=10, value=8)
    ProductPitched = st.selectbox("Product Pitched", ["Wellness Package", "Family Package", "Other"])
    NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, value=1)
    DurationOfPitch = st.number_input("Duration Of Pitch (minutes)", min_value=0, value=10)

    submitted = st.form_submit_button("Predict")

# --------------------------
# Convert inputs to DataFrame
# --------------------------
def build_input_df():
    row = {
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "ProductPitched": ProductPitched,
        "NumberOfFollowups": NumberOfFollowups,
        "DurationOfPitch": DurationOfPitch
    }
    return pd.DataFrame([row])

# --------------------------
# Predict and display result
# --------------------------
if submitted:
    input_df = build_input_df()
    prediction = model.predict(input_df)
    st.success(f"Predicted ProdTaken: {int(prediction[0])}")
