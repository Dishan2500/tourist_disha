import os
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
from pathlib import Path

st.set_page_config(page_title="Wellness Tourism Purchase Predictor", layout="centered")

# --------------------------
# Configuration
# --------------------------
# Model repo that you uploaded earlier (change if different)
MODEL_REPO_ID = "Disha252001/wellness-tourism-model"
MODEL_FILENAME = "best_model.joblib"

# Hugging Face token (should be set in env for private models)
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if HF_TOKEN is None:
    st.info("ℹ️ HF_TOKEN not found in environment — public models will still download, private ones will fail.")

# Local cache path for downloaded model
cache_dir = Path.home() / ".cache" / "hf_models"
cache_dir.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_model():
    """Load model from Hugging Face Hub or local fallback."""
    try:
        st.info(f"Loading model from {MODEL_REPO_ID}/{MODEL_FILENAME}...")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model",
            token=HF_TOKEN,
            cache_dir=str(cache_dir)
        )
        st.success("Model downloaded successfully.")
        # Compatibility shim for sklearn internal objects referenced in pickle
        try:
            import importlib, sys
            mod_name = "sklearn.compose._column_transformer"
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                import types
                mod = types.ModuleType(mod_name)
                sys.modules[mod_name] = mod
            if not hasattr(mod, "_RemainderColsList"):
                class _RemainderColsList:
                    def __init__(self, *args, **kwargs):
                        pass
                setattr(mod, "_RemainderColsList", _RemainderColsList)
        except Exception as e:
            st.warning(f"Compatibility shim setup failed: {e}")

        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.warning(f"joblib.load failed: {e}. Attempting safe unpickle fallback.")
            # Fallback: use a SafeUnpickler that dynamically creates placeholder
            # classes for missing attributes referenced by the pickle.
            try:
                import io, pickle

                class SafeUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        try:
                            return super().find_class(module, name)
                        except Exception:
                            # Create a minimal placeholder class
                            return type(name, (), {})

                with open(model_path, "rb") as f:
                    data = f.read()
                obj = SafeUnpickler(io.BytesIO(data)).load()
                st.info("Loaded model using safe unpickler fallback. Behavior may differ from original model.")
                return obj
            except Exception as e2:
                st.error(f"Safe unpickle also failed: {e2}")
                raise
    except Exception as e:
        st.error(f"Failed to load model from Hugging Face Hub: {e}")
        st.warning(
            "**Fallback**: Using a dummy model. "
            "Make sure the model is uploaded to Hugging Face Hub: "
            f"`{MODEL_REPO_ID}` with filename `{MODEL_FILENAME}`"
        )
        # Return a simple fallback model (e.g., always predict 0)
        from sklearn.linear_model import LogisticRegression
        fallback = LogisticRegression()
        st.info("Using fallback model (always predicts 0). This is a demo placeholder.")
        return fallback

model = load_model()

st.title("Wellness Tourism — Purchase Prediction")
st.write("Fill the customer details below and click **Predict**.")

# --------------------------
# Input form: fields based on your data dictionary
# Fill or remove fields to match your training data exactly
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
# Convert inputs to DataFrame and predict
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
    df = pd.DataFrame([row])
    return df

if submitted:
    input_df = build_input_df()
    st.subheader("Input (as DataFrame)")
    st.dataframe(input_df)

    # Model should be a pipeline with preprocessor so we can directly pass df
    try:
        pred = model.predict(input_df)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)
            # If binary, proba[:,1] is probability of class 1
            if proba.shape[1] == 2:
                proba = float(proba[:,1][0])
            else:
                # multi-class: show list
                proba = [float(x) for x in proba[0]]
        st.success(f"Prediction (ProdTaken): {int(pred[0])}")
        if proba is not None:
            st.info(f"Probability: {proba}")
    except Exception as e:
        st.error(f"Failed to predict: {e}")
