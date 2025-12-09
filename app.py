import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =========================================
# 1. Load trained objects (model, scaler, PCA)
# =========================================

ARTIFACT_PATHS = {
    "model": Path("best_extratrees_pca.pkl"),
    "scaler": Path("scaler.pkl"),
    "pca": Path("pca.pkl"),
}


@st.cache_resource
def load_objects():
    """Load model artifacts; fail fast with a friendly error if missing."""
    missing = [p.name for p in ARTIFACT_PATHS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifact file(s): {', '.join(missing)}. "
            "Place them next to app.py (best_extratrees_pca.pkl, scaler.pkl, pca.pkl)."
        )
    model = joblib.load(ARTIFACT_PATHS["model"])
    scaler = joblib.load(ARTIFACT_PATHS["scaler"])
    pca = joblib.load(ARTIFACT_PATHS["pca"])
    return model, scaler, pca


try:
    model, scaler, pca = load_objects()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()
except Exception as err:  # unexpected load errors
    st.error("Failed to load model artifacts.")
    st.exception(err)
    st.stop()

# These are the feature columns used in training (X in your notebook)
FEATURE_COLUMNS = [
    "male",
    "age",
    "education",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
    "pulse_pressure",
    "MAP",
    "age_x_sysBP",
]

# =========================================
# 2. Preprocessing and prediction functions
# =========================================

def preprocess_input(user_input: dict) -> np.ndarray:
    """
    Take raw user input (original Framingham-style features),
    create engineered features, then apply scaler + PCA
    to get the final PCA features for the model.
    """
    df = pd.DataFrame([user_input])

    # Feature engineering (same logic as in the notebook)
    df["pulse_pressure"] = df["sysBP"] - df["diaBP"]
    df["MAP"] = (2 * df["diaBP"] + df["sysBP"]) / 3
    df["age_x_sysBP"] = df["age"] * df["sysBP"]

    # Order columns exactly as during training
    X = df[FEATURE_COLUMNS]

    # Apply scaler and PCA
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return X_pca


def predict_chd_risk(user_input: dict):
    """
    Returns:
        pred_label: 0 or 1
        prob_pos: probability of TenYearCHD = 1
    """
    X_pca = preprocess_input(user_input)
    prob_pos = model.predict_proba(X_pca)[0, 1]  # class 1 probability
    pred_label = int(prob_pos >= 0.5)
    return pred_label, prob_pos

# =========================================
# 3. Streamlit UI
# =========================================

st.title("10-Year Coronary Heart Disease Risk")

st.markdown("---")
st.header("Patient information")

# Sex -> male (0/1)
sex = st.selectbox("Sex", ["Female", "Male"])
male = 1 if sex == "Male" else 0

# Basic demographics
age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
education = st.selectbox("Education level (1-4)", [1, 2, 3, 4])

# Smoking
currentSmoker_str = st.selectbox("Current smoker?", ["No", "Yes"])
currentSmoker = 1 if currentSmoker_str == "Yes" else 0
cigsPerDay = st.number_input("Cigarettes per day", min_value=0, max_value=100, value=0)

# Medications / history
BPMeds_str = st.selectbox("On blood pressure meds (BPMeds)?", ["No", "Yes"])
BPMeds = 1 if BPMeds_str == "Yes" else 0

prevalentStroke_str = st.selectbox("History of stroke (prevalentStroke)?", ["No", "Yes"])
prevalentStroke = 1 if prevalentStroke_str == "Yes" else 0

prevalentHyp_str = st.selectbox("Prevalent hypertension (prevalentHyp)?", ["No", "Yes"])
prevalentHyp = 1 if prevalentHyp_str == "Yes" else 0

diabetes_str = st.selectbox("Diabetes?", ["No", "Yes"])
diabetes = 1 if diabetes_str == "Yes" else 0

# Lab values / vitals
totChol = st.number_input("Total cholesterol (mg/dL)", min_value=100, max_value=500, value=200)
sysBP = st.number_input("Systolic blood pressure (mmHg)", min_value=80, max_value=260, value=120)
diaBP = st.number_input("Diastolic blood pressure (mmHg)", min_value=40, max_value=140, value=80)
BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0)
heartRate = st.number_input("Heart rate (bpm)", min_value=30, max_value=200, value=70)
glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=400, value=90)

st.markdown("---")

if st.button("Predict 10-year CHD risk"):
    # Pack inputs into a dict matching the notebook's df columns
    user_input = {
        "male": male,
        "age": age,
        "education": int(education),
        "currentSmoker": currentSmoker,
        "cigsPerDay": cigsPerDay,
        "BPMeds": BPMeds,
        "prevalentStroke": prevalentStroke,
        "prevalentHyp": prevalentHyp,
        "diabetes": diabetes,
        "totChol": totChol,
        "sysBP": sysBP,
        "diaBP": diaBP,
        "BMI": BMI,
        "heartRate": heartRate,
        "glucose": glucose,
    }

    pred_label, prob_pos = predict_chd_risk(user_input)
    risk_pct = prob_pos * 100

    st.subheader("Result")
    st.write(f"**Estimated 10-year CHD risk:** {risk_pct:.1f}%")

    if risk_pct < 30:
        st.success("Risk band: **Low (< 30%)**")
    elif risk_pct < 50:
        st.warning("Risk band: **Intermediate (30-49%)**")
    else:
        st.error("Risk band: **High (50%+)**")

    # Show the underlying model class decision at a 0.5 threshold
    st.caption(f"Model class prediction (0/1 @0.5 threshold): {pred_label}")
