import os
import json
import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# 1. Load trained model and classification threshold
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "tourism_xgb_model.joblib")
THRESHOLD_PATH = os.path.join(BASE_DIR, "decision_threshold.json")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_threshold():
    with open(THRESHOLD_PATH, "r") as file:
        threshold_data = json.load(file)

    # Support a numeric threshold or a dictionary
    if isinstance(threshold_data, (int, float)):
        return float(threshold_data)

    for key in ["threshold", "decision_threshold", "selected_threshold"]:
        if key in threshold_data:
            return float(threshold_data[key])

    raise ValueError("Classification threshold not found in JSON file.")


# --------------------------------------------------
# 2. Streamlit application
# --------------------------------------------------

st.set_page_config(
    page_title="Wellness Tourism Purchase Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("Visit with Us")
st.subheader("Wellness Tourism Package Purchase Prediction")

st.write(
    """
    This application predicts whether a customer is likely to purchase
    the Wellness Tourism Package based on their demographic details,
    preferences and sales interaction information.
    """
)


# Load model and threshold
try:
    model = load_model()
    threshold = load_threshold()

except Exception as e:
    st.error(f"Unable to load model or threshold: {e}")
    st.stop()


# --------------------------------------------------
# 3. Collect customer information
# --------------------------------------------------

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=37,
        step=1
    )

    CityTier = st.selectbox(
        "City Tier",
        [1, 2, 3],
        index=0
    )

    TypeofContact = st.selectbox(
        "Type of Contact",
        ["Self Enquiry", "Company Invited"]
    )

    Occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"]
    )

    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5],
        index=1
    )

    NumberOfTrips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=100,
        value=3,
        step=1
    )


with col2:
    Passport = st.selectbox(
        "Passport",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    PitchSatisfactionScore = st.selectbox(
        "Pitch Satisfaction Score",
        [1, 2, 3, 4, 5],
        index=2
    )

    OwnCar = st.selectbox(
        "Own Car",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=0,
        max_value=1000000,
        value=25000,
        step=1000
    )

    DurationOfPitch = st.number_input(
        "Duration of Pitch",
        min_value=0,
        max_value=10000,
        value=15,
        step=1
    )

    NumberOfFollowups = st.number_input(
        "Number of Follow-ups",
        min_value=0,
        max_value=100,
        value=3,
        step=1
    )

    ProductPitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )


# --------------------------------------------------
# 4. Create input DataFrame
# --------------------------------------------------

input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "ProductPitched": ProductPitched
}])


# --------------------------------------------------
# 5. Generate purchase prediction
# --------------------------------------------------

if st.button("Predict Purchase", type="primary"):

    try:
        # Predict probability of purchasing the package
        purchase_probability = model.predict_proba(input_data)[0][1]

        # Apply the threshold selected during model training
        prediction = int(purchase_probability >= threshold)

        st.subheader("Prediction Result")

        st.metric(
            "Purchase Probability",
            f"{purchase_probability * 100:.2f}%"
        )

        if prediction == 1:
            st.success(
                "**Likely to Purchase**\n\n"
                "The customer is predicted to purchase the "
                "Wellness Tourism Package."
            )
        else:
            st.info(
                "**Unlikely to Purchase**\n\n"
                "The customer is predicted not to purchase the "
                "Wellness Tourism Package."
            )

        st.caption(
            f"Classification threshold used: {threshold:.2f}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
