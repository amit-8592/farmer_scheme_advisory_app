"""
Smart Agriculture Scheme Advisor
--------------------------------
A Streamlit web app that recommends the most suitable government
agriculture scheme for a farmer using a pre-trained ML model.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Agriculture Scheme Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Custom CSS – green agriculture theme
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 50%, #dcedc8 100%);
    }

    /* Main header */
    .main-header {
        background: linear-gradient(90deg, #2e7d32 0%, #43a047 50%, #66bb6a 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        color: #ffffff;
        box-shadow: 0 8px 24px rgba(46, 125, 50, 0.25);
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .main-header p {
        color: #e8f5e9;
        margin: 0.5rem 0 0 0;
        font-size: 1.05rem;
    }

    /* Section card */
    .section-card {
        background: #ffffff;
        padding: 1.5rem 1.75rem;
        border-radius: 14px;
        border-left: 5px solid #43a047;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.25rem;
    }
    .section-title {
        color: #1b5e20;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }

    /* Recommended scheme highlight */
    .recommend-box {
        background: linear-gradient(135deg, #43a047 0%, #2e7d32 100%);
        color: #ffffff;
        padding: 1.75rem 1.5rem;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(46, 125, 50, 0.35);
        margin: 1rem 0;
    }
    .recommend-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.9;
    }
    .recommend-name {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .recommend-confidence {
        font-size: 1.1rem;
        background: rgba(255, 255, 255, 0.2);
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        margin-top: 0.5rem;
    }

    /* Alternative scheme rows */
    .alt-scheme {
        background: #ffffff;
        padding: 0.9rem 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #81c784;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }
    .alt-scheme-name {
        font-weight: 600;
        color: #1b5e20;
        font-size: 1.05rem;
    }
    .alt-scheme-conf {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-weight: 700;
        font-size: 0.95rem;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(90deg, #2e7d32 0%, #43a047 100%);
        color: #ffffff;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 30px;
        border: none;
        width: 100%;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(46, 125, 50, 0.45);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b5e20 0%, #2e7d32 100%);
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #558b2f;
        padding: 1.5rem 0 0.5rem 0;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🌱 Smart Agriculture Scheme Advisor</h1>
        <p>AI-powered recommendation of the most suitable Government scheme for farmers</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Load artifacts (cached)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the trained model, feature columns and label encoder."""
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    return model, list(columns), label_encoder


try:
    model, FEATURE_COLUMNS, label_encoder = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load model files: {e}")
    st.stop()

# ----------------------------------------------------------------------
# Sidebar – About
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌾 About")
    st.write(
        "This advisor uses a trained **CatBoost** classifier to match farmer "
        "profiles with the most suitable Government agriculture scheme."
    )
    st.markdown("---")
    st.markdown("### 📋 Available Schemes")
    for scheme in label_encoder.classes_:
        st.markdown(f"• **{scheme}**")
    st.markdown("---")
    st.caption("Built with Streamlit & scikit-learn 🌱")

# ----------------------------------------------------------------------
# Input Form
# ----------------------------------------------------------------------
st.markdown('<div class="section-title">📝 Farmer Input Form</div>', unsafe_allow_html=True)

with st.form("farmer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        land_size = st.selectbox("🌾 Land Size", ["Small", "Medium", "Large"], index=0)
        income_level = st.selectbox("💰 Income Level", ["Low", "Medium", "High"], index=0)
        loan_status = st.selectbox("🏦 Loan Status", ["No", "Yes"], index=0)

    with col2:
        crop_type = st.selectbox(
            "🌽 Crop Type",
            ["Rice", "Wheat", "Cotton", "Maize", "Soybean"],
            index=0,
        )
        irrigation = st.selectbox("💧 Irrigation", ["Yes", "No"], index=0)
        soil_type = st.selectbox(
            "🪴 Soil Type", ["Sandy", "Clay", "Loamy", "Black"], index=2
        )

    with col3:
        weather_risk = st.selectbox("🌦️ Weather Risk", ["Low", "Medium", "High"], index=0)
        experience = st.selectbox(
            "👨‍🌾 Experience", ["Beginner", "Intermediate", "Expert"], index=1
        )
        state = st.selectbox(
            "📍 State", ["Punjab", "UP", "MP", "MH", "Bihar"], index=0
        )

    submitted = st.form_submit_button("🔍 Predict Recommended Scheme")

# ----------------------------------------------------------------------
# Helpers – encoding + feature engineering
# ----------------------------------------------------------------------
LAND_MAP = {"Small": 1, "Medium": 2, "Large": 3}
INCOME_MAP = {"Low": 1, "Medium": 2, "High": 3}
EXP_MAP = {"Beginner": 1, "Intermediate": 2, "Expert": 3}
RISK_MAP = {"Low": 1, "Medium": 2, "High": 3}
LOAN_MAP = {"No": 0, "Yes": 1}


def build_feature_row(
    land_size: str,
    income_level: str,
    loan_status: str,
    crop_type: str,
    irrigation: str,
    soil_type: str,
    weather_risk: str,
    experience: str,
    state: str,
) -> pd.DataFrame:
    """Encode raw UI inputs and align them to FEATURE_COLUMNS."""
    row = {c: 0 for c in FEATURE_COLUMNS}

    # Numeric / ordinal
    ls = LAND_MAP[land_size]
    il = INCOME_MAP[income_level]
    ls_loan = LOAN_MAP[loan_status]
    exp = EXP_MAP[experience]
    risk = RISK_MAP[weather_risk]

    row["Land_Size"] = ls
    row["Income_Level"] = il
    row["Loan_Status"] = ls_loan
    row["Farm_Experience"] = exp

    # Derived features (matching training feature-engineering conventions)
    row["Land_Income"] = ls * il
    row["Smart_Factor"] = ls + il + exp
    row["Risk_Score"] = risk
    row["Financial_Score"] = il - ls_loan
    row["Eligible_PM_Kisan"] = 1 if ls == 1 else 0
    row["Eligible_Loan_Scheme"] = 1 if ls_loan == 1 else 0

    # One-hot: State
    state_col = f"State_{state}"
    if state_col in row:
        row[state_col] = 1

    # One-hot: Crop
    crop_col = f"Crop_{crop_type}"
    if crop_col in row:
        row[crop_col] = 1

    # One-hot: Irrigation
    row[f"Irrigation_{irrigation}"] = 1

    # One-hot: Soil Type
    soil_col = f"Soil_Type_{soil_type}"
    if soil_col in row:
        row[soil_col] = 1

    # One-hot: Weather Risk
    row[f"Weather_Risk_{weather_risk}"] = 1

    # Interaction: Crop x Soil
    cs_col = f"Crop_Soil_{crop_type}_{soil_type}"
    if cs_col in row:
        row[cs_col] = 1

    # Align columns exactly
    df = pd.DataFrame([row])[FEATURE_COLUMNS]
    return df


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if submitted:
    with st.spinner("🌿 Analyzing farmer profile and finding the best scheme..."):
        try:
            X = build_feature_row(
                land_size,
                income_level,
                loan_status,
                crop_type,
                irrigation,
                soil_type,
                weather_risk,
                experience,
                state,
            )

            proba = model.predict_proba(X)[0]
            classes = label_encoder.classes_

            # Top-1
            top_idx = int(np.argmax(proba))
            top_scheme = classes[top_idx]
            top_conf = float(proba[top_idx])

            # Top-3
            top3_idx = np.argsort(proba)[::-1][:3]

            # Recommended card
            st.markdown(
                f"""
                <div class="recommend-box">
                    <div class="recommend-label">✅ Recommended Scheme</div>
                    <div class="recommend-name">{top_scheme}</div>
                    <div class="recommend-confidence">
                        Confidence: {top_conf * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Top 3 alternatives
            st.markdown(
                '<div class="section-title">🏆 Top 3 Scheme Suggestions</div>',
                unsafe_allow_html=True,
            )

            for rank, i in enumerate(top3_idx, start=1):
                st.markdown(
                    f"""
                    <div class="alt-scheme">
                        <div class="alt-scheme-name">#{rank} &nbsp; {classes[i]}</div>
                        <div class="alt-scheme-conf">{proba[i] * 100:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Full probability chart
            st.markdown(
                '<div class="section-title">📊 Probability Distribution</div>',
                unsafe_allow_html=True,
            )
            chart_df = (
                pd.DataFrame({"Scheme": classes, "Probability": proba})
                .sort_values("Probability", ascending=False)
                .set_index("Scheme")
            )
            st.bar_chart(chart_df, color="#43a047", height=350)

            # Submitted profile (expandable)
            with st.expander("👀 View submitted farmer profile"):
                profile = {
                    "Land Size": land_size,
                    "Income Level": income_level,
                    "Loan Status": loan_status,
                    "Crop Type": crop_type,
                    "Irrigation": irrigation,
                    "Soil Type": soil_type,
                    "Weather Risk": weather_risk,
                    "Experience": experience,
                    "State": state,
                }
                st.table(pd.DataFrame(profile.items(), columns=["Attribute", "Value"]))

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
else:
    st.info("👆 Fill in the farmer details above and click **Predict Recommended Scheme** to get a recommendation.")

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown(
    '<div class="footer">🌱 Smart Agriculture Scheme Advisor · Empowering farmers with AI</div>',
    unsafe_allow_html=True,
)
