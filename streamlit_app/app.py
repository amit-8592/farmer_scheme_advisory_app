"""
Smart Agriculture Scheme Advisor
--------------------------------
Streamlit web app that recommends the most suitable Government agriculture
scheme for a farmer using a pre-trained ML model.

Features:
    1. ML-based scheme recommendation (top-1 and top-3 with confidence)
    2. "View Scheme Details" — eligibility, benefits, documents, official link
    3. "Apply Now" — auto-filled application form using st.session_state

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Smart Agriculture Scheme Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# CUSTOM CSS — green agriculture theme
# ======================================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 50%, #dcedc8 100%);
    }
    .main-header {
        background: linear-gradient(90deg, #2e7d32 0%, #43a047 50%, #66bb6a 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        color: #ffffff;
        box-shadow: 0 8px 24px rgba(46, 125, 50, 0.25);
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: #fff; font-size: 2.2rem; font-weight: 800; margin: 0; }
    .main-header p  { color: #e8f5e9; margin: 0.5rem 0 0 0; font-size: 1.05rem; }

    .section-title {
        color: #1b5e20; font-size: 1.25rem; font-weight: 700; margin-bottom: 0.75rem;
    }
    .recommend-box {
        background: linear-gradient(135deg, #43a047 0%, #2e7d32 100%);
        color: #fff; padding: 1.75rem 1.5rem; border-radius: 14px;
        text-align: center; box-shadow: 0 8px 20px rgba(46, 125, 50, 0.35);
        margin: 1rem 0;
    }
    .recommend-label { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px; opacity: 0.9; }
    .recommend-name  { font-size: 2rem; font-weight: 800; margin: 0.5rem 0; }
    .recommend-confidence {
        font-size: 1.1rem; background: rgba(255,255,255,0.2);
        display: inline-block; padding: 0.3rem 1rem; border-radius: 20px; margin-top: 0.5rem;
    }
    .alt-scheme {
        background: #fff; padding: 0.9rem 1.2rem; border-radius: 10px;
        border-left: 4px solid #81c784; margin: 0.5rem 0;
        display: flex; justify-content: space-between; align-items: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .alt-scheme-name { font-weight: 600; color: #1b5e20; font-size: 1.05rem; }
    .alt-scheme-conf {
        background: #e8f5e9; color: #2e7d32; padding: 0.25rem 0.75rem;
        border-radius: 15px; font-weight: 700; font-size: 0.95rem;
    }

    /* Scheme details card */
    .detail-card {
        background: #ffffff;
        padding: 1.5rem 1.75rem;
        border-radius: 14px;
        border-left: 5px solid #2e7d32;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        margin: 1rem 0;
    }
    .detail-card h3 { color: #1b5e20; margin: 0 0 0.5rem 0; }
    .detail-row { margin: 0.6rem 0; font-size: 1.02rem; line-height: 1.55; }
    .detail-label { color: #2e7d32; font-weight: 700; }

    .stButton > button {
        background: linear-gradient(90deg, #2e7d32 0%, #43a047 100%);
        color: #fff; font-weight: 700; font-size: 1.05rem;
        padding: 0.6rem 1.5rem; border-radius: 30px; border: none;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
        transition: all 0.2s ease; width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 6px 16px rgba(46, 125, 50, 0.45);
    }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1b5e20 0%, #2e7d32 100%); }
    [data-testid="stSidebar"] * { color: #fff !important; }
    .footer { text-align: center; color: #558b2f; padding: 1.5rem 0 0.5rem 0; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================================
# HEADER
# ======================================================================
st.markdown(
    """
    <div class="main-header">
        <h1>🌱 Smart Agriculture Scheme Advisor</h1>
        <p>AI-powered recommendation of the most suitable Government scheme for farmers</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ======================================================================
# LOAD MODEL ARTIFACTS (cached)
# ======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load trained model, feature columns and label encoder."""
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    return model, list(columns), label_encoder


try:
    model, FEATURE_COLUMNS, label_encoder = load_artifacts()
except Exception as e:
    st.error(f"❌ Failed to load model files: {e}")
    st.stop()

# ======================================================================
# SCHEME DETAILS DATABASE
# ======================================================================
scheme_data = {
    "PM-Kisan": {
        "description": (
            "PM-Kisan Samman Nidhi is a central sector scheme launched by the "
            "Government of India that provides income support of ₹6,000 per year "
            "to eligible small and marginal farmer families."
        ),
        "eligibility": (
            "• Small & marginal farmers owning cultivable land up to 2 hectares\n\n"
            "• Must be Indian citizen with valid Aadhaar\n\n"
            "• Family income below prescribed limit"
        ),
        "benefits": (
            "• ₹6,000 per year in three equal installments of ₹2,000\n\n"
            "• Direct Benefit Transfer (DBT) to bank account\n\n"
            "• No intermediaries involved"
        ),
        "documents": (
            "• Aadhaar Card\n\n• Land ownership records\n\n"
            "• Bank account details (IFSC + account number)\n\n• Recent passport photo"
        ),
        "link": "https://pmkisan.gov.in/",
    },
    "PMFBY": {
        "description": (
            "Pradhan Mantri Fasal Bima Yojana (PMFBY) is a crop insurance scheme "
            "that provides financial support to farmers suffering crop loss or "
            "damage due to unforeseen events."
        ),
        "eligibility": (
            "• All farmers growing notified crops in notified areas\n\n"
            "• Both loanee and non-loanee farmers are eligible\n\n"
            "• Sharecroppers & tenant farmers included"
        ),
        "benefits": (
            "• Comprehensive risk cover — pre-sowing to post-harvest\n\n"
            "• Low premium: 2% (Kharif), 1.5% (Rabi), 5% (Commercial/Horticulture)\n\n"
            "• Sum insured matches scale of finance"
        ),
        "documents": (
            "• Aadhaar Card\n\n• Land records (Khasra/Khatauni)\n\n"
            "• Sowing certificate\n\n• Bank account details"
        ),
        "link": "https://pmfby.gov.in/",
    },
    "KCC": {
        "description": (
            "Kisan Credit Card (KCC) scheme provides farmers with timely and "
            "adequate credit for their cultivation needs, including post-harvest "
            "expenses and consumption requirements."
        ),
        "eligibility": (
            "• All farmers — individual/joint borrowers who are owner cultivators\n\n"
            "• Tenant farmers, oral lessees & sharecroppers\n\n"
            "• Self-Help Groups (SHGs) of farmers"
        ),
        "benefits": (
            "• Short-term loans at 7% interest (4% on prompt repayment)\n\n"
            "• Credit limit up to ₹3 lakh without collateral\n\n"
            "• Crop insurance coverage included\n\n• Flexible repayment"
        ),
        "documents": (
            "• Aadhaar & PAN Card\n\n• Land documents\n\n"
            "• Passport-size photos\n\n• Bank account statement"
        ),
        "link": "https://www.myscheme.gov.in/schemes/kcc",
    },
    "Crop Insurance": {
        "description": (
            "Crop Insurance scheme protects farmers from financial losses caused "
            "by natural calamities, pests, and diseases that damage their crops."
        ),
        "eligibility": (
            "• Farmers cultivating notified crops\n\n"
            "• Both loanee and non-loanee farmers\n\n• Owner and tenant farmers"
        ),
        "benefits": (
            "• Financial compensation for crop loss\n\n"
            "• Coverage for natural calamities (drought, flood, hail)\n\n"
            "• Subsidized premium rates"
        ),
        "documents": (
            "• Aadhaar Card\n\n• Land records\n\n"
            "• Crop sowing proof\n\n• Bank account details"
        ),
        "link": "https://pmfby.gov.in/",
    },
    "Soil Health Card": {
        "description": (
            "Soil Health Card scheme issues farmers a soil health card containing "
            "the nutrient status of their soil along with recommendations on the "
            "appropriate dosage of nutrients to improve productivity."
        ),
        "eligibility": (
            "• All farmers across India\n\n"
            "• No income or landholding restrictions\n\n• Free for every land-owning farmer"
        ),
        "benefits": (
            "• Free soil testing every 2 years\n\n"
            "• Customized fertilizer recommendations\n\n"
            "• Improved crop yield and reduced input cost"
        ),
        "documents": (
            "• Aadhaar Card\n\n• Land ownership records\n\n• Contact details"
        ),
        "link": "https://soilhealth.dac.gov.in/",
    },
    "Organic Farming": {
        "description": (
            "Paramparagat Krishi Vikas Yojana (PKVY) promotes organic farming by "
            "providing financial assistance, training and certification support."
        ),
        "eligibility": (
            "• Farmers willing to convert to organic farming\n\n"
            "• Must form clusters of 20 hectares or 50+ farmers\n\n"
            "• Commitment of at least 3 years"
        ),
        "benefits": (
            "• ₹50,000 per hectare over 3 years\n\n"
            "• Free certification support\n\n"
            "• Training and marketing assistance\n\n• Better price realization"
        ),
        "documents": (
            "• Aadhaar Card\n\n• Land records\n\n"
            "• Bank account details\n\n• Cluster membership proof"
        ),
        "link": "https://pgsindia-ncof.gov.in/pkvy/",
    },
    "Irrigation Subsidy": {
        "description": (
            "Pradhan Mantri Krishi Sinchayee Yojana (PMKSY) — Per Drop More Crop "
            "provides subsidies for micro-irrigation systems (drip / sprinkler) to "
            "enhance water use efficiency."
        ),
        "eligibility": (
            "• All categories of farmers with cultivable land\n\n"
            "• Priority to small / marginal farmers, SC/ST, women\n\n"
            "• Land must have assured water source"
        ),
        "benefits": (
            "• Up to 55% subsidy for small & marginal farmers\n\n"
            "• Up to 45% subsidy for other farmers\n\n"
            "• Water savings of 30–60%\n\n• Yield increase of 20–50%"
        ),
        "documents": (
            "• Aadhaar Card\n\n• Land records\n\n"
            "• Water source certificate\n\n• Bank details"
        ),
        "link": "https://pmksy.gov.in/",
    },
    "Farm Mechanization": {
        "description": (
            "Sub-Mission on Agricultural Mechanization (SMAM) promotes the use of "
            "farm machinery by providing subsidies to make equipment affordable "
            "for small and marginal farmers."
        ),
        "eligibility": (
            "• All farmers with valid land documents\n\n"
            "• Priority to SC/ST, women and small/marginal farmers\n\n"
            "• Farmer Producer Organizations (FPOs) eligible"
        ),
        "benefits": (
            "• 40–50% subsidy on agricultural machinery\n\n"
            "• Custom Hiring Centres (CHCs) for affordable machine rental\n\n"
            "• Training on machinery usage"
        ),
        "documents": (
            "• Aadhaar Card\n\n• Land records\n\n"
            "• Caste certificate (if applicable)\n\n• Bank account details"
        ),
        "link": "https://agrimachinery.nic.in/",
    },
    "Farmer Training": {
        "description": (
            "Agricultural Technology Management Agency (ATMA) scheme provides "
            "training, exposure visits and demonstrations to farmers on modern "
            "agricultural practices."
        ),
        "eligibility": (
            "• All farmers engaged in agriculture or allied activities\n\n"
            "• No income or land restrictions\n\n"
            "• Priority to small / marginal & women farmers"
        ),
        "benefits": (
            "• Free training on modern farming techniques\n\n"
            "• Exposure visits to progressive farms\n\n"
            "• Demonstration of new technology\n\n• Skill development certification"
        ),
        "documents": (
            "• Aadhaar Card\n\n• Land records (if applicable)\n\n• Contact details"
        ),
        "link": "https://agricoop.gov.in/",
    },
    "Subsidy Scheme": {
        "description": (
            "General agricultural subsidy scheme offering support on inputs such "
            "as seeds, fertilizers, pesticides and equipment to reduce cost of "
            "cultivation for farmers."
        ),
        "eligibility": (
            "• All farmers with valid land records\n\n"
            "• Resident of India\n\n• Priority to small and marginal farmers"
        ),
        "benefits": (
            "• Subsidized seeds and fertilizers\n\n"
            "• Equipment purchase discount\n\n"
            "• Direct Benefit Transfer to bank account"
        ),
        "documents": (
            "• Aadhaar Card\n\n• Land records\n\n"
            "• Bank account details\n\n• Caste/Income certificate (if applicable)"
        ),
        "link": "https://www.myscheme.gov.in/",
    },
}

# ======================================================================
# SIDEBAR
# ======================================================================
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

# ======================================================================
# SESSION STATE — initialise once
# ======================================================================
def _init_state():
    defaults = {
        "prediction_done": False,
        "top_scheme": None,
        "top_conf": 0.0,
        "top3": [],          # list of (scheme_name, prob) tuples
        "user_inputs": {},   # stored UI form values for auto-fill later
        "show_details": False,
        "show_apply": False,
        "applied": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ======================================================================
# INPUT FORM
# ======================================================================
st.markdown(
    '<div class="section-title">📝 Farmer Input Form</div>', unsafe_allow_html=True
)

with st.form("farmer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        land_size = st.selectbox("🌾 Land Size", ["Small", "Medium", "Large"], index=0)
        income_level = st.selectbox("💰 Income Level", ["Low", "Medium", "High"], index=0)

    with col2:
        crop_type = st.selectbox(
            "🌽 Crop Type",
            ["Rice", "Wheat", "Cotton", "Maize", "Soybean"],
            index=0,
        )
        irrigation = st.selectbox("💧 Irrigation", ["Yes", "No"], index=0)

    with col3:
        loan_status = st.selectbox("🏦 Loan Status", ["No", "Yes"], index=0)
        state = st.selectbox("📍 State", ["Punjab", "UP", "MP", "MH", "Bihar"], index=0)

    submitted = st.form_submit_button("🔍 Predict Recommended Scheme")

# ======================================================================
# ENCODING HELPERS
# ======================================================================
LAND_MAP   = {"Small": 1, "Medium": 2, "Large": 3}
INCOME_MAP = {"Low": 1, "Medium": 2, "High": 3}
EXP_MAP    = {"Beginner": 1, "Intermediate": 2, "Expert": 3}
RISK_MAP   = {"Low": 1, "Medium": 2, "High": 3}
LOAN_MAP   = {"No": 0, "Yes": 1}

# Internal defaults for features removed from the UI.
# These keep the trained model's 49-feature alignment intact.
DEFAULT_SOIL_TYPE    = "Loamy"         # most common / neutral soil type
DEFAULT_WEATHER_RISK = "Medium"        # middle-of-the-road risk
DEFAULT_EXPERIENCE   = "Intermediate"  # middle-of-the-road experience


def build_feature_row(
    land_size, income_level, loan_status, crop_type, irrigation, state,
) -> pd.DataFrame:
    """Encode raw UI inputs and align them to FEATURE_COLUMNS.

    Soil type, weather risk and experience are no longer asked in the UI;
    sensible defaults are injected so the model still receives its full
    49-column input vector.
    """
    row = {c: 0 for c in FEATURE_COLUMNS}

    soil_type    = DEFAULT_SOIL_TYPE
    weather_risk = DEFAULT_WEATHER_RISK
    experience   = DEFAULT_EXPERIENCE

    ls   = LAND_MAP[land_size]
    il   = INCOME_MAP[income_level]
    loan = LOAN_MAP[loan_status]
    exp  = EXP_MAP[experience]
    risk = RISK_MAP[weather_risk]

    row["Land_Size"]            = ls
    row["Income_Level"]         = il
    row["Loan_Status"]          = loan
    row["Farm_Experience"]      = exp
    row["Land_Income"]          = ls * il
    row["Smart_Factor"]         = ls + il + exp
    row["Risk_Score"]           = risk
    row["Financial_Score"]      = il - loan
    row["Eligible_PM_Kisan"]    = 1 if ls == 1 else 0
    row["Eligible_Loan_Scheme"] = 1 if loan == 1 else 0

    for key in (
        f"State_{state}", f"Crop_{crop_type}", f"Irrigation_{irrigation}",
        f"Soil_Type_{soil_type}", f"Weather_Risk_{weather_risk}",
        f"Crop_Soil_{crop_type}_{soil_type}",
    ):
        if key in row:
            row[key] = 1

    return pd.DataFrame([row])[FEATURE_COLUMNS]


# ======================================================================
# PREDICTION (only runs when form submitted)
# ======================================================================
if submitted:
    with st.spinner("🌿 Analyzing farmer profile and finding the best scheme..."):
        try:
            X = build_feature_row(
                land_size, income_level, loan_status, crop_type,
                irrigation, state,
            )
            proba = model.predict_proba(X)[0]
            classes = label_encoder.classes_

            top_idx = int(np.argmax(proba))
            top3_idx = np.argsort(proba)[::-1][:3]

            # Persist prediction + user inputs for later (details/apply buttons)
            st.session_state.prediction_done = True
            st.session_state.top_scheme = str(classes[top_idx])
            st.session_state.top_conf = float(proba[top_idx])
            st.session_state.top3 = [(str(classes[i]), float(proba[i])) for i in top3_idx]
            st.session_state.all_proba = {str(classes[i]): float(proba[i]) for i in range(len(classes))}
            st.session_state.user_inputs = {
                "Land Size": land_size,
                "Income Level": income_level,
                "Loan Status": loan_status,
                "Crop Type": crop_type,
                "Irrigation": irrigation,
                "State": state,
            }
            # Reset toggles on new prediction
            st.session_state.show_details = False
            st.session_state.show_apply = False
            st.session_state.applied = False

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# ======================================================================
# RESULT DISPLAY — persists across button clicks via session_state
# ======================================================================
if st.session_state.prediction_done:
    top_scheme = st.session_state.top_scheme
    top_conf = st.session_state.top_conf

    # ----- Recommended scheme card -----
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

    # ----- Top-3 alternatives -----
    st.markdown(
        '<div class="section-title">🏆 Top 3 Scheme Suggestions</div>',
        unsafe_allow_html=True,
    )
    for rank, (name, prob) in enumerate(st.session_state.top3, start=1):
        st.markdown(
            f"""
            <div class="alt-scheme">
                <div class="alt-scheme-name">#{rank} &nbsp; {name}</div>
                <div class="alt-scheme-conf">{prob * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ----- Probability chart -----
    st.markdown(
        '<div class="section-title">📊 Probability Distribution</div>',
        unsafe_allow_html=True,
    )
    chart_df = (
        pd.DataFrame(
            {"Scheme": list(st.session_state.all_proba.keys()),
             "Probability": list(st.session_state.all_proba.values())}
        )
        .sort_values("Probability", ascending=False)
        .set_index("Scheme")
    )
    st.bar_chart(chart_df, color="#43a047", height=350)

    # ----- Profile recap -----
    with st.expander("👀 View submitted farmer profile"):
        profile = st.session_state.user_inputs
        st.table(pd.DataFrame(profile.items(), columns=["Attribute", "Value"]))

    # ==================================================================
    # ACTION BUTTONS — View Details + Apply Now
    # ==================================================================
    st.markdown("---")
    st.markdown(
        '<div class="section-title">⚡ Next Steps</div>', unsafe_allow_html=True
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("📄 View Scheme Details", key="btn_details"):
            st.session_state.show_details = not st.session_state.show_details

    with btn_col2:
        if st.button("📝 Apply Now", key="btn_apply"):
            st.session_state.show_apply = not st.session_state.show_apply

    # ==================================================================
    # FEATURE 1 — SCHEME DETAILS VIEW
    # ==================================================================
    if st.session_state.show_details:
        st.markdown("")
        st.subheader("📄 Scheme Details")

        # Safe lookup using .get()
        details = scheme_data.get(top_scheme)

        if details is None:
            st.warning(
                f"⚠️ Detailed information for **{top_scheme}** is not available yet. "
                "Please check the official portal for more info."
            )
        else:
            st.markdown(
                f"""
                <div class="detail-card">
                    <h3>🌾 {top_scheme}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**📖 Description**")
            st.write(details.get("description", "Not available"))

            st.markdown("**✅ Eligibility Criteria**")
            st.markdown(details.get("eligibility", "Not available"))

            st.markdown("**🎁 Benefits**")
            st.markdown(details.get("benefits", "Not available"))

            st.markdown("**📑 Required Documents**")
            st.markdown(details.get("documents", "Not available"))

            link = details.get("link", "")
            if link:
                st.markdown("**🔗 Official Link**")
                st.markdown(f"[Visit official portal →]({link})")

    # ==================================================================
    # FEATURE 2 — AUTO-FILLED APPLICATION FORM
    # ==================================================================
    if st.session_state.show_apply:
        st.markdown("")
        st.subheader("📝 Application Form")
        st.caption(
            f"Apply for **{top_scheme}** — fields pre-filled from your profile."
        )

        ui = st.session_state.user_inputs

        with st.form("apply_form"):
            col_a, col_b = st.columns(2)

            with col_a:
                full_name = st.text_input("👤 Full Name *", placeholder="Enter your full name")
                phone = st.text_input("📞 Phone Number *", placeholder="10-digit mobile number")
                aadhaar = st.text_input("🆔 Aadhaar Number *", placeholder="XXXX-XXXX-XXXX")

                # Auto-filled from session_state
                st.selectbox(
                    "🌾 Land Size (auto-filled)",
                    ["Small", "Medium", "Large"],
                    index=["Small", "Medium", "Large"].index(ui.get("Land Size", "Small")),
                    disabled=True,
                    key="apply_land",
                )

            with col_b:
                email = st.text_input("✉️ Email", placeholder="optional")
                address = st.text_area("🏠 Address *", placeholder="Village, District, State, PIN")

                # Auto-filled from session_state
                st.selectbox(
                    "💰 Income Level (auto-filled)",
                    ["Low", "Medium", "High"],
                    index=["Low", "Medium", "High"].index(ui.get("Income Level", "Low")),
                    disabled=True,
                    key="apply_income",
                )
                st.selectbox(
                    "💧 Irrigation (auto-filled)",
                    ["Yes", "No"],
                    index=["Yes", "No"].index(ui.get("Irrigation", "Yes")),
                    disabled=True,
                    key="apply_irrigation",
                )

            st.markdown(
                "_\\* Required fields. Submitting this form creates a mock "
                "application ticket in-app for demo purposes._"
            )
            apply_submit = st.form_submit_button("✅ Submit Application")

            if apply_submit:
                # Basic validation
                if not full_name.strip() or not phone.strip() or not aadhaar.strip() or not address.strip():
                    st.error("⚠️ Please fill in all required fields (marked with *).")
                else:
                    st.session_state.applied = True
                    st.session_state.application = {
                        "Name": full_name,
                        "Phone": phone,
                        "Email": email or "—",
                        "Aadhaar": aadhaar,
                        "Address": address,
                        "Scheme": top_scheme,
                        **ui,
                    }

        # Success message outside the form so it stays after rerun
        if st.session_state.applied:
            st.success(
                f"🎉 Application submitted successfully for **{top_scheme}**! "
                "You will receive a confirmation on your registered phone number."
            )
            with st.expander("📋 Application Summary"):
                app_df = pd.DataFrame(
                    st.session_state.application.items(),
                    columns=["Field", "Value"],
                )
                st.table(app_df)

else:
    st.info(
        "👆 Fill in the farmer details above and click **Predict Recommended "
        "Scheme** to get a recommendation."
    )

# ======================================================================
# FOOTER
# ======================================================================
st.markdown(
    '<div class="footer">🌱 Smart Agriculture Scheme Advisor · Empowering farmers with AI</div>',
    unsafe_allow_html=True,
)
