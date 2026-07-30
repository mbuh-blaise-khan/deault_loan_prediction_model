import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="🏦 Credit Risk Assessment System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    # Try multiple paths
    paths = [
        'xgb.joblib',
        '../models/xgb.joblib',
        'models/xgb.joblib'
    ]
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path)
    st.error("⚠️ Model file not found. Please ensure 'xgb.joblib' is in the app folder.")
    st.stop()

model = load_model()

# ============================================
# FEATURE COLUMNS (Must match training order)
# ============================================
FEATURE_COLUMNS = [
    'loanamount_x', 'totaldue_x', 'termdays_x', 'longitude_gps', 'latitude_gps',
    'loanamount_y', 'totaldue_y', 'termdays_y', 'age', 'first_payment_delay_days',
    'prev_actual_days', 'is_weekend', 'daily_payment_burden',
    'employment_status_clients_Permanent', 'employment_status_clients_Retired',
    'employment_status_clients_Self-Employed', 'employment_status_clients_Student',
    'employment_status_clients_Unemployed', 'bank_account_type_Other',
    'bank_account_type_Savings', 'bank_name_encoded'
]

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_loan_status(
    loanamount_x, totaldue_x, termdays_x, longitude_gps, latitude_gps,
    loanamount_y, totaldue_y, termdays_y, age, first_payment_delay_days,
    prev_actual_days, is_weekend, daily_payment_burden, employment_status,
    bank_account_type, bank_name_encoded
):
    # Create input dictionary
    input_data = {
        'loanamount_x': float(loanamount_x),
        'totaldue_x': float(totaldue_x),
        'termdays_x': int(termdays_x),
        'longitude_gps': float(longitude_gps),
        'latitude_gps': float(latitude_gps),
        'loanamount_y': float(loanamount_y),
        'totaldue_y': float(totaldue_y),
        'termdays_y': float(termdays_y),
        'age': int(age),
        'first_payment_delay_days': int(first_payment_delay_days),
        'prev_actual_days': int(prev_actual_days),
        'is_weekend': 1 if is_weekend == "Yes" else 0,
        'daily_payment_burden': float(daily_payment_burden),
        'employment_status_clients_Permanent': 1 if employment_status == "Permanent" else 0,
        'employment_status_clients_Retired': 1 if employment_status == "Retired" else 0,
        'employment_status_clients_Self-Employed': 1 if employment_status == "Self-Employed" else 0,
        'employment_status_clients_Student': 1 if employment_status == "Student" else 0,
        'employment_status_clients_Unemployed': 1 if employment_status == "Unemployed" else 0,
        'bank_account_type_Other': 1 if bank_account_type == "Other" else 0,
        'bank_account_type_Savings': 1 if bank_account_type == "Savings" else 0,
        'bank_name_encoded': float(bank_name_encoded)
    }
    
    # Create DataFrame and reorder columns
    input_df = pd.DataFrame([input_data])
    input_df = input_df[FEATURE_COLUMNS]
    
    # Predict
    probabilities = model.predict_proba(input_df)[0]
    bad_prob, good_prob = probabilities[0], probabilities[1]
    
    return good_prob, bad_prob

# ============================================
# HEADER
# ============================================
st.markdown("""
<div style="text-align: center; padding: 20px 0; border-bottom: 3px solid #1565c0; margin-bottom: 20px;">
    <h1 style="color: #0d47a1; font-size: 40px; margin: 0;">🏦 Credit Risk Assessment</h1>
    <p style="color: #1a237e; font-size: 18px; margin: 5px 0 0 0;">Predict loan default risk using machine learning</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - INPUTS
# ============================================
with st.sidebar:
    st.header("📝 Enter Client Details")
    st.markdown("---")
    
    st.subheader("📦 Current Loan Data")
    loanamount_x = st.number_input("Loan Amount (x)", value=30000.0, step=1000.0)
    totaldue_x = st.number_input("Total Due (x)", value=35000.0, step=1000.0)
    termdays_x = st.slider("Term Days (x)", 7, 90, 30, 1)
    daily_payment_burden = st.number_input("Daily Payment Burden", value=1166.0, step=50.0)
    is_weekend = st.radio("Is Weekend Request?", ["No", "Yes"], index=0)
    
    st.markdown("---")
    st.subheader("⏳ Historical Metrics")
    age = st.number_input("Client Age", value=32, step=1)
    first_payment_delay_days = st.number_input("First Payment Delay Days", value=0, step=1)
    prev_actual_days = st.number_input("Previous Actual Days Taken", value=14, step=1)
    loanamount_y = st.number_input("Previous Loan Amount (y)", value=20000.0, step=1000.0)
    totaldue_y = st.number_input("Previous Total Due (y)", value=24500.0, step=1000.0)
    termdays_y = st.number_input("Previous Term Days (y)", value=30, step=1)
    
    st.markdown("---")
    st.subheader("🔑 Demographics")
    employment_status = st.radio(
        "Employment Status",
        ["Permanent", "Self-Employed", "Student", "Unemployed", "Retired"],
        index=0
    )
    bank_account_type = st.radio(
        "Bank Account Type",
        ["Savings", "Other"],
        index=0
    )
    bank_name_encoded = st.number_input("Bank Name Encoded Value", value=0.15, step=0.01)
    longitude_gps = st.number_input("GPS Longitude", value=3.379, step=0.001)
    latitude_gps = st.number_input("GPS Latitude", value=6.524, step=0.001)
    
    predict_btn = st.button("🔮 Evaluate Risk", type="primary", use_container_width=True)

# ============================================
# MAIN CONTENT
# ============================================
if predict_btn:
    with st.spinner("🔮 Analyzing application..."):
        good_prob, bad_prob = predict_loan_status(
            loanamount_x, totaldue_x, termdays_x, longitude_gps, latitude_gps,
            loanamount_y, totaldue_y, termdays_y, age, first_payment_delay_days,
            prev_actual_days, is_weekend, daily_payment_burden, employment_status,
            bank_account_type, bank_name_encoded
        )
        
        # Determine result
        if good_prob >= 0.5:
            status = "✅ APPROVED"
            status_color = "#2e7d32"
            bg_color = "#e8f5e9"
            recommendation = "Low default risk. Loan can be approved."
        else:
            status = "❌ REJECTED"
            status_color = "#c62828"
            bg_color = "#ffebee"
            recommendation = "High default risk. Loan should be rejected."
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Risk Assessment Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Decision",
                value=status,
                delta="Pass" if good_prob >= 0.5 else "Fail"
            )
        
        with col2:
            st.metric(
                label="Confidence - Good",
                value=f"{good_prob:.1%}",
                delta="Low Risk" if good_prob >= 0.5 else "High Risk"
            )
        
        with col3:
            st.metric(
                label="Confidence - Bad",
                value=f"{bad_prob:.1%}",
                delta="High Risk" if bad_prob > 0.5 else "Low Risk"
            )
        
        # Gauge chart
        st.markdown("---")
        st.subheader("📊 Risk Score Gauge")
        
        risk_score = bad_prob * 100
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Risk Score %"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#0d47a1'},
                'steps': [
                    {'range': [0, 30], 'color': '#e8f5e9'},
                    {'range': [30, 60], 'color': '#fff3e0'},
                    {'range': [60, 100], 'color': '#ffebee'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        st.markdown("---")
        st.subheader("💡 Recommendation")
        if good_prob >= 0.5:
            st.success(f"✅ {recommendation}")
            st.balloons()
        else:
            st.error(f"❌ {recommendation}")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #1a237e; font-size: 0.9em; padding: 15px 0;">
    Built with ❤️ using XGBoost | SEED AIML Internship 2026
    <br>
    <span style="color: #0d47a1;">Credit Risk Assessment System</span>
</div>
""", unsafe_allow_html=True)