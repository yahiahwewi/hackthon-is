import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. CONFIG & STYLE
# ==========================================
st.set_page_config(
    page_title="NeuroGuard | Stroke Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look (Glassmorphism + Animations)
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Cards/Containers */
    .css-1r6slb0, .stButton>button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border-color: #38bdf8;
    }

    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700;
    }
    
    h1 {
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Result Badges */
    .risk-badge {
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
        animation: fadeIn 0.8s ease-out;
    }
    
    .risk-low {
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid #22c55e;
        color: #4ade80;
        box-shadow: 0 0 20px rgba(34, 197, 94, 0.2);
    }
    
    .risk-high {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid #ef4444;
        color: #f87171;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.2);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    try:
        artifact = joblib.load('model.pkl')
        return artifact
    except FileNotFoundError:
        st.error("Model file not found. Please run train.py first.")
        return None

artifact = load_model()

# ==========================================
# 3. SIDEBAR - INPUTS
# ==========================================
with st.sidebar:
    st.title("Patient Data")
    st.markdown("Enter clinical parameters below.")
    
    # Demographics
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 0, 100, 50)
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    marital = st.selectbox("Marital Status", ["Married", "Not Married"])
    work = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])

    # Physiology
    st.subheader("Physiology")
    glucose = st.number_input("Avg. Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.number_input("BMI (Leave 0 if unknown)", 0.0, 100.0, 0.0)
    
    # History
    st.subheader("Medical History")
    hypertension = st.checkbox("Hypertension")
    heart_disease = st.checkbox("Heart Disease")
    smoking = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# ==========================================
# 4. MAIN LOGIC
# ==========================================
st.title("🧠 NeuroGuard AI")
st.markdown("### Intelligent Stroke Risk Prediction System")

if st.button("Analyze Risk Profile", type="primary"):
    if artifact:
        model = artifact['model']
        base_threshold = artifact['base_threshold']
        
        # Prepare Input Data
        # We must match the columns expected by the model
        # Note: The model pipeline handles OneHotEncoding, so we pass raw strings
        
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [1 if hypertension else 0],
            'heart_disease': [1 if heart_disease else 0],
            'ever_married': ["Yes" if marital == "Married" else "No"],
            'work_type': [work],
            'Residence_type': [residence],
            'avg_glucose_level': [glucose],
            'bmi': [np.nan if bmi == 0 else bmi], # Handle missing BMI
            'smoking_status': [smoking]
        })

        # Predict Probability
        prob = model.predict_proba(input_data)[0][1]
        
        # Smart Prediction Logic (Dynamic Thresholds)
        # Example: Lower threshold for elderly with high glucose
        adjusted_threshold = base_threshold
        risk_factors = []
        
        if age > 60:
            adjusted_threshold -= 0.05
            risk_factors.append("Advanced Age (>60)")
        
        if glucose > 200:
            adjusted_threshold -= 0.05
            risk_factors.append("High Glucose Levels (>200)")
            
        if hypertension:
            risk_factors.append("History of Hypertension")
            
        if heart_disease:
            risk_factors.append("History of Heart Disease")

        # Final Decision
        is_stroke_risk = prob >= adjusted_threshold
        
        # Display Results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if is_stroke_risk:
                st.markdown(f"""
                <div class="risk-badge risk-high">
                    ⚠️ HIGH RISK DETECTED<br>
                    <span style="font-size: 16px; opacity: 0.8">Probability: {prob:.1%} (Threshold: {adjusted_threshold:.1%})</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-badge risk-low">
                    ✅ LOW RISK PROFILE<br>
                    <span style="font-size: 16px; opacity: 0.8">Probability: {prob:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
                
        with col2:
            st.metric("Risk Probability", f"{prob:.1%}")
            st.metric("Model Threshold", f"{adjusted_threshold:.1%}")

        # Explainability
        st.markdown("### 🔍 Analysis Report")
        if risk_factors:
            st.warning(f"**Key Aggravating Factors:** {', '.join(risk_factors)}")
            st.markdown("These factors significantly contribute to the elevated risk profile according to the AI model.")
        else:
            st.success("No major aggravating clinical factors detected.")

else:
    st.info("👈 Please configure the patient profile in the sidebar and click 'Analyze Risk Profile'.")

# Footer
st.markdown("---")
st.markdown("*Powered by Advanced Random Forest & SMOTE Technology*")
