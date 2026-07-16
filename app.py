import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------------------------
# 1. PAGE CONFIG & STYLING
# ---------------------------------------------------------
st.set_page_config(page_title="HR Attrition & Burnout Tracker", page_icon="🏢", layout="wide")

# Sedikit custom CSS biar tampilannya lebih rapi
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2c3e50;}
    .stAlert {font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏢 HR Attrition & Burnout Predictor (Advanced Model)")
st.markdown("Predict employee resignation risk based on demographics, commute, and workplace environment.")

# ---------------------------------------------------------
# 2. LOAD MODEL & DATA
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_features():
    model = joblib.load('xgb_attrition_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv('HR_Analytics_Engineered_Data.csv')

try:
    model, expected_features = load_model_and_features()
    df = load_data()
except FileNotFoundError:
    st.error("Error: Model files or dataset not found. Please run data_prep.py and train_model.py first.")
    st.stop()

# ---------------------------------------------------------
# 3. SIDEBAR: HR CONTROL PANEL (INPUT)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Employee Profile Settings")
st.sidebar.markdown("Adjust parameters to simulate retention strategies.")

# Mengambil nilai unik untuk dropdown dari dataset
departments = df['Department'].unique()
job_roles = df['JobRole'].unique()
overtime_options = df['OverTime'].unique()

# User Inputs di Sidebar
selected_dept = st.sidebar.selectbox("Department", departments)
selected_role = st.sidebar.selectbox("Job Role", job_roles)

st.sidebar.markdown("---")
st.sidebar.subheader("Key Burnout Indicators")

# Menggunakan fitur yang sudah kita engineer sebelumnya
commute_dist = st.sidebar.slider("Commute Distance (KM)", 1, 105, 15)
overtime_status = st.sidebar.selectbox("OverTime", overtime_options)
toxic_score = st.sidebar.slider("Manager 'Toxicity' Score (1-10)", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.subheader("Demographics & Comp")
age = st.sidebar.slider("Age", 18, 65, 30)
monthly_income = st.sidebar.slider("Monthly Income (USD)", int(df['MonthlyIncome'].min()), int(df['MonthlyIncome'].max()), 5000)
dependents = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3, 4])

# ---------------------------------------------------------
# 4. PREPARE INPUT FOR PREDICTION
# ---------------------------------------------------------
# Buat dataframe dari input user (dengan nilai default/median untuk kolom lain agar prediksi jalan)
input_data = pd.DataFrame(columns=df.drop(columns=['Attrition']).columns)
input_data.loc[0] = df.drop(columns=['Attrition']).median(numeric_only=True) # Isi sisa kolom dengan median

# Update kolom sesuai input user
input_data['Department'] = selected_dept
input_data['JobRole'] = selected_role
input_data['Real_Commute_Distance_KM'] = commute_dist
input_data['OverTime'] = overtime_status
input_data['Toxic_Manager_Score'] = toxic_score
input_data['Age'] = age
input_data['MonthlyIncome'] = monthly_income
input_data['Number_of_Dependents'] = dependents

# Encoding input data supaya formatnya sama persis dengan saat training
input_encoded = pd.get_dummies(input_data)

# Reindex agar urutan kolomnya pas dengan model_features (isi nilai 0 untuk kolom yang hilang akibat dummy encoding)
input_final = input_encoded.reindex(columns=expected_features, fill_value=0)

# ---------------------------------------------------------
# 5. PREDICTION & DASHBOARD DISPLAY
# ---------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Actionable Insight")
    if st.button("Calculate Attrition Risk", type="primary"):
        # Prediksi Probabilitas
        prob_resign = model.predict_proba(input_final)[0][1] * 100
        
        st.markdown(f"### Risk Score: {prob_resign:.1f}%")
        
        # Logika Tampilan Berdasarkan Risiko
        if prob_resign >= 70:
            st.error("🚨 CRITICAL RISK: High probability of resignation.")
            st.markdown("**Recommendation:** Immediate 1-on-1 intervention required. Consider offering WFH options if commute distance is high, or review manager workload.")
        elif prob_resign >= 40:
            st.warning("⚠️ MEDIUM RISK: Employee is evaluating options.")
            st.markdown("**Recommendation:** Monitor overtime and check for engagement drops. Evaluate salary competitiveness.")
        else:
            st.success("✅ LOW RISK: Employee is currently stable.")
            st.markdown("**Recommendation:** Continue regular check-ins. Acknowledge good performance.")

with col2:
    st.subheader("Why this matters?")
    st.info("""
    **Business Context:**
    This model utilizes advanced machine learning (XGBoost) to identify hidden patterns that lead to employee turnover. 
    
    Notice how altering the **Commute Distance** or **OverTime** status dramatically shifts the risk probability. This tool allows HR to move from reactive replacement to proactive retention.
    """)
    
    # Menampilkan data sampel dari departemen yang dipilih
    st.markdown(f"**Sample Data from {selected_dept} Department:**")
    st.dataframe(df[df['Department'] == selected_dept][['Age', 'JobRole', 'Real_Commute_Distance_KM', 'OverTime', 'Attrition']].head())