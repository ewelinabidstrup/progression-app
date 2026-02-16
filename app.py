import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

# Page config
st.set_page_config(page_title="Progression Predictor", page_icon="🔬")

st.title("🔬 Progression Predictor")
st.write("Enter the three biomarker values to predict the probabiliy of a patient having a disease progression")

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    b1tp53 = st.number_input("B1TP53 copy/ml", value=0.0, format="%.4f")

with col2:
    b2mdm2 = st.number_input("B2MDM2 copy/ml", value=0.0, format="%.4f")

with col3:
    cfdnab3 = st.number_input("cfDNAB3 copy/ml", value=0.0, format="%.4f")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('progression_model.pkl')

model = load_model()

# Predict button
# Predict button
if st.button("Predict", type="primary"):
    # Create input array with the actual numeric values entered by user
    input_data = np.array([[b1tp53, b2mdm2, cfdnab3]], dtype=np.float32)
    
    # Apply the same preprocessing as training: log1p transformation
    input_data = np.log1p(input_data)
    
    # Get prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Display results
    st.divider()
    
    if prediction == 1:
        st.error(f"⚠️ Prediction: Progression likely")
    else:
        st.success(f"✅ Prediction: Progression unlikely")
    
    st.write(f"**Probability of no progression:** {probability[0]:.1%}")
    st.write(f"**Probability of progression:** {probability[1]:.1%}")