import streamlit as st
import numpy as np

# Page config
st.set_page_config(page_title="Progression Predictor", page_icon="🔬")

# Tree structure (no patient data, just decision rules)
TREE = {
    'children_left': [1, 2, -1, -1, 5, -1, -1],
    'children_right': [4, 3, -1, -1, 6, -1, -1],
    'feature': [1, 1, -2, -2, 0, -2, -2],
    'threshold': [4.59741735458374, 2.4343918561935425, -2.0, -2.0, 3.9661738872528076, -2.0, -2.0],
    'value': [[[17.999999999999996, 18.0]], [[12.214285714285714, 4.5]], [[0.6428571428571429, 4.5]], [[11.571428571428571, 0.0]], [[5.7857142857142865, 13.5]], [[0.6428571428571429, 11.25]], [[5.142857142857143, 2.25]]]
}

def predict_proba(features):
    """Traverse the decision tree and return probabilities."""
    node = 0
    while TREE['children_left'][node] != -1:
        if features[TREE['feature'][node]] <= TREE['threshold'][node]:
            node = TREE['children_left'][node]
        else:
            node = TREE['children_right'][node]
    
    # Get class counts at leaf node
    value = TREE['value'][node][0]
    total = sum(value)
    proba = [v / total for v in value]
    return proba

def predict(features):
    """Return predicted class."""
    proba = predict_proba(features)
    return 0 if proba[0] > proba[1] else 1

st.title("🔬 Progression Predictor")
st.write("Enter the three biomarker values to predict progression probability.")

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    b1tp53 = st.number_input("B1TP53", value=0.0, format="%.4f")

with col2:
    b2mdm2 = st.number_input("B2MDM2", value=0.0, format="%.4f")

with col3:
    cfdnab3 = st.number_input("cfDNAB3", value=0.0, format="%.4f")

# Predict button
if st.button("Predict", type="primary"):
    # Apply log1p transformation (same as training)
    features = [
        np.log1p(b1tp53),
        np.log1p(b2mdm2),
        np.log1p(cfdnab3)
    ]
    
    # Get prediction
    prediction = predict(features)
    probability = predict_proba(features)
    
    # Display results
    st.divider()
    
    if prediction == 1:
        st.error(f"⚠️ Prediction: Progression likely")
    else:
        st.success(f"✅ Prediction: Progression unlikely")
    
    st.write(f"**Probability of no progression:** {probability[0]:.1%}")
    st.write(f"**Probability of progression:** {probability[1]:.1%}")
