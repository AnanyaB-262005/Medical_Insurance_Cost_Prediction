# app.py

import streamlit as st
import joblib

# Load trained model and encoders
model_data = joblib.load("insurance_model.pkl")
model = model_data["model"]
le_sex = model_data["le_sex"]
le_smoker = model_data["le_smoker"]
le_region = model_data["le_region"]

def main():
    st.markdown("""
        <div style="background-color:lightblue;padding:16px">
        <h2 style="color:black;text-align:center">Health Insurance Cost Prediction</h2>
        </div> 
    """, unsafe_allow_html=True)

    age = st.slider("Enter Your Age", 18, 100)
    sex = st.selectbox("Sex", le_sex.classes_)
    bmi = st.number_input("Enter Your BMI Value", min_value=10.0, max_value=50.0)
    children = st.slider("Enter Number of Children", 0, 5)
    smoker = st.selectbox("Smoker", le_smoker.classes_)
    region = st.selectbox("Region", le_region.classes_)

    # Encode inputs
    sex_encoded = le_sex.transform([sex])[0]
    smoker_encoded = le_smoker.transform([smoker])[0]
    region_encoded = le_region.transform([region])[0]

    if st.button("Predict"):
        features = [[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]]
        prediction = model.predict(features)
        st.success(f"💰 Predicted Insurance Cost: ${prediction[0]:,.2f}")
        st.balloons()

if __name__ == '__main__':
    main()
