import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the pre-trained model
model = joblib.load('LR.pkl')

# Feature names
feature_names = ["Gender", "Age", "PIR", "Drinking", "Sleep_disorder", "Moderate_physical_activity", "Total_cholesterol"]

# Streamlit user interface
st.title("Predictors of Depression in Stroke Patients")

# Collect user inputs
Gender = st.selectbox("Gender (0=Male, 1=Female):", options=[0, 1], format_func=lambda x: 'Male (0)' if x == 0 else 'Female (1)')
Age = st.number_input("Age:", min_value=20, max_value=85, value=50)
PIR = st.number_input("PIR:", min_value=0, max_value=5, value=3)
Drinking = st.selectbox("Drinking (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Sleep_disorder = st.selectbox("Sleep Disorder (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Moderate_physical_activity = st.selectbox("Moderate Physical Activity (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Total_cholesterol = st.number_input("Total Cholesterol:", min_value=2.07, max_value=9.98, value=6.0)

# Process inputs and make predictions
feature_values = [Gender, Age, PIR, Drinking, Sleep_disorder, Moderate_physical_activity, Total_cholesterol]
features = np.array([feature_values])

if st.button("Predict"):
    try:
        # Predict class and probabilities
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # Display prediction results
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Prediction Probabilities:** {predicted_proba}")

        # Generate advice based on prediction results
        probability = predicted_proba[predicted_class] * 100

        if predicted_class == 1:
            advice = (
                f"According to our model, you have a high risk of Depression. "
                f"The model predicts that your probability of having Depression is {probability:.1f}%. "
                "I recommend that you consult a doctor as soon as possible for further evaluation."
            )
        else:
            advice = (
                f"According to our model, you have a low risk of Depression. "
                f"The model predicts that your probability of not having Depression is {probability:.1f}%. "
                "However, maintaining a healthy lifestyle is still very important."
            )

        st.write(advice)

        # Calculate SHAP values and display force plot
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.Explainer(model, features)
            shap_values = explainer(features)

            # Create a force plot
            shap.force_plot(explainer.expected_value, shap_values.values, features, matplotlib=True)
            plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

        # Display the SHAP force plot image
        st.image("shap_force_plot.png")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
