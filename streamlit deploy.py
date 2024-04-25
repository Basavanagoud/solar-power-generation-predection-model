# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:39:49 2024

@author: PMC
"""

import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("rfc.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Define the prediction function
def predict_defect(data):
    prediction = rf_model.predict(data)
    return prediction

# Load data
df = pd.read_excel(r"F:/project 360 digitmg/project 2/Solar Power Generation Prediction Model/Solar Power Generation Prediction Model/GPVS-Faults.xlsx")
# Rename the column to remove spaces and special characters
df.rename(columns={'Defective/Non Defective ': 'Defective_Non_Defective'}, inplace=True)

# Get list of feature columns
feature_columns = [col for col in df.columns if col != 'Defective_Non_Defective']

# Create the Streamlit app
def main():
    # Set title and description
    st.title("Solar Power Generation Prediction")
    st.write("This app predicts whether a solar power generation system is defective or non-defective based on input data.")

    # Add user input for feature values
    st.sidebar.header("Input Features")
    input_data = {}
    for feature in feature_columns:
        input_data[feature] = st.sidebar.number_input(feature, value=0.0)

    # Prepare input data as a DataFrame
    input_df = pd.DataFrame([input_data])

    # Make predictions
    if st.sidebar.button("Predict"):
        prediction = predict_defect(input_df)
        st.write("Prediction:", prediction[0])

if __name__ == "__main__":
    main()

  
  
    


