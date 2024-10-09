import streamlit as st
import pandas as pd
import pickle

# Load the saved Random Forest model
with open('Customer_Churn_Prediction.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to preprocess input data
def preprocess_input(data):
    # Perform one-hot encoding on Geography and Gender
    data = pd.get_dummies(data, drop_first=True)
    return data

# Function to make predictions
def make_prediction(features):
    # Preprocess input data
    input_data = pd.DataFrame([features])
    input_data = preprocess_input(input_data)

    # Make prediction using the model
    prediction = model.predict(input_data)

    return prediction[0]

# Streamlit app
def main():
    st.title("Netflix Churn Prediction App")

    # Collect user input for features
    st.sidebar.header("User Input Features")
    features = {}

    features['CreditScore'] = st.sidebar.slider("Credit Score", 300, 850, 650)
    features['Age'] = st.sidebar.slider("Age", 18, 100, 35)
    fe
    # Add other features as needed

    # Make prediction when the user clicks the "Predict" button
    if st.sidebar.button("Predict"):
        prediction = make_prediction(features)
        st.sidebar.success(f"The predicted outcome is {prediction}")

if __name__ == '__main__':
    main()
