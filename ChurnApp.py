import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# 1. App title
st.title("Customer Churn Prediction App")

# 2. Load the predetermined CSV file
data = pd.read_csv('Churn_Modelling.csv')  # Replace with the path to your predetermined CSV file

# Display dataset preview
st.subheader("Dataset Preview")
st.write(data.head())

# 3. Data Preprocessing
# Define features (x) and target (y)
x = data.iloc[:, 3:-1]
y = data.iloc[:, -1].values

# Encoding categorical features
le_geography = LabelEncoder()
le_gender = LabelEncoder()

# Fit the encoders on the dataset
x['Geography'] = le_geography.fit_transform(x['Geography'])
x['Gender'] = le_gender.fit_transform(x['Gender'])

# Feature scaling
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# 4. Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# 5. Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# Model evaluation
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Predict on new input
st.subheader("Predict Churn for a New Customer")

# Create form to accept new customer details
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1, value=650)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, step=1, value=35)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, step=1, value=5)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, step=1, value=1)
has_credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Convert Yes/No input to 1/0
has_credit_card_encoded = 1 if has_credit_card == "Yes" else 0
is_active_member_encoded = 1 if is_active_member == "Yes" else 0

# Button to initiate prediction
if st.button("Predict Churn"):
    # Encode inputs
    geography_encoded = le_geography.transform([geography])[0]
    gender_encoded = le_gender.transform([gender])[0]

    input_data = sc.transform([[credit_score, geography_encoded, gender_encoded, age, tenure, 
                                balance, num_of_products, has_credit_card_encoded, is_active_member_encoded, estimated_salary]])

    # Predict churn
    prediction = clf.predict(input_data)
    if prediction[0] == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is not likely to churn.")
