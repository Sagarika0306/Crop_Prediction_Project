import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Streamlit UI setup
st.title("🌱 Crop Prediction using Machine Learning")

# File uploader
uploaded_file = st.file_uploader("Upload Crop Recommendation CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Display dataset
    if st.checkbox("Show dataset"):
        st.write(df.head())
    
    # Preprocessing
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    # Splitting data
    X = df.drop(columns=['label'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {accuracy:.2f}")
    
    # User input fields
    st.header("Enter Soil and Climate Parameters")
    N = st.number_input("Nitrogen (N) level", min_value=0.0, format="%.2f")
    P = st.number_input("Phosphorus (P) level", min_value=0.0, format="%.2f")
    K = st.number_input("Potassium (K) level", min_value=0.0, format="%.2f")
    temperature = st.number_input("Temperature (°C)", format="%.2f")
    humidity = st.number_input("Humidity (%)", format="%.2f")
    ph = st.number_input("pH value", format="%.2f")
    rainfall = st.number_input("Rainfall (mm)", format="%.2f")
    
    if st.button("Predict Crop"):
        new_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        predicted_label = model.predict(new_data)
        predicted_crop = label_encoder.inverse_transform(predicted_label)
        st.success(f"✅ Recommended Crop: {predicted_crop[0]}")
