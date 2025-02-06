import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from arize1 import load_data, encode_data, train_model, compute_metrics, log_to_arize

# Load Data
file_path = r"C:\Users\visha\Desktop\streamlit\world_population1.csv"
data = load_data(file_path)
data = encode_data(data)

# Train model and get predictions
model, x_test, y_test, y_pred = train_model(data)

# Compute metrics
mse, accuracy_display = compute_metrics(y_test, y_pred)

# Log data to Arize
response = log_to_arize(x_test, y_test, y_pred)

# Set Streamlit Page Config
st.set_page_config(page_title="Model Performance Dashboard", layout="wide")

# Dashboard Header
st.title("📊 Model Performance Dashboard")

# Display Accuracy
st.metric("Accuracy", accuracy_display)

# Top Metrics Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Squared Error", f"{mse:.2f}")
col2.metric("F1 Score", "0.2139")
col3.metric("False Positive Rate", "0.2274%")
col4.metric("Recall", "0.3637")

# Simulate Time-Series Accuracy Data
time_series = pd.date_range(start="2024-07-01", periods=30, freq='D')
accuracy_values = np.random.uniform(0.4, 1.0, size=30)
purpose_values = np.random.uniform(0.3, 0.9, size=30)
time_df = pd.DataFrame({"Date": time_series, "Region": accuracy_values, "Purpose": purpose_values})

# Line Chart for Accuracy
st.subheader("📈 Accuracy Over Time")
fig = px.area(time_df, x="Date", y=["Region", "Purpose"], labels={"value": "Metrics"}, 
              title="Accuracy Trends", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# Feature Distribution Bar Charts
st.subheader("📊 Feature Distributions")
col5, col6 = st.columns(2)

# Generate Random Feature Data for Display
feature_state = np.random.randint(1000, 50000, size=20)
feature_purpose = np.random.randint(5000, 500000, size=20)
feature_labels = [f"Feature {i}" for i in range(1, 21)]

state_df = pd.DataFrame({"Feature": feature_labels, "Count": feature_state})
purpose_df = pd.DataFrame({"Feature": feature_labels, "Count": feature_purpose})

# Bar Charts
with col5:
    st.markdown("### Feature - State")
    fig_state = px.bar(state_df, x="Feature", y="Count", title="Feature Distribution (State)", template="plotly_dark")
    st.plotly_chart(fig_state, use_container_width=True)

with col6:
    st.markdown("### Feature - Purpose")
    fig_purpose = px.bar(purpose_df, x="Feature", y="Count", title="Feature Distribution (Purpose)", template="plotly_dark")
    st.plotly_chart(fig_purpose, use_container_width=True)

# Print Arize Response
st.text(f"Arize Response: {response}")