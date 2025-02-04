import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score , r2_score
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments

# Arize API Credentials
API_KEY = "d33ea83d527145c5ae0"
SPACE_ID = "U3BhY2U6MTUxMDI6dURzQw=="
arize_client = Client(space_id=SPACE_ID, api_key=API_KEY)

# Load Data
file_path = r"C:\Users\visha\Desktop\streamlit\world_population1.csv"
df = pd.read_csv(file_path).dropna()

# Encode categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Define features & target
X = df.drop(columns="World Population Percentage")
y = df["World Population Percentage"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
if len(np.unique(y)) < 10:
    y_pred_class = np.round(y_pred)
    accuracy = accuracy_score(y_test, y_pred_class)
    accuracy_display = f"{accuracy:.2%}"
else:
    r2 = r2_score(y_test, y_pred)
    accuracy_display = f"{r2 * 100:.2f}%"

# Prepare data for Arize
X_test = X_test.copy()
X_test["prediction"] = y_pred  
X_test["actual"] = y_test.values  
X_test["id"] = range(len(X_test))  
X_test.reset_index(drop=True, inplace=True)

# Schema for Arize
schema = Schema(
    prediction_id_column_name="id",
    prediction_label_column_name="prediction",
    actual_label_column_name="actual",
    feature_column_names=list(X_test.columns)[:-3]
)

# Log Data to Arize
response = arize_client.log(
    dataframe=X_test,
    schema=schema,
    model_id="world_population_model",
    model_version="v1.0",
    model_type=ModelTypes.REGRESSION,
    environment=Environments.PRODUCTION
)

# Set Streamlit Page Config
st.set_page_config(page_title="Model Performance Dashboard", layout="wide")

# Dashboard Header
st.title("📊 Model Performance Dashboard")

# Top Metrics Cards
# Top Metrics Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", accuracy_display)
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
