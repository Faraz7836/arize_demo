import streamlit as st
import pandas as pd
import numpy as np

# Set the title and description of the app
st.title("Interactive Data Visualization")
st.markdown("""
This app allows you to visualize random data using different chart types.
Use the sidebar to select the chart type and filter the data by date.
""")

# Create a DataFrame with specific data
data = {
    "Date": pd.date_range(start="2023-01-01", periods=10, freq="D"),
    "Value 1": np.random.randn(10),
    "Value 2": np.random.randn(10),
    "Value 3": np.random.randn(10)
}
df = pd.DataFrame(data)

# Sidebar for chart selection and filters
st.sidebar.header("Chart Options")
chart_type = st.sidebar.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Area Chart"])
date_filter = st.sidebar.date_input("Filter by Date", [])

# Filter DataFrame by date
if date_filter:
    if isinstance(date_filter, list) and len(date_filter) > 0:
        df = df[df["Date"].isin(date_filter)]
    elif isinstance(date_filter, pd.Timestamp):
        df = df[df["Date"] == date_filter]

# Display DataFrame
st.write("## Data Table")
st.dataframe(df.style.format({"Value 1": "{:.2f}", "Value 2": "{:.2f}", "Value 3": "{:.2f}"}))

# Display chart based on selection
st.write("## Chart")
if not df.empty:
    if chart_type == "Line Chart":
        st.line_chart(df.set_index("Date"))
    elif chart_type == "Bar Chart":
        st.bar_chart(df.set_index("Date"))
    elif chart_type == "Area Chart":
        st.area_chart(df.set_index("Date"))
else:
    st.write("No data available for the selected date(s).")
    
    # Custom CSS for additional styling
st.markdown(
    """
    <style>
    .stDataFrame {border: 2px solid #4CAF50; border-radius: 10px;}
    .stDataFrame table {border-collapse: collapse; width: 100%;}
    .stDataFrame th, .stDataFrame td {text-align: center; padding: 8px;}
    .stDataFrame th {background-color: #4CAF50; color: white;}
    .stDataFrame tr:nth-child(even) {background-color: #f2f2f2;}
    </style>
    """,
    unsafe_allow_html=True
)