import streamlit as st
import pandas as pd
import numpy as np

# Set the title and description of the app
st.title("Interactive Data Visualization")
st.markdown("""
This app allows you to visualize random data using different chart types.
Use the filters below the chart to select the chart type and filter the data by date.
""")

# Create a DataFrame with specific data
data = {
    "Date": pd.date_range(start="2023-01-01", periods=10, freq="D"),
    "Value 1": np.random.randn(10),
    "Value 2": np.random.randn(10),
    "Value 3": np.random.randn(10)
}
df = pd.DataFrame(data)

# Display chart based on selection
st.write("## Chart")
chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Area Chart"])

# Display the initial chart
if chart_type == "Line Chart":
    st.line_chart(df.set_index("Date"))
elif chart_type == "Bar Chart":
    st.bar_chart(df.set_index("Date"))
elif chart_type == "Area Chart":
    st.area_chart(df.set_index("Date"))
else:
    st.write("No data available for the selected date range.")

# Filters below the chart
st.write("## Filters")
date_range = st.date_input("Filter by Date Range", [])

# Filter DataFrame by date range
if date_range:
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

# Display the filtered chart



# Display DataFrame
st.write("## Data Table")
st.dataframe(df.style.format({"Value 1": "{:.2f}", "Value 2": "{:.2f}", "Value 3": "{:.2f}"}))