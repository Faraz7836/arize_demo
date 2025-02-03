import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
from arize.pandas.logger import Client, Schema
from arize.utils.types import ModelTypes, Environments

# Arize API Credentials
API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE3Mzg1NjMyNTYsInVzZXJJZCI6MjEwMTksInV1aWQiOiJiNjEyOTBlNS02Nzg5LTQyMDctOWFjMC00MzczNjQ2Y2ZlNzAiLCJpc3MiOiJodHRwczovL2FwcC5hcml6ZS5jb20ifQ.3KKDYd0gI-ro80FcmWuatt9_DYhtJAjZQGnghSuG9u4"
SPACE_ID = "U3BhY2U6MTUxMDI6dURzQw=="
arize_client = Client(space_id=SPACE_ID, api_key=API_KEY)

# Load dataset
df = pd.read_csv("/Users/puravgupta/Desktop/python/folder/project/project1/streamli_test/arize_demo/world_population1.csv")
data = df.dropna().copy()  # Copy to avoid SettingWithCopyWarning

# Encode categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])  # Safe encoding

# Define features & target
x = data.drop(columns="World Population Percentage")
y = data["World Population Percentage"]

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

print(f"Number of unique values in y: {len(np.unique(y))}")
if len(np.unique(y)) < 10:  
    y_pred_class = np.round(y_pred) 
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Accuracy: {accuracy:.2f}")
else:
    print("Skipping accuracy_score because this is a regression problem.")

# Prepare data for Arize
x_test = x_test.copy()
x_test["prediction"] = y_pred  
x_test["actual"] = y_test.values  
x_test["id"] = range(len(x_test))  # Unique ID for Arize

# Reset index to avoid Arize Invalid_Index Error
x_test.reset_index(drop=True, inplace=True)

# Corrected Schema
schema = Schema(
    prediction_id_column_name="id",
    prediction_label_column_name="prediction",
    actual_label_column_name="actual",
    feature_column_names=list(x_test.columns)[:-3]  # Excluding added columns
)

# Log Data to Arize
response = arize_client.log(
    dataframe=x_test,
    schema=schema,
    model_id="world_population_model",
    model_version="v1.0",
    model_type=ModelTypes.REGRESSION,
    environment=Environments.PRODUCTION
)

print(f"Arize Response: {response}")
