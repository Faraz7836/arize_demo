import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,accuracy_score
from arize.pandas.logger import Client,Schema

API_KEY="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE3Mzg1NjMyNTYsInVzZXJJZCI6MjEwMTksInV1aWQiOiJiNjEyOTBlNS02Nzg5LTQyMDctOWFjMC00MzczNjQ2Y2ZlNzAiLCJpc3MiOiJodHRwczovL2FwcC5hcml6ZS5jb20ifQ.3KKDYd0gI-ro80FcmWuatt9_DYhtJAjZQGnghSuG9u4"
SPACE_ID="U3BhY2U6MTUxMDI6dURzQw=="
arize_client=Client(space_id=SPACE_ID, api_key=API_KEY)

df=pd.read_csv("/Users/puravgupta/Desktop/python/folder/project/project1/streamli_test/arize_demo/world_population1.csv")

data = df.dropna()

categorical_columns = data.select_dtypes(include=['object']).columns

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])  # Encode the categorical column
    label_encoders[col] = le 

x=data.drop(columns="World Population Percentage")
y=data["World Population Percentage"]

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.3, random_state=42)

label_encoder=LabelEncoder()

model=LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


if len(np.unique(y)) < 10:  
    y_pred_class = np.round(y_pred) 
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Accuracy: {accuracy:.2f}")


#     Log modal data to arize

