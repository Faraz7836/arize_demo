import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from arize.api import Client
from arize.pandas.logger import Schema,Environments,ModelTypes
from sklearn.metrics import mean_squared_error, r2_score
from arize.pandas.logger import Client, Schema, Environments, ModelTypes




#                                           load the dataset

data = pd.read_csv("arize_demo/Salary.csv")
# print(data.head())


#                                           feature independent and dependent variable

x = data ["YearsExperience"].values.reshape(-1,1)
y = data ["Salary"].values

#                                           split the dataset

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#                                           train the model

model = LinearRegression()  
model.fit(x_train, y_train)

#                                           prediction

y_pred = model.predict(x_test)      


#                                           save the model 

mse=mean_squared_error(y_test, y_pred)
print(mse)

r2=r2_score(y_test, y_pred)
print(f"r2 score is {r2}, mse is {mse}")


#                                          Api set up for arize ai

arize_client = Client(space_id="U3BhY2U6MTUwNTk6Y1JQdw==", api_key="d64ab228893fc810549")

#                                          prepare the data for arize   

schema= Schema(
    prediction_id_column_name="id",
    prediction_label_column_name="prediction",
    actual_label_column_name="actual",
    feature_column_names=["YearsExperience"]    
    
)

#                                       log Trianing data to arize

traing_data = pd.DataFrame({
    'id': range(len(x_train)),
    'actual': y_train,
    'prediction': model.predict(x_train),
    'YearsExperience': x_train.flatten()
})




response = arize_client.log(
    dataframe=traing_data,
    schema=schema,
    model_id="salary_model",
    model_version="1.0",
    environment=Environments.PRODUCTION,
    model_type=ModelTypes.REGRESSION,

)

print("Training Data logged successfully!") if response.status_code == 200 else print(f"Failed: {response.text}")




prediction_data = pd.DataFrame({
    'id': range(len(x_test)),
    'actual': y_test,
    'prediction': y_pred,
    'YearsExperience': x_test.flatten()
})




#                                          log the data to arize

response = arize_client.log(
    dataframe=prediction_data,
    schema=schema,
    model_id="salary_model",
    model_version="v1.0",
    environment=Environments.PRODUCTION,
    model_type=ModelTypes.REGRESSION,

)

print("Prediction Data logged successfully!") if response.status_code == 200 else print(f"Failed: {response.text}")

