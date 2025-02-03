# import numpy as np
# import pandas as pd
# from arize.pandas.logger import Client
# from arize.utils.types import ModelTypes, Environments, Schema
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error


  
# #                                                                   Api keys
                
# api_key="d64ab228893fc810549"
# space_id="U3BhY2U6MTUwNTk6Y1JQdw=="


# arize_client= Client(space_id=space_id,api_key=api_key)



# #                                                   Train a Simple Machine Learning Model

# # Generate sample data
# np.random.seed(42)

# X = np.random.rand(100, 1) * 10  # Features
# y = 3.5 * X + np.random.randn(100, 1) * 2  # Target with noise

# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Calculate error
# mae = mean_absolute_error(y_test, y_pred)
# print("Mean Absolute Error:", mae)



# #                                                                    Log Model Data to Arize
                    
# # Convert test data to Pandas DataFrame
# df = pd.DataFrame({
#     "feature": X_test.flatten(),
#     "prediction": y_pred.flatten(),
#     "actual": y_test.flatten()
# })



# #                                                                      Define model schema
# schema = Schema(
#     prediction_id_column_name="feature",
#     prediction_score_column_name="prediction",
#     actual_score_column_name="actual",
#     feature_column_names=["feature"]
# )



# # df["prediction_id"] = df["prediction_id"].astype(str)  # Convert IDs to string
# df["feature"] = df["feature"].astype(float)  # Convert features to float
# df["prediction"] = df["prediction"].astype(float)  # Convert predictions to float
# df["actual"] = df["actual"].astype(float)  # Convert actuals to float




# #                                                                            Log data to Arize

# response = arize_client.log(
#     dataframe=df,
#     model_id="house_price_model",
#     model_version="1.0",
#     model_type=ModelTypes.REGRESSION,
#     environment=Environments.PRODUCTION,
#     schema=schema
# )

# #                                                                Check if the data was logged successfully

# if response.status_code == 200:
#     print("Data logged successfully to Arize!")
# else:
#     print(f"Failed to log data: {response.text}")


