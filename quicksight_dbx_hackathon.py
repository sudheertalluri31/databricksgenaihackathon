# Databricks notebook source
df = spark.sql("SELECT * FROM customer_loans")

# Convert the Spark DataFrame to a Pandas DataFrame if necessary
pdf = df.toPandas()

# COMMAND ----------

import pandas as pd

# COMMAND ----------

pdf.head(1)

# COMMAND ----------

pdf.fillna(method='ffill', inplace=True)

pdf_y = pdf['loan_status']

X_predict = pdf[pdf["loan_status"]=="current"]
X_known = pdf[pdf["loan_status"]!="current"]
y_known = pdf[pdf["loan_status"]!="current"]['loan_status']
pdf = pdf[pdf["loan_status"]!="current"]

X_predict = X_predict.drop(['ssn', 'loan_id', 'cust_id', 'issued_on', 'first_name', 'last_name', 'loan_status'], axis=1)
X_known = X_known.drop(['ssn', 'loan_id', 'cust_id', 'issued_on', 'first_name', 'last_name', 'loan_status'], axis=1)

pdf.head()

# COMMAND ----------


# Encode categorical variables

X_known = pd.get_dummies(X_known, drop_first=True)
X_predict = pd.get_dummies(X_predict, drop_first=True)



# COMMAND ----------

import pickle
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import os

# Load the pickle file
pickle_file_path = '/Workspace/Users/sudheer.talluri.usa@gmail.com/loan_default_model.pkl'
with open(pickle_file_path, 'rb') as file:
    model = pickle.load(file)

signature = infer_signature(X_known.head(1), model.predict(X_known.head(1)))

# Define a wrapper class for the model
class PickleModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)

# Log the model with MLflow
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="pickle_model",
        python_model=PickleModelWrapper(),
        signature=signature,
        input_example=X_known.head(1)
    )
    # Optionally, save run_id for future reference
    run_id = run.info.run_id

print(f"Model logged in run: {run_id}")


# COMMAND ----------

# Registe
model_uri = f"runs:/{run_id}/pickle_model"
registered_model_name = "PickleModel"
mlflow.register_model(model_uri, registered_model_name)


# COMMAND ----------

# Register the model
model_uri = f"runs:/{run_id}/pickle_model"
registered_model_name = "PickleModel"
mlflow.register_model(model_uri, registered_model_name)


# COMMAND ----------


