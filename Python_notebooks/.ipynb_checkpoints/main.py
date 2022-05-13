import joblib
import sklearn as sk
import numpy as np
import pandas as pd
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI

app = FastAPI()

##just simple server test
@app.get('/')
def index():
    return {'message': 'System is up and running'}

@app.get('/Welcome')
def get_name(name: str):
    return {'Welcome to moes page': f'{name}'}

if __name__ == '__name__':
    uvicorn.run(app, host='127.0.0.1', port=8000)



##I nitialize model artifact files. this will be loaded at the start of fastapi server
ohe = joblib.load('../../Models/onehot_encoder.joblib')
sc = joblib.load('../../Models/scalar.joblib')
pred = joblib.load('../../Models/predict.joblib')
#Reminder!!: numerical_features = joblib.load('The-Vectors/Models/numerical_features.joblib')
#Reminder!!: categorical_features = joblib.load('The-Vectors/Models/categorical_features.joblib')



##This structure will be used for json validation
#with just that Python decleration, FastApi will perform below operations on the request data
#1) read the body of the request as json
#2) convert the corresponding type if needed
#3) Validate the data, if the data is invalid it should return nice and clear error
##Indicating exactly what and where was the incorrect data
class Data(BaseModel):
    Gender: str
    Credit_History: float
    Education: str
    Married: str
    Self_Employed: str
    Property_Area: str

@app.post('The-Vectors/loan_data/inference.py')
def predict(data: Data):
    #extract data in the correct order
    data_dict = data.dict()
    data_df = pd.DataFrame.from_dict([data_dict])
    #select features required for making predictions
    data_df = data[features]
    #perform one hot encoding
    data_df[categorical_features] = ohe.transform(data_df[categorical_features])
    #create predictions
    prediction = pred.predict(data_df)
    #Map prediction to appropriate label
    prediction_label = ['Not Approved' if label == 'N' else 'Approved' for label in prediction]
    #return response back to client
    return {'prediction': prediction_label}

if __name__ == '__name__':
    uvicorn.run(app, host='127.0.0.1', port=8000, reload=True)

