#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from joblib import dump
from sklearn import linear_model
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split


# In[ ]:


def train_model(loan_train, max_depth=2):
    # Split data
    X = loan_train[['Gender','Credit_History','Education','Married','Self_Employed','Property_Area']]
    y = loan_train[["Loan_Status"]]
    x_train = x_train.drop(['Loan_Status'], axis=1)
    x_test = x_test.drop(['Loan_Status'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit model
    model = RandomForestRegressor(max_depth=max_depth)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Test mse = {mse}, Test RMSE = {rmse}, Random forest max depth = {max_depth}")
    return model, mse, rmse

