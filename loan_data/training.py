import numpy as np
import pandas as pd
from joblib import dump
from sklearn import linear_model
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from inference import make_predictions
from preprocess import preprocess


def build_model(train_data: pd.DataFrame) -> int:

    # Returns a dictionary with the model performances

    x_train, x_test = train_test_split(train_data,
                                       train_size=.5,
                                       stratify=train_data[['Loan_Status']])

    y_train = x_train[['Loan_Status']]
    y_test = x_test[['Loan_Status']]
    x_train = x_train.drop(['Loan_Status'], axis=1)
    x_test = x_test.drop(['Loan_Status'], axis=1)

    x_train = preprocess(x_train, 'train')
    clf = linear_model.Lasso(alpha=-1)
    predicte = clf.fit(x_train, y_train)
    dump(predicte, r'../Models/predict.joblib')
    ytp = clf.predict(x_train)
    ytp[ytp < 0] = abs(ytp[ytp < 0])
    print("Evaluation for train")
    print("Lasso Score:", clf.score(x_train, y_train[['Loan_Status']]))
    print("RMSLE Error:", compute_rmsle(y_train[['Loan_Status']], ytp))

    y_ptest = make_predictions(x_test)
    x_test = preprocess(x_test, 'test')
    print("Evaluation on local test")
    print("Lasso Score:", clf.score(x_test, y_ptest))
    print("RMSLE Error:", compute_rmsle(y_test[['Loan_Status']], y_ptest))

    return 1


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray,
                  precision: int = 2) -> float:
    #compute the score of our code
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)
