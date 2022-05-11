import pandas as pd
from joblib import load
from loan_data.preprocess import preprocess


def make_predictions(trained_data: pd.DataFrame) -> pd.DataFrame:
    # It returns the predictions made by our model
    pred = load(r'../Models/predict.joblib')
    trained_data = preprocess(trained_data, 'test')
    y_test_pred = pred.predict(trained_data)
    y_test_pred[y_test_pred < 0] = abs(y_test_pred[y_test_pred < 0])
    return y_test_pred
