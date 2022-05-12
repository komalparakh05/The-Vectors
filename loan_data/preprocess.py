import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load


def preprocess(data: pd.DataFrame, data_context: str) -> pd.DataFrame:
    # encoding and arranging dtartata
    data = data[['Gender','Education',
                 'Married','Self_Employed','Property_Area']]
    if data_context == 'train':
        Onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        Onehot_encoder = Onehot_encoder.fit(data[['Gender',
                                                  'Education',
                                                  'Married'
                                                  'Self_Employed',
                                                  'Property_Area',]])
        dump(Onehot_encoder, r"../Models/onehot_encoder.joblib")

    Onehot_encoder = load(r"../Models/onehot_encoder.joblib")
    encoded_data = pd.DataFrame(Onehot_encoder.transform
                                (data[['Gender',
                                       'Education','Self_Employed',
                                       'Property_Area','Married']]),


                                columns=Onehot_encoder.
                                get_feature_names_out())

    encoded_data = pd.concat([data.reset_index(drop=True),
                              encoded_data.reset_index(drop=True)], axis=1)
    return encoded_data



