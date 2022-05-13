import streamlit
import requests
import json

def run():
    streamlit.title("Loan Classifier")

    Gender = streamlit.selectbox("Gender", ['Male', 'Female'])
    Education = streamlit.selectbox("Education", ['Graduate', 'Not Graduate'])
    Married = streamlit.selectbox("Married", ['Yes', 'No'])
    Self_Employed = streamlit.selectbox("Employment", ['Yes', 'No'])
    Property_Area = streamlit.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])

    data = {
        'Gender': Gender,
        'Education': Education,
        'Married': Married,
        'Self_Employed': Self_Employed,
        'Property_Area': Property_Area,
    }

    if streamlit.button("Predict"):
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        prediction = response.text
        streamlit.success(f"The prediction from model: {prediction}")


if __name__ == '__main__':
    # by default it will run at 8501 port
    run()