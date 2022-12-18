import pickle

import numpy as np
import streamlit as st

loaded_model = pickle.load(open('gradientBoost.sav', 'rb'))


def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():
    # giving a title
    st.title('Diabetes in Woman Prediction Web Application')

    # getting the input data from the user

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # code for Prediction
    diagnosisResult = ''

    # creating a button for Prediction

    if st.button('Diabetes Prediction Test Result'):
        diagnosisResult = diabetes_prediction(
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosisResult)


if __name__ == '__main__':
    main()
