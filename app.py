import streamlit as st
import pandas as pd
import pickle
import numpy as np


pipe = pickle.load(open('modelllli.pkl', 'rb'))

dff = pickle.load(open('dff.pkl', 'rb'))

st.title("Car Price Predictor Model")
st.write("This app predicts the price of a car you want to sell.Please fill the details below.")

from PIL import Image


company = st.selectbox('Select the Company', dff['company'].unique())
name = st.selectbox('Select the model', dff['name'].unique())

year = st.selectbox('Select year of Purchase', dff['year'].unique())
fuel_type = st.selectbox('Select the fuel type', dff['fuel_type'].unique())
kms_driven = st.text_input('Enter the number of Kms that car has travelled', 'Enter the kilometer driven')
st.sidebar.title('Car Price Predictor Model')


def add_logo(logo_path, width, height):
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo


my_logo = add_logo(logo_path="pic.jpeg", width=500, height=500)
st.sidebar.image(my_logo,"This app predicts the price of a car you want to sell")

if st.button('Predict Price'):
    query = np.array([name, company, year, kms_driven, fuel_type])
    query = query.reshape(1, 5)
    a = str(int(np.round(pipe.predict(query), 2)))
    st.title("The predictor price of a car is " + a)
