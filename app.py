import pandas as pd 
import numpy as np 
import pickle as pk
import streamlit as st
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#from flask import Flask, request, jsonify

model = pk.load(open('model.pkl', 'rb'))

html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Car Price Prediction</h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)
st.title('Welcome to Getyourcar')
st.header(' CAR PRICE PREDICTION MODEL ')


cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    return car_name.split()[0]

cars_data['name'] = cars_data['name'].apply(get_brand_name)

name=st.selectbox('Select Car Brand', cars_data['name'].unique())
year=st.slider('Year of Manufacture', min_value=1994, max_value=2024, value=1994, step=1)
km_driven=st.slider('Kilometers Driven', min_value=0, max_value=200000, value=0, step=500)
fuel=st.selectbox('Fuel Type', cars_data['fuel'].unique())
seller_type=st.selectbox('Select the Seller', cars_data['seller_type'].unique())
transmission=st.selectbox('Transmission Type', cars_data['transmission'].unique())
owner=st.selectbox('Select Owner', cars_data['owner'].unique())
mileage=st.slider('Mileage', min_value=0, max_value=30, value=0, step=1)
engine=st.slider('Engine', min_value=0, max_value=5000, value=0, step=50)
max_power=st.slider('Select Maximum Power', min_value=0, max_value=500, value=0, step=5)
seats=st.slider('Number of Seats', min_value=5, max_value=10, value=5, step=1)



if  st.button("Predict"):
    input_data_model = pd.DataFrame(
    [[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],
    columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])
   
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'],
                           [1,2,3,4,5], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4], inplace=True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3], inplace=True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'],[1,2], inplace=True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                          [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
                          ,inplace=True)

    car_price = model.predict(input_data_model)
    if car_price[0] < 0:
        st.markdown('No such model available')
    else:
        formatted_price = f"{car_price[0]:.2f}"
        st.success(f'Car Price will be {formatted_price}')