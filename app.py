import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pickle
import streamlit as st 
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np 

st.cache_data.clear()
model = load_model('model.h5')
with open("label_encoder_gender.pkl", 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open("Onehot_encoder_geo.pkl", 'rb') as file:
    Onehot_encoder_geo = pickle.load(file)
with open("scalar.pkl", 'rb') as file:
    scalar = pickle.load(file)

st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', Onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
gender = label_encoder_gender.transform([gender])[0]
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


#st.write(geography,gender,age,balance,credit_score,estimated_salary,tenure,num_of_products,has_cr_card,is_active_member)
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender] ,
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

st.write(input_data)
encoded_geo = Onehot_encoder_geo.transform([[geography]]).toarray()
encoded_geo_df = pd.DataFrame(encoded_geo, columns=Onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data,encoded_geo_df],axis=1)
input_data = scalar.transform(input_data)
prediction = model.predict(input_data)
prediction_pa = prediction[0][0]
st.write(prediction_pa)
if prediction_pa > 0.5:
    st.write("customer is likely to churn")
else :
    st.write("customer is not going to churn")   