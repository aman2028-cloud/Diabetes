import pickle
import streamlit as st
import numpy as np


df=pickle.load(open('data.pkl','rb'))
lr=pickle.load(open('logistic.pkl','rb'))

st.title('Diabetes-Predictor')


gender=st.selectbox('Select Gender',{'Female':0,'Male':1,'Other':3})
age=st.selectbox('Age of the person',df['age'].unique())
hyper=st.selectbox('Hypertension',df['hypertension'].unique())
heart_disease=st.selectbox('Heart-Disease',df['heart_disease'].unique())
bmi=st.selectbox('BMI',df['bmi'].unique())
HbA1c_level=st.selectbox('HbA1c_level',df['HbA1c_level'].unique())
blood_glucose_level=st.selectbox('Glucose Level',df['blood_glucose_level'].unique())




if st.button('Predict'):
    if gender=='Female':
        gender=0
    elif gender=='Male':
        gender=1
    else:
        gender=3
    
    q=[]
    q=np.array([gender,age,hyper,heart_disease,bmi,HbA1c_level,blood_glucose_level])
    q=q.reshape(1,7)
    pred=lr.predict(q)
    
    if pred==0:
        st.title('Not Diabetic')
    else:
        st.title('Diabetic')


