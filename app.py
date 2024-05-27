import streamlit as st
import pickle
from streamlit_option_menu import option_menu
import numpy as np

heart_model = pickle.load(open('HEART_DISEASE.sav', 'rb'))##Moo hinh ML su dung o day

with st.sidebar:

    selected = option_menu('Multiple Diseases Prediction System',
                           ['Diabetes Prediction','Heart Disease Prediction'],
                           icons=['activity','heart'],default_index=1)

if (selected == 'Diabetes Prediction'):
    # title
    st.title("Diabetes Prediction use ML")
    st.header("Vui lòng nhập chỉ số sức khoẻ của bạn")

if (selected == 'Heart Disease Prediction'):
    #mapping for 'Sex'
    sex_mapping = {'Female': 0,'Male':1}
    

    #title 
    st.title("Heart Diseases Prediction use ML")
    st.header("Vui lòng nhập chỉ số sức khoẻ của bạn")

    col1,col2,col3=st.columns(3)                 
    with col1:
        male = st.selectbox('Sex',['Male','Female'])
        male = sex_mapping.get(male,0)                                                                
        #male = st.number_input("Sex")
    with col2:
        age = st.number_input("Your Age")
    with col3:
        cigsPerDay = st.number_input("Cigarettes Per Day")
    with col1:
        totChol=st.number_input('Total Cholesterol Level')
    with col2:
        sysBP= st.number_input('Systolic Blood Pressure')
    with col3:
        glucose = st.number_input('Glucose Level')

    #Prediction
    heart_diagnosis= ''

    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_model.predict([[age, male, cigsPerDay, totChol, sysBP, glucose]])
        if(heart_prediction[0] == 1):
            heart_diagnosis = 'The Person is Heart Disease'
        else:
            heart_diagnosis = 'The Person is Not Heart Disease'
    st.success(heart_diagnosis)