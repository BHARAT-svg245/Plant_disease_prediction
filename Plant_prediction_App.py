import streamlit as st
import numpy as np
import joblib

model=joblib.load("plant_disease_model.joblib",)
#Title the app
st.title("Plant Disease prediction App")
st.write("0-No,1-Yes")

#Input from the user
temp=st.number_input("Enter my temperature:",min_value=4.0,max_value=57.0,value=27.48)
humidity=st.number_input("enter humidity:",min_value=6.00,max_value=103.0,value=34.21)
rainfall=st.number_input("Enter rainfall:",min_value=0.0,max_value=85.0,value=0.57255)
soil_ph=st.number_input("Enter ph:",min_value=4.00,max_value=9.00,value=5.21)

#make the slider example
#age=st.slider("selecct the age:",1,100,25)

#Button action
if st.button("Predict Disease"):
    feature=np.array([[temp,humidity,rainfall,soil_ph]])
    prediction=model.predict(feature)
    st.success(f"{prediction[0]}")
