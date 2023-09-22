import streamlit as st 
import joblib 
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import pandas as pd
import json
import sklearn

house_price = joblib.load(r"C:\Users\Anmino\Desktop\Machine-Learning_2\Linear_Regression\usa_house_pricing.pkl")
heart_fail_model = joblib.load(r"C:\Users\Anmino\Desktop\Machine-Learning_2\Logistic_Regression\heart_failure_clinical_records_datase.pkl")

def load_lottyfile(filepath:str):
    with open(filepath,'r') as f:
        return json.load(f)

house_price_pred_img = load_lottyfile('house.json')


def main():
    select = option_menu(menu_title=None,
                     options=['Home','Project','About'],
                     icons=['house','laptop','book'],
                     orientation='horizontal')
    
    if select=='Project':
        model_type1 = ['House Price','Heart Failure']
        model_type = st.sidebar.selectbox("📚Project's",model_type1)
    
    try:
        if model_type=='House Price':
            col1,col2,col3 = st.columns([1,2,1])
            #with col2:
                #st_lottie(house_price_pred_img,width=300,height=200,quality='high')
            col1,col2 = st.columns([1,1])
    
            with col1:
                avg_area_income = st.text_input('Area Income')
            with col2:
                avg_area_house_age = st.text_input('House Age')
            with col1:
                 area_number_of_rooms = st.text_input('Number of Rooms')
            with col2:
                 number_of_bedrooms = st.text_input('Number of Bedrooms')
            with col1:
                area_population = st.text_input('Area Population')
                
            with col1:
                submit_button = st.button('Predict')
            if submit_button:
                pred = house_price.predict([[avg_area_income,
                                       avg_area_house_age,
                                          area_number_of_rooms,
                                          number_of_bedrooms,
                                          area_population]])
            st.success(f"🤖Prediction: {pred[0]}")
        #2nd project        
        elif model_type=="Heart Failure":
            with st.form('form1',clear_on_submit=True):
                col1,col2,col3 = st.columns([1,1,1])
            with col1:
                age = st.text_input('Age')
                    
            with col2:
                an = ['Select',1,0]
                anaemia = st.selectbox('Anaemia',an)
            with col3:
                creatinine_phosphokinase = st.text_input('Creatinine Phosphokinase')
            with col1:
                diabetes_box = ['Select',0,1]
                diabetes = st.selectbox('Diabetes',diabetes_box)
            with col2:
                ejection_fraction = st.text_input("Ejection Fraction")
            with col3:
                high_blood_pressure_box = ["Select",0,1]
                high_blood_pressure = st.selectbox('High Blood Pressure',high_blood_pressure_box)
            with col1:
                platelets = st.text_input("Platelets")
            with col2:
                serum_creatinine = st.text_input("Serum Creatinine")
            with col3:
                serum_sodium = st.text_input("Serum Sodium")
            with col1:
                gender = ["Select",'1',"0"]
                sex = st.selectbox("Gender",gender)
            with col2:
                smoking_box = ["Select",1,0]
                smoking = st.selectbox("Smoking",smoking_box)  
            with col3:
                time = st.slider("Smoking Per/Day",1,300)  
            with col1:
                button = st.form_submit_button("Predict")
            if button:
                model_pred = heart_fail_model.predict([[age,
                                                        anaemia,
                                                        creatinine_phosphokinase,
                                                        diabetes,
                                                        ejection_fraction,
                                                        high_blood_pressure,
                                                        platelets,
                                                        serum_creatinine,
                                                        serum_sodium,
                                                        sex,
                                                        smoking,
                                                        time]])
                st.success(f"🤖Prediction: {model_pred[0]}") 
                    
    except:
        st.warning('❗Please enter the correct value for prediction...')
        
        
    

if __name__=='__main__':
    main()

