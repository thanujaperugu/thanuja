import streamlit as st
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

dtr = joblib.load('decision_tree_model.pkl')

def preprocess_input(input_data, preprocessing_pipeline):
    preprocessed_data = preprocessing_pipeline.transform(input_data)
    return preprocessed_data

preprocessing_pipeline = joblib.load('preprocessing_pipeline.pkl')

st.set_page_config(layout="wide")
with st.sidebar:
    st.title("Medication Recommendation System")  
    st.markdown(
        """
        ## About the Project
        
        This is a Medication Recommendation System designed to provide personalized medication recommendations 
        based on user inputs such as age, gender, blood type, disease, and test results. Simply fill in the 
        required information and click the button to get your medication recommendation.
        """
    )
with st.container():
    st.subheader("User Inputs")
    with st.form(key='my_form'):
        age = st.slider("Age", min_value=10, max_value=84, value=10)
        diseases = ["Acne", "Osteoarthritis", "Bronchial Asthma", "Alcoholic hepatitis", "Impetigo",
                                           "Tonsillitis", "(Vertigo) Paroxysmal Positional Vertigo",
                                           "Dimorphic hemorrhoids (piles)", "Tuberculosis", "Pneumonia", "Varicose veins",
                                           "Hypothyroidism", "Heart attack", "Hypoglycemia", "Cervical spondylosis",
                                           "Diabetes", "Common Cold", "Arthritis", "Hypertension", "Chronic cholestasis",
                                           "Migraine", "Urinary tract infection", "Hyperthyroidism",
                                           "GERD (Gastroesophageal Reflux Disease)", "Allergy", "Chickenpox", "Dengue",
                                           "Psoriasis", "Malaria", "Fungal infection", "Jaundice", "Hepatitis A",
                                           "Paralysis (brain hemorrhage)", "Peptic ulcer disease",
                                           "Vertigo (Paroxysmal Positional Vertigo)", "Hepatitis B", "Gastroenteritis",
                                           "Typhoid", "AIDS", "Hepatitis E", "Drug Reaction", "Hepatitis C", "Hepatitis D"]
        disease = st.selectbox("Disease", [""] + diseases)
        test_result = st.selectbox("Test Result", [""] + ["Normal", "Abnormal", "Inconclusive"])
        submit_button = st.form_submit_button(label='Get Recommendation', help='Click to get medication recommendation')

        if submit_button and (disease == "" or test_result == ""):
            st.warning("Please select a disease and test result before submitting.")

    if submit_button:
        input_data = pd.DataFrame({
            'Age': [age],
            'Disease': [disease],
            'Test Result': [test_result]
        })


        preprocessed_input = preprocess_input(input_data, preprocessing_pipeline)
        medication = dtr.predict(preprocessed_input)
        st.subheader("Recommended Medication")
        st.write(medication[0])