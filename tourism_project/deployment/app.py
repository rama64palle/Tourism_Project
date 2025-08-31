]import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="rama64palle/Tourism_Project_Model", filename="tourism_project_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism project prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
""")

# Mapping of encoded value to display name
ENCODING_MAP_TypeofContact =  {1: "Company Invited", 2:"Self Inquiry" }
ENCODING_MAP_CityTier =  {1: 'Tier 1',2: 'Tier 2', 3:'Tier 3'}
ENCODING_MAP_Occupation =  {1: 'Salaried', 2:'Freelancer'}
ENCODING_MAP_Gender =  {1: 'Male', 2:'Female'}
ENCODING_MAP_MaritalStatus =  {1: 'Married', 2:'Single', 3:'Divorced'}
ENCODING_MAP_Designation =  {1: 'Executive', 2:'Manager', 3:'Senior Manager', 4:'AVP', 5:'VP'}
ENCODING_MAP_ProductPitched =  {1: 'Deluxe', 2:'Basic'}
ENCODING_MAP_Passport =  {1: 'Yes', 2:'No'}
ENCODING_MAP_OwnCar =  {1: 'Yes', 2:'No'}
ENCODING_MAP_ProdTaken =  {1: 'Yes', 2:'No'}

def format_label_TypeofContact(code):
    return ENCODING_MAP_TypeofContact[code]

def format_label_CityTier(code):
    return ENCODING_MAP_CityTier[code]

def format_label_Occupation(code):
    return ENCODING_MAP_Occupation[code]

def format_label_Gender(code):
    return ENCODING_MAP_Gender[code]

def format_label_MaritalStatus(code):
    return ENCODING_MAP_MaritalStatus[code]

def format_label_Designation(code):
    return ENCODING_MAP_Designation[code]

def format_label_ProductPitched(code):
    return ENCODING_MAP_ProductPitched

def format_label_Passport(code):
    return ENCODING_MAP_Passport[code]

def format_label_OwnCar(code):
    return ENCODING_MAP_OwnCar[code]

def format_label_ProdTaken(code):
    return ENCODING_MAP_ProdTaken[code]




# User input
Age = st.number_input("Age", min_value=0, max_value=100, value=10)
typeOfContact = st.selectbox("TypeofContact", options=list(ENCODING_MAP_TypeofContact.keys()),format_func=format_label_TypeofContact)
CityTier = st.selectbox("CityTier",  options=list(ENCODING_MAP_CityTier.keys()),format_func=format_label_CityTier)
Occupation = st.selectbox("Occupation", options=list(ENCODING_MAP_Occupation.keys()),format_func=format_label_Occupation)
Gender = st.selectbox("Gender", options=list(ENCODING_MAP_Gender.keys()),format_func=format_label_Gender)
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=0, max_value=10, value=1)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=1)
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=0, max_value=10, value=1)
NumberOfTrips = st.number_input("NumberOfTrips", min_value=0, max_value=10, value=1)
Passport = st.selectbox("Passport", options=list(ENCODING_MAP_Passport.keys()),format_func=format_label_Passport)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=0, max_value=10, value=1)
OwnCar = st.selectbox("OwnCar", options=list(ENCODING_MAP_OwnCar.keys()),format_func=format_label_OwnCar)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=1)
MaritalStatus = st.selectbox("MaritalStatus", options=list(ENCODING_MAP_MaritalStatus.keys()),format_func=format_label_MaritalStatus)
MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, max_value=100000, value=1000)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=0, max_value=100, value=10)
ProductPitched =  st.selectbox('ProductPitched', options=list(ENCODING_MAP_ProductPitched.keys()),format_func=format_label_ProductPitched)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])


# Assemble input into DataFrame
input_data = pd.DataFrame([{'Age': Age,
                            'typeOfContact': typeOfContact,
                            'CityTier': CityTier,
                            'Occupation': Occupation,
                            'Gender': Gender,
                            'NumberOfPersonVisiting': NumberOfPersonVisiting,
                            'NumberOfFollowups': NumberOfFollowups,
                            'PreferredPropertyStar': PreferredPropertyStar,
                            'NumberOfTrips': NumberOfTrips,
                            'Passport': Passport,
                            'PitchSatisfactionScore': PitchSatisfactionScore,
                            'OwnCar': OwnCar,
                            'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
                            'MaritalStatus': MaritalStatus,
                            'MonthlyIncome': MonthlyIncome,
                            'DurationOfPitch': DurationOfPitch,
                            'ProductPitched': ProductPitched,
                            'Designation': Designation,


                            }])


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Purchased a package" if prediction == 1 else "Not Purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
