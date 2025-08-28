import streamlit as st
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

# User input
Age = st.number_input("Age", min_value=0, max_value=100, value=10)
typeOfContact = st.selectbox("TypeofContact", ["Company Invited","Self Inquiry"])
CityTier = st.selectbox("CityTier", ["1","2","3"])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer"])
Gender = st.selectbox("Gender", ["Male","Female"])
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=0, max_value=10, value=1)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=1)
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=0, max_value=10, value=1)
NumberOfTrips = st.number_input("NumberOfTrips", min_value=0, max_value=10, value=1)
Passport = st.selectbox("Passport", ["Yes","No"])
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=0, max_value=10, value=1)
OwnCar = st.selectbox("OwnCar", ["Yes","No"])
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=1)
MaritalStatus = st.selectbox("MaritalStatus", ["Married","Single","Divorced"])
MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, max_value=100000, value=1000)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=0, max_value=100, value=10)
ProductPitched =  st.selectbox('ProductPitched',["Deluxe","Basic"])
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
