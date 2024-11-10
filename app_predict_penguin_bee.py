import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load the model
with open('model_penguin_66130701902.pkl', 'rb') as file:
    obj = pickle.load(file)
    model = obj[0]

# Streamlit app
st.title("Penguin Species Prediction")

# Input features
island = st.selectbox("Island", ['Torgersen', 'Biscoe', 'Dream'])
culmen_length_mm = st.number_input("Culmen Length (mm)", value=37.0)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", value=19.3)
flipper_length_mm = st.number_input("Flipper Length (mm)", value=192.3)
body_mass_g = st.number_input("Body Mass (g)", value=3750)
sex = st.selectbox("Sex", ['MALE', 'FEMALE'])

# Create input DataFrame
x_new = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

# Make prediction
if st.button("Predict"):
    y_pred_new = model.predict(x_new)
    result = y_pred_new[0]  # Get the predicted species

    st.success(f"Predicted Specie: {result}")
