
import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Import OneHotEncoder
from sklearn.compose import ColumnTransformer  # Import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# Load the model and encoders
with open('model_penguin_66130701902.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Streamlit app
st.title("Penguin Species Prediction")

# Input form for user to enter penguin data
st.header("Input Penguin Features")
island = st.selectbox("Island", ["Torgersen", "Biscoe", "Dream"])
culmen_length_mm = st.number_input("Culmen Length (mm)", value=37.0)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", value=19.3)
flipper_length_mm = st.number_input("Flipper Length (mm)", value=192.3)
body_mass_g = st.number_input("Body Mass (g)", value=3750)
sex = st.selectbox("Sex", ["MALE", "FEMALE"])

# Prepare input as a DataFrame
x_new = pd.DataFrame({
    "island": [island],
    "culmen_length_mm": [culmen_length_mm],
    "culmen_depth_mm": [culmen_depth_mm],
    "flipper_length_mm": [flipper_length_mm],
    "body_mass_g": [body_mass_g],
    "sex": [sex]
})

# Encode categorical features
x_new['island'] = island_encoder.transform(x_new['island'])
x_new['sex'] = sex_encoder.transform(x_new['sex'])

# Make prediction
try:
    y_pred_new = model.predict(model.predict())
    result = species_encoder.inverse_transform(y_pred_new)
    st.success(f"Predicted Species: {result[0]}")
except Exception as e:
    st.error(f"An error occurred: {e}")


