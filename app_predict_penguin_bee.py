
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
st.title("Penguin Species Predictor")

st.write("""
Provide the penguin features below to predict its species:
""")

# User input for penguin features
island = st.selectbox("Island", ['Torgersen', 'Biscoe', 'Dream'])
culmen_length_mm = st.number_input("Culmen Length (mm)", min_value=0.0, step=0.1)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", min_value=0.0, step=0.1)
flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=0.0, step=0.1)
body_mass_g = st.number_input("Body Mass (g)", min_value=0.0, step=1.0)
sex = st.selectbox("Sex", ['MALE', 'FEMALE'])

# Prepare input DataFrame
x_new = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

# Predict and display the result
if st.button("Predict Species"):
    # Transform categorical columns using the encoders
    x_new['island'] = island_encoder.transform(x_new['island'])
    x_new['sex'] = sex_encoder.transform(x_new['sex'])
    
    # Make prediction
    y_pred_new = model.predict(x_new)
    predicted_species = species_encoder.inverse_transform(y_pred_new)
    
    st.success(f"Predicted Species: {predicted_species[0]}")


