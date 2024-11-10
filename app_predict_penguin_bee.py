import streamlit as st
import pickle
import pandas as pd

# Load the model and the necessary transformers (pickled)
with open('model_penguin_66130701902.pkl', 'rb') as file:
    obj = pickle.load(file)
    model = obj[0]  # Assuming model is the first element in the pickled object
    column_transformer = obj[1]  # Assuming the transformer is the second element

# Streamlit app
st.title("Penguin Species Prediction")

# Introduction
st.write("""
This app allows you to predict the species of a penguin based on its physical characteristics.
Please enter the details below:
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

# Transform the new input data using the column transformer
try:
    x_new_transformed = column_transformer.transform(x_new)
except Exception as e:
    st.error(f"Error during transformation: {str(e)}")
    st.stop()

# Prediction button
if st.button("Predict Species"):
    try:
        # Make prediction using the model
        y_pred_new = model.predict(x_new_transformed)
        
        # Assuming that the species encoder is also loaded
        # If you use label encoding, decode the prediction
        predicted_species = y_pred_new[0]  # In case it is a single class prediction
        
        # Display the result
        st.success(f"The predicted species is: {predicted_species}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
