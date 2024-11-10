import streamlit as st
import pickle
import pandas as pd

# Load the model and column transformer
with open('model_penguin_66130701902.pkl', 'rb') as file:
    obj = pickle.load(file)
    model = obj[0]  # Assuming the first object is the model
    column_transformer = obj[1]  # Assuming the second object is the transformer

# Check if the column transformer is fitted (if not, we need to fit it)
if not hasattr(column_transformer, 'transformers_'):
    # Ensure to fit it with your training data (you need to have X_train data)
    column_transformer.fit(X_train)  # Assuming you have access to the training data

# Now proceed to prediction
st.title("Penguin Species Prediction")

# Input form for Streamlit
island = st.selectbox("Island", ['Torgersen', 'Biscoe', 'Dream'])
culmen_length_mm = st.number_input("Culmen Length (mm)", min_value=0.0, step=0.1)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", min_value=0.0, step=0.1)
flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=0.0, step=0.1)
body_mass_g = st.number_input("Body Mass (g)", min_value=0.0, step=1.0)
sex = st.selectbox("Sex", ['MALE', 'FEMALE'])

# Prepare the input data as a DataFrame
x_new = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

# Transform the input data
x_new_transformed = column_transformer.transform(x_new)

# Predict the species
if st.button("Predict Species"):
    y_pred_new = model.predict(x_new_transformed)
    predicted_species = y_pred_new[0]  # In case it's a single prediction
    st.success(f"The predicted species is: {predicted_species}")
