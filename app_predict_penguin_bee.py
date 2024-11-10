import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Load the model and transformers from the pickle file
with open('model_penguin_66130701902.pkl', 'rb') as file:
    obj = pickle.load(file)
    model = obj[0]  # Assuming the first object is the model
    column_transformer = obj[1]  # Assuming the second object is the column transformer

# Ensure the transformer is fitted (re-fit it on your training data if necessary)
# For this example, we'll assume you need to load the training data to fit the transformer

# Assuming original training data is available:
# You can either load it from a CSV file or re-define the DataFrame for the purposes of this demo
# Example:
# X_train = pd.read_csv('your_training_data.csv')
# y_train = X_train.pop('species')  # Target column (species)

# Fit the ColumnTransformer again if necessary (make sure you have the training data)
# This is typically necessary when you don't save the full pipeline.
if not hasattr(column_transformer, 'transformers_'):
    column_transformer.fit(X_train)  # Re-fit the transformer with training data

# Now, proceed with the user input as before
st.title("Penguin Species Prediction")

# User input for penguin features
island = st.selectbox("Island", ['Torgersen', 'Biscoe', 'Dream'])
culmen_length_mm = st.number_input("Culmen Length (mm)", min_value=0.0, step=0.1)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", min_value=0.0, step=0.1)
flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=0.0, step=0.1)
body_mass_g = st.number_input("Body Mass (g)", min_value=0.0, step=1.0)
sex = st.selectbox("Sex", ['MALE', 'FEMALE'])

# Prepare the input data as a DataFrame (Ensure it's 2D, with one row and multiple columns)
x_new = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

# Transform the input data using the fitted column transformer
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
        
        # Assuming the species is encoded, you might need to decode it
        predicted_species = y_pred_new[0]  # In case it's a single prediction
        st.success(f"The predicted species is: {predicted_species}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
