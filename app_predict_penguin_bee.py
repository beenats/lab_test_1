import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Load the model and data
with open('model_penguin_66130701902.pkl', 'rb') as file:
    obj = pickle.load(file)
    model = obj[0]
    X_train = obj[1]  # Assuming X_train is also saved in the pickle file
    y_train = obj[2]  # Assuming y_train is also saved in the pickle file

# Fit the model (this should be done once when the app starts)
model.fit(X_train, y_train) 

def prepare_data_for_prediction(data):
    """Prepares input data for the model using the same steps as during training."""

    # 1. Get categorical and numerical feature names from the fitted pipeline
    categorical_features = model.named_steps['preprocessor'].transformers_[1][2]  # Get categorical feature names
    numerical_features = model.named_steps['preprocessor'].transformers_[0][2]  # Get numerical feature names

    # 2. Create a ColumnTransformer with the same steps as in the pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)  # handle_unknown='ignore' is important
        ])

    # 3. Fit the preprocessor using the training data (or a representative sample)
    preprocessor.fit(X_train)

    # 4. Transform the input data
    transformed_data = preprocessor.transform(data)
    
    # 5. Get feature names after transformation
    feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

    # 6. Create a DataFrame with the transformed data and correct column names
    transformed_df = pd.DataFrame(transformed_data, columns=feature_names)
    
    return transformed_df

# Streamlit app
st.title("Penguin Species Prediction")

# Input features
# ... (your input widgets) ...

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
    # Prepare the input data
    prepared_x_new = prepare_data_for_prediction(x_new)  
    
    y_pred_new = model.predict(prepared_x_new)
    result = y_pred_new[0] 
    st.success(f"Predicted Specie: {result}")
