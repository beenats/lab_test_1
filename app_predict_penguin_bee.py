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

def prepare_data_for_prediction(data, model):
    """Prepares input data for the model using the same steps as during training."""

    # Get the preprocessor from the pipeline
    preprocessor = model.named_steps['preprocessor']

    # Fit the preprocessor if it's not already fitted
    if not hasattr(preprocessor, 'transformers_'):  # Check if it's fitted
        preprocessor.fit(X_train) # Use representative data to fit

    # Transform the input data
    transformed_data = preprocessor.transform(data)

    # Get feature names after transformation
    categorical_features = preprocessor.transformers_[1][2]
    numerical_features = preprocessor.transformers_[0][2]
    feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

    # Create a DataFrame with the transformed data and correct column names
    transformed_df = pd.DataFrame(transformed_data, columns=feature_names)

    return transformed_df

# ... (rest of your Streamlit app code)

# Make prediction
if st.button("Predict"):
    prepared_x_new = prepare_data_for_prediction(x_new, model)  # Pass model
    y_pred_new = model.predict(prepared_x_new)
    result = y_pred_new[0]
    st.success(f"Predicted Specie: {result}")
