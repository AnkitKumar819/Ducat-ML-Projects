#type:ignore
import streamlit as st
import pandas as pd
import joblib
from model_reg import train_model_from_csv

reg_model = joblib.load('insurance_RandomForestRegressor_pipeline.pkl')

st.title("📊 General Regression Predictor")

st.sidebar.button("Home")

st.header("📁 Upload Dataset for Classification")
file = st.file_uploader("Upload your file here....", type=['txt','csv'])

if file is not None:
    df = pd.read_csv(file)
    dataset_name = file.name.replace('.csv', '')
    
    st.dataframe(df)
    st.success("✅Null Values successfully removed")
    st.write(df.shape)
    
    if st.button("Train-Model"):
        train_model_from_csv(df,dataset_name)
        
        
    X = df.iloc[:,:-1]

    st.info(f"you have to write only {X.shape[1]} features here")
    sample = st.text_input("Enter the regression numeric features here")
    
    
    
    if st.button("🧠 Predict"):
        input_values = [x.strip() for x in sample.split(",")]
        input_dict = {col: [val] for col, val in zip(X.columns, input_values)}
        input_df = pd.DataFrame(input_dict)
        
        pred = reg_model.predict(input_df)
        st.success(f"📈 Predicted Value: {pred[0]:.2f}")

    
