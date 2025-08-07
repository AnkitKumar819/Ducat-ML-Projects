#type:ignore

from regex import F
import streamlit as st
import pandas as pd
import joblib

spam_model = joblib.load('spam_ham_model.pkl')
class_model = joblib.load('best_iris_model.pkl')

st.title("LENS Xpert(ML/NLP)")

Home, Spam, Model, Sentiments = st.tabs(["Home", "Spam Classification", "Model Recommendation", "Food Review Sentiments"])

with Home:
    st.header("this is Home")
    
with Spam:
    st.header("This is Spam Classification Page")
    message = st.text_input("Enter text here....")
    if st.button("Predict"):
        pred = spam_model.predict([message])
        if pred==1:
            st.write("spam")
            #st.image("")
        else:
            st.write("ham")
            #st.image()
            
    path = st.file_uploader("Upload your file here....", type=['txt','csv'])
    if path:
        st.write("file before prediction")
        df = pd.read_csv(path, header=None, names=['msg'], encoding='latin-1')
        df.index= range(1, df.shape[0]+1)
        st.dataframe(df)

        st.write("file after prediction")
        df = df.dropna()
        df.index= range(1, df.shape[0]+1)
        pred = spam_model.predict(df.msg)
        df['Prediction'] = pred
        st.dataframe(df)
        
    
with Model:
    # Load pre-trained model
    @st.cache_resource
    def load_model():
        return joblib.load("best_iris_model.pkl")

    class_model = load_model()

    # Sidebar
    st.sidebar.title("🔎 ML Model Analyzer")
    select = st.sidebar.selectbox("Choose your Task", ['Regression', 'Classification', "Clustering", "NLP"])

    # Main Area
    st.title("📊 Streamlit ML Model Inference App")

    # Only do classification if selected
    if select == 'Classification':
        st.header("📁 Upload Dataset for Classification")
        file = st.file_uploader("Upload your CSV file...", type=['csv'])
        st.success("✅ file uploaded successfully")

        if file is not None:
            # Read and clean data
            df = pd.read_csv(file)
            st.write("📄 Original Data", df.head())

            df.dropna(inplace=True)
            st.success("✅ Null values removed")
            
            feature = st.text_input("Enter the features to be predicted....")

            # Split features and target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Predict
            if st.button("🧠 Predict"):
                predictions = class_model.predict(X)
                st.subheader("✅ Prediction Results")
                st.write(predictions)

                # Combine with original data for download
                result_df = df.copy()
                result_df["Prediction"] = predictions

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", data=csv, file_name="classification_results.csv", mime="text/csv")








    with Sentiments:
        st.header("This is Sentiment Page")
        
    st.sidebar.button("Home")
    st.sidebar.button("Spam Classification")
    st.sidebar.button("Model")

    with st.sidebar.expander("About US"):
        st.write("Contact Phone No: 999999999")
        st.write("email : Abc@gmail.com")