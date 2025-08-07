# reg_model.py
#type:ignore
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def train_model_from_csv(df, dataset_name):
    st.success("✅ Data received successfully")

    # Drop null values
    df.dropna(inplace=True)
    object_cols = [col for col in df.columns if df[col].dtype == 'object']
    encoder = LabelEncoder()
    for col in object_cols:
        df[col] = encoder.fit_transform(df[col])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {}
        },
        "Ridge": {
            "model": Ridge(),
            "params": {"alpha": [0.1, 1.0, 10.0]}
        },
        "Lasso": {
            "model": Lasso(),
            "params": {"alpha": [0.01, 0.1, 1.0]}
        },
        "RandomForestRegressor": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10]
            }
        }
    }

    best_model = None
    best_rmse = float("inf")
    best_name = ""

    for name, mp in models.items():
        grid = GridSearchCV(mp["model"], mp["params"], cv=5, scoring="neg_root_mean_squared_error")
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if rmse < best_rmse:
            best_model = model
            best_rmse = rmse
            best_name = name

    st.success("✅ Model Trained Successfully")
    st.write(f"Test R² Score: {r2:.4f}")

    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
    st.write(f"Mean Cross-Validation Score: {np.mean(cv_scores):.4f}")

    # Save model with dataset name
    model_filename = f"{dataset_name}_{best_name}_model.pkl"
    joblib.dump(best_model, model_filename)
    st.success(f"✅ Saved best model: {model_filename}")

    # Optional: Download button
    with open(model_filename, "rb") as f:
        st.download_button("📥 Download Trained Model", f, file_name=model_filename)