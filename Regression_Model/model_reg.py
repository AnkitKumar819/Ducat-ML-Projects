# reg_model.py
# type: ignore
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

def train_model_from_csv(df, dataset_name):
    st.success("✅ Data received successfully")

    # Drop nulls
    df.dropna(inplace=True)

    # Split features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Define preprocessing
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Define models
    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {}
        },
        "Ridge": {
            "model": Ridge(),
            "params": {"regressor__alpha": [0.1, 1.0, 10.0]}
        },
        "Lasso": {
            "model": Lasso(),
            "params": {"regressor__alpha": [0.01, 0.1, 1.0]}
        },
        "RandomForestRegressor": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "regressor__n_estimators": [50, 100],
                "regressor__max_depth": [None, 10]
            }
        }
    }

    best_model = None
    best_rmse = float("inf")
    best_name = ""
    best_pipeline = None

    for name, model_def in models.items():
        # Full pipeline: preprocess + regressor
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model_def["model"])
        ])

        grid = GridSearchCV(pipeline, model_def["params"], cv=5, scoring="neg_root_mean_squared_error")
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
            best_pipeline = model

    # Report results
    st.success("✅ Model Trained Successfully")
    st.write(f"✅ Best Model: `{best_name}`")
    st.write(f"Test R² Score: {r2:.4f}")
    st.write(f"Test RMSE: {best_rmse:.4f}")
    st.write(f"Test MAE: {mae:.4f}")

    # CV score
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
    st.write(f"Mean Cross-Validation Score: {np.mean(cv_scores):.4f}")

    # Save pipeline
    model_filename = f"{dataset_name}_{best_name}_pipeline.pkl"
    joblib.dump(best_pipeline, model_filename)
    st.success(f"✅ Saved full pipeline: `{model_filename}`")

    # Download link
    with open(model_filename, "rb") as f:
        st.download_button("📥 Download Trained Model", f, file_name=model_filename)
