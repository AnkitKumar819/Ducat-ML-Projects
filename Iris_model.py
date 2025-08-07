# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import numpy as np

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models and hyperparameters for GridSearchCV
models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=200),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 5, 10]
        }
    }
}

best_model = None
best_score = 0
best_name = ""
final_conf_matrix = None

# Perform GridSearchCV and Cross-Validation
for model_name, mp in models.items():
    print(f"\nTraining {model_name}...")
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='accuracy')
    clf.fit(X_train, y_train)
    
    print(f"Best Parameters for {model_name}: {clf.best_params_}")
    model = clf.best_estimator_
    
    # Cross-validation score (optional info)
    scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation Accuracy for {model_name}: {np.mean(scores):.4f}")
    
    # Evaluation on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Test Accuracy for {model_name}: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Save best model based on test accuracy
    if acc > best_score:
        best_score = acc
        best_model = model
        best_name = model_name
        final_conf_matrix = cm

# Save the best model to 'iris_model.pkl'
joblib.dump(best_model, "iris_model.pkl")
print(f"\n✅ Best model: {best_name} with accuracy: {best_score:.4f}")
print("✅ Saved as 'iris_model.pkl'")
print(f"✅ Confusion Matrix of saved model:\n{final_conf_matrix}")
