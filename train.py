import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

# Create model directory if not exists
if not os.path.exists('model'):
    os.makedirs('model')

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Drop student_id as it's not a feature
    if 'student_id' in df.columns:
        df = df.drop(columns=['student_id'])
    
    # Separate features and target
    # Target is 'passed' based on assignment/implementation plan
    target = 'passed'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")
    
    X = df.drop(columns=[target, 'final_score','performance_category']) # performance_category is derived/leaky usually, or alternate target
    y = df[target]
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    # Scale numerical features (important for KNN and LR)
    # We'll stick to basic scaling for simplicity and consistency across models, 
    # though tree models don't strictly need it.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save preprocessors for the app
    joblib.dump(encoders, 'model/encoders.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(X.columns.tolist(), 'model/feature_names.pkl')

    print(f"Columns in the training data {X_scaled.columns}...")
    
    return X_scaled, y

def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'kNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Save model
        filename = f"model/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, filename)
        
    return trained_models

def evaluate_models(models, X_test, y_test):
    results = []
    confusion_matrices = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(4,3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred 0", "Pred 1",],
            yticklabels=["Actual 0", "Actual 1"]
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        os.makedirs("model/cm_images", exist_ok=True)
        
        filename = f"model/cm_images/{name.replace(' ', '_').lower()}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        
        results.append({
            'ML Model Name': name,
            'Accuracy': acc,
            'AUC': auc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'MCC Score': mcc
        })

     # Save all confusion matrices to one JSON file
    with open('model/confusion_matrix.json', "w") as f:
        json.dump(confusion_matrices, f, indent=4)
        
    return pd.DataFrame(results)

def main():
    data_path = 'ai_impact_student_performance_dataset.csv'
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # 1. Load and Preprocess
    X, y = load_and_preprocess_data(data_path)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Models
    models = train_models(X_train, y_train)
    
    # 4. Evaluate Models
    metrics_df = evaluate_models(models, X_test, y_test)
    
    # Save Metrics
    metrics_df.to_csv('model/metrics.csv', index=False)
    
    print("\nModel Evaluation Metrics:")
    print(metrics_df)
    print("\nTraining completed. Artifacts saved in 'model/' directory.")

if __name__ == "__main__":
    main()
