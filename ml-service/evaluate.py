import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from compare_models import load_data

MODEL_PATH = "models/pcos_risk_model.pkl"

def evaluate_model():

    """
    Evaluates the trained PCOS risk prediction model on unseen data.
    Prints probability distribution, ROC-AUC, and classification metrics.
    """
    
    # 1. Load dataset using the same logic as training
    X, y = load_data()
    
    # 2. Split data to test on 'unseen' records
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. Load the saved pipeline
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print("Error: Model file not found. Please run train.py first.")
        return

    # 4. Generate Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # 5. Metrics Calculation
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\n[PROBABILITY DISTRIBUTION]")
    print(f"Min Risk:  {np.min(y_prob)*100:.2f}%")
    print(f"Max Risk:  {np.max(y_prob)*100:.2f}%")
    print(f"Mean Risk: {np.mean(y_prob)*100:.2f}%")

    print(f"\n[CORE METRICS]")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    print("\n[CLASSIFICATION REPORT]")
    
    # This shows Precision, Recall, and F1-Score
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    print("\n[CONFUSION MATRIX]")
    # Shows True Positives vs False Positives
    print(confusion_matrix(y_test, y_pred))

    # 6. Interpretation
    print("\n[INTERPRETATION]")
    if roc_auc > 0.90:
        print("Excellent: The model has successfully captured the clinical logic.")
    elif roc_auc > 0.70:
        print("Good: The model shows strong predictive power.")
    else:
        print("Warning: Model performance is low. Consider more feature engineering.")

if __name__ == "__main__":
    evaluate_model()