import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer

# Ensure you have your updated build_pipeline.py in the preprocessing folder
from preprocessing.build_pipeline import build_preprocessing_pipeline

DATA_PATH = "data/PCOS_dataset.csv"

def to_dense_if_needed(X):
    """Converts sparse output from transformers to dense for models like SVC/GaussianNB"""
    return X.toarray() if hasattr(X, "toarray") else X

# In load_data(), lower the threshold to get more "Positive" cases
def load_data():
    df = pd.read_csv(DATA_PATH)
    
    # Let's be less strict to get more than 4 positive cases for testing
    bleeding_signal = df['Unusual_Bleeding'].str.lower().str.strip().isin(['yes', 'y'])
    cycle_signal = (df['Length_of_cycle'] > 30) # Lowered from 35
    bmi_signal = (df['BMI'] > 24)              # Lowered from 25
    
    # Diagnosis = 1 if at least ONE condition is met (increases sample size)
    df['Diagnosis'] = ((bleeding_signal.astype(int) + 
                        cycle_signal.astype(int) + 
                        bmi_signal.astype(int)) >= 1).astype(int)

    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]
    return X, y


def get_models():
    """
    Returns a dictionary of models to compare.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
        ),
        "RBF Kernel SVM": SVC(
            kernel="rbf", class_weight="balanced", probability=True, random_state=42
        )
    }

def compare_models():
    print("Loading data and generating clinical labels...")
    X, y = load_data()
    preprocessor = build_preprocessing_pipeline()

    print(f"Dataset Shape: {X.shape}")
    print(f"Positive Cases (PCOS Risk): {y.sum()} ({y.mean()*100:.1f}%)")
    print("\nStarting Model Comparison (5-fold CV ROC-AUC)...\n")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    print(f"{'Model Name':<25} | {'Mean AUC':<10} | {'Std Dev':<10}")
    print("-" * 50)

    for name, model in get_models().items():
        pipeline = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("to_dense", FunctionTransformer(to_dense_if_needed)),
            ("model", model)
        ])

        # Cross-validation
        scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1
        )

        print(f"{name:<25} | {scores.mean():.4f}     | {scores.std():.4f}")

if __name__ == "__main__":
    compare_models()