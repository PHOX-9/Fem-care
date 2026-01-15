import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from preprocessing.build_pipeline import build_preprocessing_pipeline
from compare_models import load_data, to_dense_if_needed

MODEL_OUTPUT_PATH = "models/pcos_risk_model.pkl"


# ---------------------------------------------------------
# Training function for the final production model
# ---------------------------------------------------------
def train_final_model():
    
    X, y = load_data()
    
    # Initialize the preprocessor
    preprocessor = build_preprocessing_pipeline()

    # Define the final model (Random Forest was the top performer)
    final_model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=10, 
        class_weight="balanced", 
        random_state=42
    )

    # full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("to_dense", FunctionTransformer(to_dense_if_needed)),
        ("model", final_model)
    ])

    print("Training final model on all 162 records...")
    pipeline.fit(X, y)

    print(f"Saving model to {MODEL_OUTPUT_PATH}...")
    joblib.dump(pipeline, MODEL_OUTPUT_PATH) 
    print("Success!")

if __name__ == "__main__":
    train_final_model()