import pandas as pd
import joblib

MODEL_PATH = "models/pcos_risk_model.pkl"

def predict_risk(user_input_dict):
    """
    Takes a dictionary of user inputs and returns the PCOS risk percentage.
    """
    model = joblib.load(MODEL_PATH)
    
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([user_input_dict])
    
    # Predict probability of PCOS risk (class = 1)
    risk_prob = model.predict_proba(df)[0][1]
    
    # Return clean Python float (API/JSON friendly)
    return float(round(risk_prob * 100, 2))


