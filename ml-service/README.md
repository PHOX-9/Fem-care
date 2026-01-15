# PCOS Risk Prediction – ML Service

This repository contains a **complete machine learning pipeline** that estimates **PCOS risk** based on **simple, symptom-based inputs**.  
The system outputs a **risk percentage**, not a medical diagnosis.

The focus of this implementation is to keep inputs **easy for normal users**, while handling all medical logic **internally in code**.

---

# What is implemented in this PR

- A **Random Forest–based machine learning model** for PCOS risk prediction  
- A **single preprocessing pipeline** shared across training, evaluation, and prediction  
- Clean separation between:
  - preprocessing
  - training
  - evaluation
  - backend inference
- A saved model artifact (`pcos_risk_model.pkl`) that includes both preprocessing and model logic  
- The model returns a **risk score (percentage)** instead of a binary Yes/No output  

---

## 1. Project structure and file responsibilities

ml-service/
│
├── data/
│ └── pcos_prediction_dataset.csv
├── preprocessing/
│ └── build_pipeline.py
├── models/
│ └── pcos_risk_model.pkl
├── train_final_model.py
├── evaluate.py
├── predict.py
├── compare_models.py
└── README.md


## 2.File Responsibilities

| File | Purpose |
|----|----|
| `build_pipeline.py` | Cleans data, handles missing values, encodes categories, scales features |
| `train_model.py` | Trains the final Random Forest model and saves it |
| `evaluate.py` | Evaluates model performance using ROC-AUC and classification metrics |
| `predict.py` | Used by backend to generate PCOS risk percentage |
| `compare_models.py` | Used only for experimentation and model comparison |
| `pcos_risk_model.pkl` | Saved preprocessing + model pipeline |

---

## 3.How User Input Is Taken (Simple Language)

Users are **not asked medical or technical questions**.  
They answer **simple questions**, and medical features are derived internally.

---

### How user input is taken (simple language)

Users are **not asked medical questions**.  
They only answer **simple, observable questions**, and medical features are derived internally.

| Internal Column | What the user is asked |
|-----------------|------------------------|
| `Age` | How old are you (in years)? |
| `Length_of_cycle` | How many days are there between your periods? |
| `Estimated_day_of_ovulution` | Around which day of your cycle do you think ovulation happens? (optional) |
| `Length_of_Leutal_Phase` | Calculated internally (cycle length − ovulation day) |
| `Length_of_menses` | How many days does your period usually last? |
| `Unusual_Bleeding` | Do you experience heavy bleeding more than once in a cycle? |
| `Height` | What is your height? (example: 5'6) |
| `Weight` | What is your weight in kilograms? |
| `BMI` | Calculated internally from height and weight |
| `number_of_peak` | How many times heavy bleeding occurs in one cycle |
| `Mean_of_length_of_cycle` | Average cycle length over recent months |
| `Menses_score` | Overall period discomfort score (1–5 scale) |

Users never enter terms like **BMI** or **luteal phase** directly.

---

## 4. How to Integrate with Backend
The backend can use the `predict_risk` function from `predict.py`. 

```python
from predict import predict_risk

# 1. Collect data from Frontend API request
user_data = {
    "Age": 24, 
    "Length_of_cycle": 35, 
    "Unusual_Bleeding": "yes",
    "Height": "5'4", 
    "Weight": 70, 
    # ... include other fields
}

# 2. Get the score
score = predict_risk(user_data)