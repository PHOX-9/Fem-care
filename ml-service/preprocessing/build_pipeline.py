import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------
# Custom function to clean and standardize height values
# ---------------------------------------------------------

def clean_height(X):
    """
    Converts height values into inches.
    Supports inputs like:
    - '5 ft 4 in','5.4','5'
    Missing or invalid values are converted to NaN.
    """
    def parse_height(h):
        # Handle missing or empty values
        if pd.isna(h) or str(h).strip() == "":
            return np.nan

        # Extract all numeric values from the string
        nums = re.findall(r'\d+', str(h))

        # Handle different formats
        if len(nums) >= 2:
            return int(nums[0]) * 12 + int(nums[1])
        elif len(nums) == 1:
            return int(nums[0]) * 12
        return np.nan  # Fall back for unexpected formats

    # Change parsing logic column-wise 
    return X.map(parse_height)

# ---------------------------------------------------------
# Feature grouping based on data type and preprocessing needs
# ---------------------------------------------------------

def get_feature_groups():
    """
    Feature groups based on source 
    """
    # Numerical features directly usable after scaling
    numeric_features = [
        "Age", "Length_of_cycle", "Estimated_day_of_ovulution", 
        "Length_of_Leutal_Phase", "Length_of_menses", "Weight", 
        "BMI", "Mean_of_length_of_cycle", "Menses_score", "number_of_peak"
    ]

    # Binary categorical feature (Yes/No style)
    categorical_features = ["Unusual_Bleeding"]

    # Features that require custom parsing logic
    custom_features = ["Height"]
    
    return numeric_features, categorical_features, custom_features


# ---------------------------------------------------------
# Build the complete preprocessing pipeline
# ---------------------------------------------------------

def build_preprocessing_pipeline():
    num_feats, cat_feats, custom_feats = get_feature_groups()

    # 1. Standard Numerical Pipeline
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 2. Categorical Pipeline 
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
    ])

    # 3. Custom Height Pipeline
    height_pipeline = Pipeline(steps=[
        ("cleaner", FunctionTransformer(clean_height)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Combine everything
    preprocessing_pipeline = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_feats),
            ("cat", categorical_pipeline, cat_feats),
            ("height", height_pipeline, custom_feats)
        ],
        remainder="drop" # Drop unused columns
    )

    return preprocessing_pipeline