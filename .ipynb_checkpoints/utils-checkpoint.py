import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def preprocess_any_file(df):

    features_min_max = [f for f in [
        'Temperature_C', 'Vibration_mms', 'Sound_dB',
        'Oil_Level_pct', 'Coolant_Level_pct',
        'Power_Consumption_kW', 'Heat_Index'
    ] if f in df.columns]

    features_standard = [f for f in [
        'Installation_Year', 'Operational_Hours',
        'Last_Maintenance_Days_Ago', 'Maintenance_History_Count',
        'Failure_History_Count'
    ] if f in df.columns]


    known_features = set(features_min_max + features_standard)
    unknown_features = [
        col for col in df.select_dtypes(include=['number']).columns
        if col not in known_features
    ]

    scalers = ColumnTransformer([
        ("min_max", MinMaxScaler(), features_min_max),
        ("standard", StandardScaler(), features_standard + unknown_features)
    ])

    num_transformer = Pipeline([("scalers", scalers)])
    cat_transformer = Pipeline([("encoder", OneHotEncoder())])

    preprocessor = ColumnTransformer([
        ('num_transformer', num_transformer, make_column_selector(dtype_include='number')),
        ('cat_transformer', cat_transformer, make_column_selector(dtype_include='object'))
    ])
    return preprocessor



def preprocessor_pipeline(regression_model, df=None):
    
    if df is not None:
        preprocessor = preprocess_any_file(df)
    else:
        preprocessor = preprocess_any_file 

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", regression_model)
    ])
    return pipeline
