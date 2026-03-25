import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import os

print("Loading dataset...")
df = pd.read_csv("../runtime_dataset_final.csv")

feature_columns = [
    'workload_type', 'workload_name', 'workload_complexity', 'cpu_cores', 
    'memory_total_gb', 'cpu_avg_pct', 'effective_cpu', 'memory_avg_gb', 
    'memory_pressure', 'disk_read_mb', 'disk_write_mb', 'io_intensity', 
    'disk_speed_class'
]

numeric_feature_columns = [
    'workload_complexity', 'cpu_cores', 'memory_total_gb', 'cpu_avg_pct',
    'effective_cpu', 'memory_avg_gb', 'memory_pressure', 'disk_read_mb',
    'disk_write_mb', 'io_intensity', 'disk_speed_class'
]
categorical_feature_columns = ['workload_type', 'workload_name']

X = df[feature_columns]
y = df['runtime_sec']

print("Creating Preprocessor...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_feature_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feature_columns)
    ])

print("Training Linear Regression...")
linear_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', LinearRegression())])
linear_pipeline.fit(X, y)
joblib.dump(linear_pipeline, "models/linear_regression_model.pkl")

print("Training Decision Tree...")
dt_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', DecisionTreeRegressor(max_depth=10, random_state=42))])
dt_pipeline.fit(X, y)
joblib.dump(dt_pipeline, "models/decision_tree_model.pkl")

print("Training Random Forest...")
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))])
rf_pipeline.fit(X, y)
joblib.dump(rf_pipeline, "models/random_forest_model.pkl")

print("Training XGBoost...")
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', xgb.XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1))])
xgb_pipeline.fit(X, y)
joblib.dump(xgb_pipeline, "models/xgboost_model.pkl")

print("All models have been retrained and saved successfully in models/ directory!")
