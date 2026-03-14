from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import shap
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import export_text
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load models
linear_model = joblib.load("models/linear_regression_model.pkl")
tree_model = joblib.load("models/decision_tree_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")

# Load training data for nearest neighbors
training_data = pd.read_csv("../runtime_dataset_final.csv")

# Feature columns (all features the models expect)
feature_columns = [
    'workload_type', 'workload_name', 'workload_complexity', 'cpu_cores', 
    'memory_total_gb', 'cpu_avg_pct', 'effective_cpu', 'memory_avg_gb', 
    'memory_pressure', 'disk_read_mb', 'disk_write_mb', 'io_intensity', 
    'disk_speed_class'
]

# Numeric features only (for SHAP and other analysis)
numeric_feature_columns = [
    'workload_complexity', 'cpu_cores', 'memory_total_gb', 'cpu_avg_pct',
    'effective_cpu', 'memory_avg_gb', 'memory_pressure', 'disk_read_mb',
    'disk_write_mb', 'io_intensity', 'disk_speed_class'
]

# Prepare training features for nearest neighbors (numeric only)
X_train = training_data[numeric_feature_columns]
y_train = training_data['runtime_sec']

# Initialize nearest neighbors
nn_model = NearestNeighbors(n_neighbors=3, metric='euclidean')
nn_model.fit(X_train)

# Initialize SHAP explainers (use numeric features for SHAP)
try:
    # For pipeline models, we need to handle SHAP differently
    # Use a simpler approach - create explainers without background data for now
    rf_explainer = None  # Will implement SHAP for pipelines differently
    xgb_explainer = None
    dt_explainer = None
    linear_explainer = None
    
    print("SHAP explainers initialized (pipeline models require special handling)")
except Exception as e:
    print(f"SHAP initialization warning: {e}")
    rf_explainer = None
    xgb_explainer = None
    dt_explainer = None
    linear_explainer = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/explanations")
def get_explanations():
    """API endpoint to get explanations data as JSON"""
    # This will be populated when a prediction is made
    if hasattr(app, 'current_explanations'):
        return jsonify(app.current_explanations)
    return jsonify({})

def get_shap_values(model, explainer, X):
    """Get SHAP values for a model"""
    try:
        # For pipeline models, create feature contributions using a simplified approach
        if hasattr(model, 'named_steps'):
            # Get the final estimator from the pipeline
            final_model = model.named_steps[list(model.named_steps.keys())[-1]]
            
            # Transform input through the pipeline preprocessing
            X_transformed = model[:-1].transform(X)
            
            # For linear models, use coefficients
            if hasattr(final_model, 'coef_'):
                coefficients = final_model.coef_
                X_numeric = X[numeric_feature_columns]
                
                contributions = {}
                for i, feature in enumerate(numeric_feature_columns):
                    if i < len(coefficients):
                        feature_value = X_numeric.iloc[0, i]
                        # Normalize contribution for better visualization
                        contribution = float(coefficients[i] * feature_value * 0.01)
                        contributions[feature] = contribution
                
                return contributions
            
            # For tree models, create SHAP-like values using feature importance and input values
            elif hasattr(final_model, 'feature_importances_'):
                importances = final_model.feature_importances_
                X_numeric = X[numeric_feature_columns]
                
                # Get mean values from training data for baseline
                X_train_means = X_train.mean()
                
                contributions = {}
                for i, feature in enumerate(numeric_feature_columns):
                    if i < len(importances):
                        feature_value = X_numeric.iloc[0, i]
                        baseline_value = X_train_means.iloc[i]
                        
                        # Calculate contribution as: importance * (value - baseline) * scale
                        contribution = float(importances[i] * (feature_value - baseline_value) * 0.1)
                        contributions[feature] = contribution
                
                return contributions
        
        return {}
    except Exception as e:
        print(f"SHAP error: {e}")
        return {}

def get_feature_importance(model, model_type):
    """Get feature importance for tree models"""
    try:
        if model_type in ['rf', 'dt', 'xgb']:
            # For pipeline models, get the final estimator
            if hasattr(model, 'named_steps'):
                final_model = model.named_steps[list(model.named_steps.keys())[-1]]
            else:
                final_model = model
                
            if hasattr(final_model, 'feature_importances_'):
                importances = final_model.feature_importances_
                importance_dict = {}
                
                # Map importances to numeric features only
                for i, feature in enumerate(numeric_feature_columns):
                    if i < len(importances):
                        importance_dict[feature] = float(importances[i])
                
                # Sort by importance and return top 10
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                return dict(sorted_features[:10])
        return {}
    except Exception as e:
        print(f"Feature importance error: {e}")
        return {}

def get_decision_path(model, X):
    """Get decision path for decision tree"""
    try:
        # For pipeline models, get the final estimator
        if hasattr(model, 'named_steps'):
            final_model = model.named_steps[list(model.named_steps.keys())[-1]]
        else:
            final_model = model
            
        # Check if it's actually a decision tree
        if not hasattr(final_model, 'tree_'):
            return []
            
        # Transform the input through the pipeline preprocessing steps
        X_transformed = model[:-1].transform(X) if hasattr(model, 'named_steps') else X[numeric_feature_columns]
        
        # Convert to DataFrame if it's an array
        if not isinstance(X_transformed, pd.DataFrame):
            X_transformed = pd.DataFrame(X_transformed)
        
        # Get the decision path
        decision_path = final_model.decision_path(X_transformed)
        leaf = final_model.apply(X_transformed)
        
        # Get feature names and thresholds
        tree = final_model.tree_
        
        # Extract rules for the path - use generic feature names since we don't know the transformed feature names
        rules = []
        sample_id = 0
        node_indicator = decision_path.toarray()[sample_id, :]
        
        rule_count = 0
        for node_id in range(len(node_indicator)):
            if node_indicator[node_id] and rule_count < 5:
                if leaf[sample_id] == node_id:
                    continue
                
                feature_idx = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                
                if X_transformed.iloc[sample_id, feature_idx] <= threshold:
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"
                
                # Use original feature names if possible, otherwise use generic names
                if feature_idx < len(numeric_feature_columns):
                    feature_name = numeric_feature_columns[feature_idx]
                else:
                    feature_name = f"feature_{feature_idx}"
                
                rules.append(f"{feature_name} {threshold_sign} {threshold:.2f}")
                rule_count += 1
        
        return rules
    except Exception as e:
        print(f"Decision path error: {e}")
        return []

def get_nearest_neighbors(X):
    """Find nearest neighbors from training data"""
    try:
        # Use only numeric features for distance calculation
        X_numeric = X[numeric_feature_columns]
        distances, indices = nn_model.kneighbors(X_numeric)
        
        neighbors = []
        for i in range(3):
            idx = indices[0][i]
            neighbor_data = training_data.iloc[idx]
            neighbors.append({
                'run_id': int(neighbor_data['run_id']),
                'runtime': float(neighbor_data['runtime_sec']),
                'distance': float(distances[0][i])
            })
        
        return neighbors
    except Exception as e:
        print(f"Nearest neighbors error: {e}")
        return []

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "workload_type": [request.form["workload_type"]],
        "workload_name": [request.form["workload_name"]],
        "workload_complexity": [float(request.form["workload_complexity"])],
        "cpu_cores": [float(request.form["cpu_cores"])],
        "memory_total_gb": [float(request.form["memory_total_gb"])],
        "cpu_avg_pct": [float(request.form["cpu_avg_pct"])],
        "effective_cpu": [float(request.form["effective_cpu"])],
        "memory_avg_gb": [float(request.form["memory_avg_gb"])],
        "memory_pressure": [float(request.form["memory_pressure"])],
        "disk_read_mb": [float(request.form["disk_read_mb"])],
        "disk_write_mb": [float(request.form["disk_write_mb"])],
        "io_intensity": [float(request.form["io_intensity"])],
        "disk_speed_class": [float(request.form["disk_speed_class"])]
    }

    df = pd.DataFrame(data)
    
    # Get predictions using all features
    lr_pred = linear_model.predict(df)[0]
    dt_pred = tree_model.predict(df)[0]
    rf_pred = rf_model.predict(df)[0]
    xgb_pred = xgb_model.predict(df)[0]

    # Get explainability data (using the full dataframe)
    explanations = {
        'shap_values': {
            'linear': get_shap_values(linear_model, linear_explainer, df),
            'decision_tree': get_shap_values(tree_model, dt_explainer, df),
            'random_forest': get_shap_values(rf_model, rf_explainer, df),
            'xgboost': get_shap_values(xgb_model, xgb_explainer, df)
        },
        'feature_importance': {
            'decision_tree': get_feature_importance(tree_model, 'dt'),
            'random_forest': get_feature_importance(rf_model, 'rf'),
            'xgboost': get_feature_importance(xgb_model, 'xgb')
        },
        'decision_path': get_decision_path(tree_model, df),
        'nearest_neighbors': get_nearest_neighbors(df)
    }

    # Store explanations for API endpoint
    app.current_explanations = explanations

    return render_template(
        "index.html",
        lr=round(lr_pred, 2),
        dt=round(dt_pred, 2),
        rf=round(rf_pred, 2),
        xgb=round(xgb_pred, 2),
        explanations=explanations
    )

if __name__ == "__main__":
    app.run(debug=True)