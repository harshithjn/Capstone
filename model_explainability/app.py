from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load models
linear_model = joblib.load("models/linear_regression_model.pkl")
tree_model = joblib.load("models/decision_tree_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

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

    lr_pred = linear_model.predict(df)[0]
    dt_pred = tree_model.predict(df)[0]
    rf_pred = rf_model.predict(df)[0]
    xgb_pred = xgb_model.predict(df)[0]

    return render_template(
        "index.html",
        lr=round(lr_pred,2),
        dt=round(dt_pred,2),
        rf=round(rf_pred,2),
        xgb=round(xgb_pred,2)
    )

if __name__ == "__main__":
    app.run(debug=True)