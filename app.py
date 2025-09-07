# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

MODEL_PATH = "models/pipeline.joblib"

app = Flask(__name__)
CORS(app)

# load model artifacts
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found. Run train_model.py to create {MODEL_PATH}")

artifacts = joblib.load(MODEL_PATH)
pipeline = artifacts['pipeline']
numeric_features = artifacts['numeric_features']
categorical_features = artifacts['categorical_features']
all_features = numeric_features + categorical_features

@app.route("/")
def index():
    return jsonify({"status":"ok", "features": all_features})

def validate_and_frame(data_json):
    # data_json should be dict-like with keys equal to all_features
    # We'll create a single-row DataFrame, filling missing features with None
    row = {}
    for f in all_features:
        # allow both string and numeric for input
        row[f] = data_json.get(f, None)
    df = pd.DataFrame([row], columns=all_features)
    # convert numeric columns to numeric dtype when possible
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    if not payload:
        return jsonify({"error":"Invalid JSON payload"}), 400
    # support nested under 'data' or raw dict
    if 'data' in payload:
        data = payload['data']
    else:
        data = payload

    try:
        df = validate_and_frame(data)
    except Exception as e:
        return jsonify({"error": f"Failed to parse input: {str(e)}"}), 400

    try:
        proba = pipeline.predict_proba(df)[:, 1][0]
        pred = pipeline.predict(df)[0]
        label = 'yes' if int(pred) == 1 else 'no'
        return jsonify({
            "prediction": label,
            "probability": float(proba),
            "raw_prediction": int(pred)
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
