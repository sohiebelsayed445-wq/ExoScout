import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 1. Initialize Flask App and CORS ---
app = Flask(__name__)
# Enable CORS for all origins, allowing your frontend (index.html) to communicate with this server
CORS(app) 

# --- 2. Load Model Assets (The core of the backend) ---
try:
    MODEL = joblib.load("exoplanet_model.joblib")
    SCALER = joblib.load("exoplanet_scaler.joblib")
    MEDIANS = joblib.load("exoplanet_medians.joblib")
    FEATURES = joblib.load("exoplanet_features.joblib")
    print("✅ ML Assets loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ ERROR: Missing ML asset file: {e}")
    print("Ensure all .joblib files are in the same directory.")
    # Exit or handle gracefully if critical files are missing
    exit()

# --- 3. Define the Prediction Endpoint ---
@app.route('/api/predict', methods=['POST'])
def predict():
    """Receives planet data from the frontend, processes it, and returns a prediction."""
    try:
        # Get data sent as JSON from the frontend
        data = request.json
        
        # Convert the dictionary of features into a DataFrame row
        # We must ensure the column order is exactly the same as the trained FEATURES list
        input_df = pd.DataFrame([data])
        
        # 3.1. Imputation (Fill Nulls with Training Medians)
        # Use MEDIANS series to fill any missing/null values (which will be np.nan)
        # We handle this by setting the index and aligning the fill operation
        input_df = input_df[FEATURES]  # Ensure order
        input_df = input_df.fillna(MEDIANS)
        
        # 3.2. Scaling (Scale the input data using the trained scaler)
        X_scaled = SCALER.transform(input_df)
        
        # 3.3. Prediction
        prediction = MODEL.predict(X_scaled)[0]
        probabilities = MODEL.predict_proba(X_scaled)[0]
        
        # Determine confidence and result
        is_candidate = int(prediction) == 1
        confidence = probabilities[1] if is_candidate else probabilities[0]
        result_label = "CANDIDATE (CP)" if is_candidate else "FALSE POSITIVE (FP)"
        
        # 4. Return results as JSON
        return jsonify({
            'prediction': int(prediction),
            'result_label': result_label,
            'confidence': round(confidence * 100, 2),
            'status': 'success'
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- 4. Run the Server ---
if __name__ == '__main__':
    # Running on http://127.0.0.1:5000/
    app.run(debug=True)