import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, 
           template_folder="../templates",
           static_folder="../static")

# Add CORS headers for Vercel
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# --------------- Load saved model artifacts ---------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model():
    """Load the trained XGBoost model, scaler, and feature metadata."""
    model_path = os.path.join(BASE_DIR, "xgb_poverty_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    meta_path = os.path.join(BASE_DIR, "feature_meta.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model file not found. Please ensure model files are deployed."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    meta = joblib.load(meta_path)
    return model, scaler, meta

try:
    xgb_model, scaler, feature_meta = load_model()
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠  Model loading error: {e}")
    xgb_model, scaler, feature_meta = None, None, None
    MODEL_LOADED = False

# Feature definitions (copy from original app.py)
FORM_FIELDS = [
    {
        "name": "male",
        "label": "Gender of Head of Household",
        "type": "select",
        "options": [("1", "Male"), ("2", "Female")],
        "default": "1",
    },
    {
        "name": "owner",
        "label": "Home Ownership",
        "type": "select",
        "options": [("1", "Owner"), ("2", "Renter / Other")],
        "default": "1",
    },
    {
        "name": "urban",
        "label": "Area Type",
        "type": "select",
        "options": [("1", "Urban"), ("2", "Rural")],
        "default": "1",
    },
    {
        "name": "water",
        "label": "Access to Water",
        "type": "select",
        "options": [("1", "Access"), ("2", "No Access")],
        "default": "1",
    },
    {
        "name": "toilet",
        "label": "Access to Toilet",
        "type": "select",
        "options": [("1", "Access"), ("2", "No Access")],
        "default": "1",
    },
    {
        "name": "sewer",
        "label": "Access to Sewer",
        "type": "select",
        "options": [("1", "Access"), ("2", "No Access")],
        "default": "1",
    },
    {
        "name": "elect",
        "label": "Access to Electricity",
        "type": "select",
        "options": [("1", "Access"), ("2", "No Access")],
        "default": "1",
    },
    {
        "name": "water_source",
        "label": "Water Source",
        "type": "select",
        "options": [
            ("1", "Piped water into dwelling"),
            ("3", "Protected dug well"),
            ("4", "Surface water"),
            ("5", "Other"),
        ],
        "default": "1",
    },
    {
        "name": "sanitation_source",
        "label": "Sanitation Source",
        "type": "select",
        "options": [
            ("1", "Piped sewer system"),
            ("2", "Septic tank"),
            ("3", "Pit latrine with slab"),
            ("4", "No facilities / bush / field"),
            ("5", "Other"),
        ],
        "default": "1",
    },
    {
        "name": "dweltyp",
        "label": "Dwelling Type",
        "type": "select",
        "options": [
            ("1", "Detached house"),
            ("2", "Several buildings connected"),
            ("3", "Separate apartment"),
            ("4", "Other"),
        ],
        "default": "1",
    },
    {
        "name": "employed",
        "label": "Employment Status",
        "type": "select",
        "options": [("1", "Employed"), ("2", "Unemployed / Other")],
        "default": "1",
    },
    {
        "name": "educ_max",
        "label": "Maximum Education Level",
        "type": "select",
        "options": [
            ("1", "Complete Primary Education"),
            ("2", "Incomplete Primary Education"),
            ("3", "Complete Secondary Education"),
            ("4", "Incomplete Secondary Education"),
            ("5", "Complete Tertiary Education"),
            ("6", "Incomplete Tertiary Education"),
        ],
        "default": "3",
    },
    {
        "name": "any_nonagric",
        "label": "Non-Agricultural Income",
        "type": "select",
        "options": [("1", "Yes"), ("2", "No")],
        "default": "1",
    },
    {
        "name": "sector1d",
        "label": "Employment Sector",
        "type": "select",
        "options": [
            ("1", "Transport, storage & communications"),
            ("2", "Public administration & defence"),
            ("3", "Construction"),
            ("4", "Manufacturing"),
            ("5", "Wholesale & retail trade"),
            ("6", "Education"),
            ("7", "Agriculture, hunting, forestry & fishing"),
            ("8", "Health & social work"),
            ("9", "Other services"),
            ("10", "Hotels & restaurants"),
            ("11", "Financial intermediation"),
            ("12", "Real estate & business activities"),
            ("13", "Community & social services"),
            ("14", "Electricity, gas & water supply"),
            ("15", "Mining & quarrying"),
            ("16", "Other"),
        ],
        "default": "5",
    },
    {
        "name": "hhsize",
        "label": "Household Size (members)",
        "type": "number",
        "min": 1,
        "max": 30,
        "default": "4",
    },
    {
        "name": "rooms",
        "label": "Number of Rooms",
        "type": "number",
        "min": 1,
        "max": 20,
        "default": "3",
    },
    {
        "name": "age",
        "label": "Age of Head of Household",
        "type": "number",
        "min": 15,
        "max": 100,
        "default": "35",
    },
]

def build_feature_vector(form_data: dict) -> np.ndarray:
    """
    Convert the form inputs into a feature vector that matches
    exactly what the trained model expects (same column order & count).
    Missing features are filled with saved training-set median defaults.
    """
    feature_names = feature_meta["feature_names"]
    defaults = feature_meta["feature_defaults"]

    row = defaults.copy()  # start from median defaults

    # Map form fields → matching feature columns
    for key, value in form_data.items():
        if key in row:
            row[key] = float(value)

    # Build DataFrame with correct column order
    df = pd.DataFrame([row], columns=feature_names)
    return df.values

def classify_poverty(expenditure: float) -> dict:
    """Return a human-readable poverty classification with color."""
    if expenditure < 2.15:
        return {"level": "Extreme Poverty", "color": "#dc2626", "icon": "🔴", "desc": "Below the international extreme poverty line ($2.15/day PPP)."}
    elif expenditure < 3.65:
        return {"level": "Moderate Poverty", "color": "#ea580c", "icon": "🟠", "desc": "Below the lower-middle income poverty line ($3.65/day PPP)."}
    elif expenditure < 6.85:
        return {"level": "Vulnerable", "color": "#ca8a04", "icon": "🟡", "desc": "Below the upper-middle income poverty line ($6.85/day PPP)."}
    elif expenditure < 15:
        return {"level": "Near Average", "color": "#2563eb", "icon": "🔵", "desc": "Around the average expenditure level."}
    else:
        return {"level": "Above Average", "color": "#16a34a", "icon": "🟢", "desc": "Above average household expenditure."}

@app.route("/")
def index():
    """Main page with prediction form."""
    return render_template("index.html", fields=FORM_FIELDS, model_loaded=MODEL_LOADED)

@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded."}), 503

    try:
        form_data = request.get_json()
        features = build_feature_vector(form_data)
        features_scaled = scaler.transform(features)
        prediction = float(xgb_model.predict(features_scaled)[0])
        classification = classify_poverty(prediction)

        return jsonify({
            "prediction": round(prediction, 2),
            "classification": classification,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)