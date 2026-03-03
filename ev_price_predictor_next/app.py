from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# --- DATABASE CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'ev_predictions.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- DATABASE TABLE BLUEPRINT ---
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    brand = db.Column(db.String(50))
    model_name = db.Column(db.String(50))
    year = db.Column(db.Integer)
    predicted_price = db.Column(db.Float)

# Create the database file and table automatically
with app.app_context():
    db.create_all()

# --- ML MODEL LOADING ---
model = None
scaler = None
m_map = {}
features = []
global_avg = 50000
load_error = "None"

MODEL_PATH = os.path.join(BASE_DIR, 'model', 'ev_price_model.pkl')

try:
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        model = data.get('model')
        scaler = data.get('scaler')
        m_map = data.get('m_map', {})
        features = data.get('features', [])
        
        if m_map:
            global_avg = sum(m_map.values()) / len(m_map)
        
        if model is None:
            load_error = "File loaded, but 'model' key is missing inside the .pkl"
        else:
            print("✅ SUCCESS: XGBoost Model loaded into memory.")
    else:
        load_error = f"File not found at: {MODEL_PATH}"
except Exception as e:
    load_error = f"Joblib/XGBoost Load Error: {str(e)}"
    print(f"❌ CRITICAL LOAD ERROR: {e}")

@app.route('/')
def index():
    if model is None:
        return f"<h1>Backend Error</h1><p><b>Status:</b> {load_error}</p>", 500
    
    brands = sorted(m_map.keys())
    return render_template('index.html', brands=brands)

@app.route('/predict', methods=['POST'])
def predict():
    brands = sorted(m_map.keys())
    try:
        # 1. Get data from form
        brand = request.form.get('brand')
        model_display_name = request.form.get('model_name') # Corrected variable name
        year = int(request.form.get('year', 2024))
        battery = float(request.form.get('battery', 0))
        range_val = float(request.form.get('range', 0))
        
        # 2. Feature Engineering
        age = 2026 - year
        efficiency = range_val / (battery + 1e-6)

        user_input = {
            'M_Enc': m_map.get(brand, global_avg),
            'Battery_Capacity_kWh': battery,
            'Autonomous_Level': float(request.form.get('autonomy', 0)),
            'Safety_Rating': float(request.form.get('safety', 0)),
            'Vehicle_Age': age,
            'Warranty_Years': float(request.form.get('warranty', 0)),
            'Charge_Time_hr': float(request.form.get('charge_time', 0)),
            'Range_km': range_val,
            'Efficiency_Score': efficiency
        }

        # 3. Predict
        input_df = pd.DataFrame([user_input])[features]
        final_input = scaler.transform(input_df) if scaler else input_df
        prediction = model.predict(final_input)[0]
        final_price = float(max(5000, prediction))

        # 4. SAVE TO DATABASE
        new_entry = PredictionHistory(
            brand=brand,
            model_name=model_display_name,
            year=year,
            predicted_price=final_price
        )
        db.session.add(new_entry)
        db.session.commit()

        result_text = f"Estimated Price for {brand} {model_display_name}: ${final_price:,.2f} USD"
        return render_template('index.html', prediction=result_text, brands=brands)
    
    except Exception as e:
        error_text = f"Prediction Error: {str(e)}"
        return render_template('index.html', prediction=error_text, brands=brands)

@app.route('/history')
def history():
    all_predictions = PredictionHistory.query.order_by(PredictionHistory.id.desc()).all()
    return render_template('history.html', predictions=all_predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)