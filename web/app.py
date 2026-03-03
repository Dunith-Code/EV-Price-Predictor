from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# 1. Load the model and mapping data
# Using the path structure you provided
MODEL_PATH = 'model/ev_price_model.pkl'

try:
    if os.path.exists(MODEL_PATH):
        saved_stuff = joblib.load(MODEL_PATH)
        model = saved_stuff['model']
        m_map = saved_stuff['m_map']
        features = saved_stuff['features']
        # Global average in case a brand is missing
        global_avg = sum(m_map.values()) / len(m_map)
    else:
        print(f"Error: {MODEL_PATH} not found.")
        model, m_map, features, global_avg = None, {}, [], 50000
except Exception as e:
    print(f"Error loading model: {e}")
    model, m_map, features, global_avg = None, {}, [], 50000

@app.route('/')
def index():
    if model is None:
        return "Error: Model file missing in 'model/' folder.", 500
    brands = sorted(m_map.keys())
    return render_template('index.html', brands=brands)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        result_text = "Error: Model not initialized."
    else:
        try:
            # Get data from form
            brand = request.form.get('brand')
            safety = float(request.form.get('safety'))
            year = int(request.form.get('year'))
            warranty = float(request.form.get('warranty'))
            model_display_name = request.form.get('model_name') # Just for UI display

            # 2. Translation Logic (Must match Colab)
            m_val = m_map.get(brand, global_avg)
            luxury_sig = m_val * (safety / 5)
            age = 2026 - year
            
            # 3. Prepare DataFrame 
            # Columns must be: ['M_Enc', 'Luxury_Signal', 'Age', 'Warranty_Years']
            input_df = pd.DataFrame([[m_val, luxury_sig, age, warranty]], 
                                    columns=features)
            
            # 4. Make prediction
            prediction = model.predict(input_df)[0]
            
            # Safety check: No negative prices
            final_price = max(5000, prediction)
            
            result_text = f"Estimated Price for {brand} {model_display_name}: ${final_price:,.2f} USD"
            
        except Exception as e:
            result_text = f"Error in calculation: {str(e)}"

    brands = sorted(m_map.keys())
    return render_template('index.html', brands=brands, prediction=result_text)

if __name__ == '__main__':
    # Set to port 7860 for Hugging Face compatibility
    app.run(host='0.0.0.0', port=7860, debug=True)