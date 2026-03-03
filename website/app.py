from flask import Flask, render_template, request

import joblib

import pandas as pd



app = Flask(__name__)



# 1. Load the model and mapping data

try:

    saved_stuff = joblib.load('model/ev_price_model.pkl')

    model = saved_stuff['model']

    m_map = saved_stuff['manufacturer_map']

    model_map = saved_stuff['model_map']

except FileNotFoundError:

    print("Error: 'model/ev_price_model.pkl' not found. Ensure it is in the model folder.")

    model = None

    m_map = {}

    model_map = {}



@app.route('/')

def index():

    # Pass the list of brands to the dropdown

    if model is None:

        return "Error: Model failed to load. Check if model/ev_price_model.pkl exists.", 500

    brands = sorted(m_map.keys())

    return render_template('index.html', brands=brands)



@app.route('/predict', methods=['POST'])

def predict():

    if model is None:

        result_text = "Error: Model failed to load. Check if model/ev_price_model.pkl exists."

    else:

        try:

            # Get data from the form using the 'name' attributes from HTML

            brand = request.form.get('brand')

            model_name = request.form.get('model')

            battery = float(request.form.get('battery'))

            autonomy = float(request.form.get('autonomy'))

            safety = float(request.form.get('safety'))

            year = int(request.form.get('year'))



            # Translation Logic (Target Encoding)

            m_enc = m_map.get(brand, sum(m_map.values()) / len(m_map))

            mo_enc = model_map.get(model_name, sum(model_map.values()) / len(model_map))

           

            # Calculate Tech_Score (Battery * Autonomy)

            tech_score = battery * autonomy

           

            # Prepare DataFrame for prediction (must match training column order)

            input_df = pd.DataFrame([[m_enc, mo_enc, tech_score, safety, year]],

                                    columns=['Manufacturer_Enc', 'Model_Enc', 'Tech_Score', 'Safety_Rating', 'Year'])

           

            # Make prediction

            prediction = model.predict(input_df)[0]

            result_text = f"Estimated Price for {brand} {model_name}: ${prediction:,.2f} USD"

           

        except Exception as e:

            result_text = f"Error in calculation: {str(e)}"



    brands = sorted(m_map.keys())

    return render_template('index.html', brands=brands, prediction=result_text)



if __name__ == '__main__':

    app.run(debug=True)