from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler with error handling
model, scaler = None, None
try:
    with open('aqi_rf_model_final.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except Exception as e:
    print(f"Error loading scaler: {e}")

# Function to categorize AQI values
def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good", "Air quality is considered satisfactory."
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "People with respiratory conditions should reduce outdoor activities."
    elif aqi <= 200:
        return "Unhealthy", "Everyone may start to feel health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert: everyone may experience serious health effects."
    else:
        return "Hazardous", "Emergency conditions: the entire population is likely to be affected."

# Function to suggest solutions based on feature contribution
def suggest_solutions(input_values):
    pollutants = ['PM2.5', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3']
    try:
        max_pollutant = pollutants[np.argmax(input_values)]
    except:
        return "Unable to determine primary pollutant."
    
    if max_pollutant == 'PM2.5':
        return "Reduce PM2.5 by limiting vehicle emissions and using cleaner fuels."
    elif max_pollutant == 'NO':
        return "Reduce NO levels by using public transport or carpooling."
    elif max_pollutant == 'NO2':
        return "Improve NO2 levels by reducing industrial emissions."
    elif max_pollutant == 'NOx':
        return "Lower NOx by limiting fossil fuel combustion and vehicle emissions."
    elif max_pollutant == 'CO':
        return "Reduce CO by using clean energy sources and improving ventilation."
    elif max_pollutant == 'SO2':
        return "Reduce SO2 by minimizing coal use and promoting clean energy."
    elif max_pollutant == 'O3':
        return "Reduce O3 by avoiding the use of gas-powered engines during hot weather."
    else:
        return "Maintain clean air practices."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        pm25 = request.form.get('PM2.5', 0)
        no = request.form.get('NO', 0)
        no2 = request.form.get('NO2', 0)
        nox = request.form.get('NOx', 0)
        co = request.form.get('CO', 0)
        so2 = request.form.get('SO2', 0)
        o3 = request.form.get('O3', 0)

        # Validate inputs: ensure they are numeric and non-negative
        input_values = [pm25, no, no2, nox, co, so2, o3]
        input_values = [float(value) if value else 0 for value in input_values]

        # Handle negative values
        if any(val < 0 for val in input_values):
            return "Pollutant values cannot be negative.", 400

        # Check if model or scaler is loaded
        if model is None or scaler is None:
            return render_template('index.html', 
                                   error="Model or scaler not loaded. Please try again later.",
                                   pm25=pm25, no=no, no2=no2, nox=nox, co=co, so2=so2, o3=o3)

        # Create DataFrame for input data
        input_data = pd.DataFrame([input_values], columns=['PM2.5', 'NO', 'NO2', 'NOx', 'CO', 'SO2', 'O3'])

        # Scale input data
        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_data)
        formatted_prediction = round(prediction[0], 2)

        # Categorize AQI
        category, health_advice = categorize_aqi(formatted_prediction)

        # Suggest solutions based on highest contributing pollutant
        solutions = suggest_solutions(input_values)

        # Pass inputs, prediction, category, and suggestions to the template
        return render_template('index.html', 
                               prediction=formatted_prediction, 
                               category=category, 
                               health_advice=health_advice,
                               solutions=solutions,
                               pm25=pm25, no=no, no2=no2, nox=nox, co=co, so2=so2, o3=o3)

    except ValueError:
        return render_template('index.html', error="Invalid input. Please enter valid numeric values.", 
                               pm25=pm25, no=no, no2=no2, nox=nox, co=co, so2=so2, o3=o3)
    except Exception as e:
        return render_template('index.html', error=str(e), 
                               pm25=pm25, no=no, no2=no2, nox=nox, co=co, so2=so2, o3=o3)

@app.route('/refresh')
def refresh():
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
