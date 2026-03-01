from flask import Flask, request, jsonify
import os
import xgboost as xgb
import pandas as pd
import datetime
import requests

app = Flask(__name__)

# Build the absolute path to the model file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
model_path = os.path.join(base_dir, "production_global_drought_model.json")

# Load the model
model = xgb.XGBRegressor()
model.load_model(model_path)

# --- 2. NASA API LOGIC FROM YOUR NOTEBOOK ---
def fetch_live_weather(lat, lon):
    end_date = datetime.datetime.now() - datetime.timedelta(days=2) 
    start_date = end_date - datetime.timedelta(days=7)
    start_str, end_str = start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')
    
    parameters = "PRECTOTCORR,PS,QV2M,T2M,T2MDEW,T2MWET,T2M_MAX,T2M_MIN,T2M_RANGE,TS,WS10M"
    url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
           f"parameters={parameters}&community=AG&longitude={lon}&latitude={lat}"
           f"&start={start_str}&end={end_str}&format=JSON")
    
    response = requests.get(url)
    if response.status_code != 200:
        return None
        
    data = response.json()['properties']['parameter']
    features = {}
    for param in parameters.split(','):
        daily_values = [v for v in data[param].values() if v != -999.0]
        features[param] = sum(daily_values) / len(daily_values) if daily_values else 0.0
    return features

# --- 3. THE API ENDPOINT ---
@app.route('/predict', methods=['GET'])
def predict():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if lat is None or lon is None:
        return "Please provide lat and lon parameters.", 400

    live_features = fetch_live_weather(lat, lon)
    if not live_features:
        return "Failed to fetch NASA data.", 500
    
    # Run Inference
    expected_columns = ["PRECTOTCORR","PS","QV2M","T2M","T2MDEW","T2MWET","T2M_MAX","T2M_MIN","T2M_RANGE","TS","WS10M"]
    df_input = pd.DataFrame([live_features])[expected_columns]
    prediction = loaded_model.predict(df_input)[0]
    
    # Formatted Output String
    output = (
        f"📡 LIVE TELEMETRY RECEIVED:\n"
        f"   Avg Temperature : {live_features['T2M']:.2f} °C\n"
        f"   Max Temperature : {live_features['T2M_MAX']:.2f} °C\n"
        f"   Recent Rainfall : {live_features['PRECTOTCORR']:.2f} mm/day\n"
        f"   Specific Humidity : {live_features['QV2M']:.2f} g/kg\n"
        f"{'='*50}\n"
        f"🚨 AI DROUGHT RISK SCORE: {prediction:.2f} / 4.0\n"
        f"{'='*50}"
    )
    return output

if __name__ == '__main__':

    app.run(debug=True)
