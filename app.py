from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('house_price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        med_inc = float(request.form['MedInc'])
        house_age = float(request.form['HouseAge'])
        ave_rooms = float(request.form['AveRooms'])
        ave_bedrms = float(request.form['AveBedrms'])
        population = float(request.form['Population'])
        ave_occup = float(request.form['AveOccup'])
        latitude = float(request.form['Latitude'])
        longitude = float(request.form['Longitude'])

        # Prepare data for prediction
        input_data = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        predicted_price = prediction[0] * 100000  # Convert to actual price in dollars

        return jsonify({'prediction': float(predicted_price)})

if __name__ == '__main__':
    app.run(debug=True)


