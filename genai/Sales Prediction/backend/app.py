from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__, static_folder='../frontend')
CORS(app)  # Enable CORS

# Load the trained model
try:
    model = pickle.load(open('model_20250228_104704.pkl', 'rb'))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debug statement

        # Extract input values
        TV = float(data.get('TV', 0))
        Radio = float(data.get('Radio', 0))
        Newspaper = float(data.get('Newspaper', 0))

        # Prepare features for prediction
        features = np.array([[TV, Radio, Newspaper]])
        print("Features:", features)  # Debug statement

        # Make prediction
        prediction = model.predict(features)
        print("Prediction:", prediction)  # Debug statement

        # Return prediction as JSON
        return jsonify({'prediction': round(float(prediction[0]), 2)})
    except Exception as e:
        print(f"Error during prediction: {e}")  # Debug statement
        return jsonify({'error': str(e)}), 500

@app.route('/get-data', methods=['GET'])
def get_data():
    try:
        data = pd.read_csv('data/advertising.csv')
        print("Data loaded successfully!")  # Debug statement
        return jsonify(data.to_dict(orient='records'))
    except Exception as e:
        print(f"Error loading data: {e}")  # Debug statement
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)