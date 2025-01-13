from flask import Flask, request, jsonify
import pickle
import numpy as np

# Charger le modèle et le scaler
with open('model_rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def home():
    return "API de prédiction en ligne."

@app.route('/predict', methods=['POST'])
def predict():
    # Lire les données envoyées par le client
    data = request.get_json()
    perimeter = data.get('perimeter')
    area = data.get('area')

    # Validation des champs
    if perimeter is None or area is None:
        return jsonify({"error": "Champs manquants : 'perimeter' ou 'area'"}), 400

    # Préparer les données pour la prédiction
    features = np.array([[perimeter, area]])
    features_scaled = scaler.transform(features)

    # Faire la prédiction
    prediction = model.predict(features_scaled)
    return jsonify({"estimated_time": round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(debug=True)
