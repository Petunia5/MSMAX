from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Charger le modèle avec pickle
try:
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None

@app.route('/')
def home():
    return "API prête. Envoyez des requêtes POST à '/predict'."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Le modèle n'est pas disponible."}), 500

    try:
        # Récupérer les données envoyées dans la requête
        data = request.get_json()
        area = data.get('area')
        perimeter_to_area = data.get('perimeter_to_area')

        # Valider les entrées
        if area is None or perimeter_to_area is None:
            return jsonify({"error": "Les champs 'area' et 'perimeter_to_area' sont requis."}), 400

        # Faire une prédiction
        prediction = model.predict([[area, perimeter_to_area]])
        return jsonify({"predicted_time": prediction[0]})

    except Exception as e:
        return jsonify({"error": f"Erreur pendant la prédiction : {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
