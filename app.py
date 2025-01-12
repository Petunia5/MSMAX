from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)  # Autorise toutes les origines pour simplifier les tests

# Charger le modèle
try:
    with open("random_forest_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    print("Modèle chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

# Définir la route principale pour les prédictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les données JSON envoyées avec la requête
        data = request.get_json()

        # Vérifier que les données nécessaires sont présentes
        if "area" not in data or "perimeter_to_area" not in data:
            return jsonify({"error": "Les données doivent inclure 'area' et 'perimeter_to_area'"}), 400

        # Préparer les données pour le modèle
        area = data["area"]
        perimeter_to_area = data["perimeter_to_area"]
        input_data = np.array([[area, perimeter_to_area]])

        # Faire une prédiction
        predicted_time = model.predict(input_data)[0]

        # Retourner la prédiction en réponse JSON
        return jsonify({"predicted_time": predicted_time})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Lancer l'application
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

