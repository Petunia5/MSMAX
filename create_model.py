import pickle
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Génération de données fictives pour l'entraînement
X = np.array([[1000, 0.02], [2000, 0.03], [3000, 0.04], [4000, 0.05]])
y = np.array([10, 20, 30, 40])  # Temps estimés

# Entraîner le modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Sauvegarder le modèle avec pickle
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modèle sauvegardé avec succès sous 'random_forest_model.pkl'.")
