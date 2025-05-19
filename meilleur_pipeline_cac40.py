

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from copy import copy

# Chargement des données
tpot_data = pd.read_csv("Donnees_CAC40.csv", sep=',')

# Nettoyage
for col in tpot_data.columns:
    if tpot_data[col].dtype == 'object':
        try:
            tpot_data[col] = tpot_data[col].str.replace(",", "")
            tpot_data[col] = tpot_data[col].str.replace(",", ".")
            tpot_data[col] = tpot_data[col].astype(float)
        except:
            pass

tpot_data['Vol.'] = tpot_data['Vol.'].astype(str).str.extract(r'(\d+)')
tpot_data['Vol.'] = tpot_data['Vol.'].astype(float)
tpot_data.drop(columns=['Change %'], inplace=True)

# Rename pour TPOT
tpot_data.rename(columns={"Price": "target"}, inplace=True)
tpot_data.drop(columns=['Date'], inplace=True)

# Split
features = tpot_data.drop('target', axis=1)
target = tpot_data['target']
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=42, shuffle=False)

# Pipeline exporté par TPOT
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.8, tol=0.001)),
    ElasticNetCV(l1_ratio=1.0, tol=1e-05)
)

set_param_recursive(exported_pipeline.steps, 'random_state', 42)

# Entraînement
exported_pipeline.fit(X_train, y_train)
y_pred = exported_pipeline.predict(X_test)

# Métriques
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score du pipeline TPOT : {r2:.4f}")
print(f"Mean Squared Error : {mse:.4f}")
print(f"Mean Absolute Error : {mae:.4f}")

#################################
# VISUALISATIONS
#################################

# Courbe 1 : Évolution réelle vs prédite
plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label="Prix réel", color="blue")
plt.plot(y_pred, label="Prix prédit", color="orange")
plt.title("📈 Évolution du prix réel vs prédiction")
plt.xlabel("Index (jour)")
plt.ylabel("Prix")
plt.legend()
plt.grid(True)
plt.show()
# 🔍 Sert à comparer les dynamiques temporelles entre les vraies valeurs et les prévisions

# Courbe 2 : Erreurs de prédiction
errors = y_test.values - y_pred
plt.figure(figsize=(12, 5))
plt.plot(errors, label="Erreur de prédiction", color="red")
plt.axhline(0, color='black', linestyle='--')
plt.title("📉 Erreurs de prédiction")
plt.xlabel("Index (jour)")
plt.ylabel("Erreur (réel - prédit)")
plt.legend()
plt.grid(True)
plt.show()
# 🔍 Permet d'identifier si les erreurs sont systématiques (biais) ou aléatoires

# Courbe 3 : Distribution des erreurs
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, color="purple", edgecolor="black")
plt.title("📊 Distribution des erreurs")
plt.xlabel("Erreur")
plt.ylabel("Fréquence")
plt.grid(True)
plt.show()
# 🔍 Donne une idée de la variance et du biais du modèle

# Courbe 4 : Scatter plot Réel vs Prédit
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ligne diagonale parfaite
plt.xlabel("Valeur réelle")
plt.ylabel("Valeur prédite")
plt.title("⚖️ Réel vs Prédit")
plt.grid(True)
plt.show()
# 🔍 Si les points sont proches de la diagonale : modèle précis. Sinon : erreurs importantes



