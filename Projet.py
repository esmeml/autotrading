#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 11:11:34 2025

@author: ALEX
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import mplfinance as mpf
from mplfinance.original_flavor import candlestick_ohlc as candlestick

########################## TRAITER  LA BDD ############################

# Charger les données
df = pd.read_csv('Donnees_CAC40.csv')

print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

carac_spec = r'^[^a-zA-Z0-9 ]+$'
total = 0
for col in df.select_dtypes(include='object').columns:
    total += df[col].str.match(carac_spec).sum()
print(total)

for col in df.columns:
    if df[col].dtype == 'object':
        try:
            # Remplacer les virgules des milliers et convertir le point décimal en virgule
            df[col] = df[col].str.replace(",", "")       # Supprime les séparateurs de milliers
            df[col] = df[col].str.replace(".", ".", regex=False)  # Optionnel : garde les points
            df[col] = df[col].str.replace(",", ".")      # Remplace la virgule décimale par un point
            df[col] = df[col].astype(float)              # Conversion finale en float
        except:
            # Ignore les colonnes qui ne peuvent pas être converties (ex: noms de sociétés)
            pass
        
df['Vol.'] = df['Vol.'].str.extract(r'(\d+)')
df['Vol.'] = df['Vol.'].astype(int)


df.drop(columns=['Change %'], inplace=True)


#Date = Index
df.set_index("Date", inplace=True)



#Normalisation 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

# Supposons que X est ton DataFrame contenant les features
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Remettre dans un DataFrame avec les noms de colonnes originaux
df_scaled_df = pd.DataFrame(df_scaled, columns=df.columns)

df_scaled_df.boxplot(figsize=(15, 6))
plt.title("Boxplot des variables normalisées")
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

def plot_cac40_data(df):
    plt.figure(figsize=(12, 5))
    
    # Tracer le prix de clôture
    price_array = df["Price"].values.reshape(-1, 1)
    plt.plot(df.index, price_array[:, 0], label="Prix de clôture", color="blue")
    
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.title("Évolution du prix de clôture du CAC 40")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Tracer le volume des transactions
    plt.figure(figsize=(12, 5))
    vol_array = df["Vol."].values.reshape(-1, 1)
    plt.plot(df.index, vol_array[:, 0], label="Volume des transactions", color="green")
    
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.title("Évolution du volume des transactions du CAC 40")
    plt.legend()
    plt.grid()
    plt.show()
    

# Appel de la fonction avec tes données nettoyées
plot_cac40_data(df)


####################################### RIDGE #######################################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Séparation features / cible
X = df_scaled_df.drop(columns=["Price"])
y = df_scaled_df["Price"]  # La vraie valeur non normalisée

# Split train/test avec random_state pour reproductibilité
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Entraînement du modèle Ridge
model = Ridge()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Métriques d'évaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Affichage des résultats
print(f"R² Score : {r2:.4f}")
print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"Mean Absolute Error (MAE) : {mae:.4f}")

####################################### LASSO #######################################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Séparation features / cible
X = df_scaled_df.drop(columns=["Price"])
y = df_scaled_df["Price"]  # La vraie valeur non normalisée

# Split train/test avec random_state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)

# Entraînement du modèle LASSO
model = Lasso(alpha=0.1)  # Tu peux ajuster alpha
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Métriques d'évaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Affichage des résultats
print(f"R² Score : {r2:.4f}")
print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"Mean Absolute Error (MAE) : {mae:.4f}")

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Suppose que ton DataFrame d’origine s'appelle `engie_df` avec "Close" comme cible
# et que les features sont déjà prêtes (lags + moyennes mobiles)

# Étape 1 : Normalisation
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled_df = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

# Étape 2 : Séparer features et cible
X = df_scaled_df.drop(columns=["Price"])
y = df_scaled_df["Price"]  # on garde la vraie valeur (non normalisée)

# Étape 3 : Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Étape 4 : Définir les modèles
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42)
}

# Étape 5 : Entraînement et évaluation
for name, model in models.items():
    print(f"\n🧠 {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Scores
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"➡️ R² Score : {r2:.4f}")
    print(f"📉 Mean Squared Error (MSE) : {mse:.4f}")
    print(f"📈 Mean Absolute Error (MAE) : {mae:.4f}")


