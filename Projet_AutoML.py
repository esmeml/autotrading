#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:44:10 2025

@author: louisbage
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Pour suppression des warnings inutiles
import warnings
warnings.filterwarnings('ignore')

# ========== 1. CHARGEMENT ET NETTOYAGE ==========
df = pd.read_csv('Donnees_CAC40.csv')

# Traitement des colonnes numériques mal formatées
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = df[col].str.replace(",", "")
            df[col] = df[col].str.replace(".", ".", regex=False)
            df[col] = df[col].str.replace(",", ".")
            df[col] = df[col].astype(float)
        except:
            pass

# Nettoyage spécifique de la colonne Volume
df['Vol.'] = df['Vol.'].str.extract(r'(\d+)')
df['Vol.'] = df['Vol.'].astype(int)

# Supprimer la colonne inutile
df.drop(columns=['Change %'], inplace=True)

# Index = Date
df.set_index("Date", inplace=True)

# ========== 2. NORMALISATION ==========
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)
df_scaled_df = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

# ========== 3. AFFICHAGE DES COURBES ==========
def plot_cac40_data(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Price"], label="Prix de clôture", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.title("Évolution du prix de clôture du CAC 40")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Vol."], label="Volume des transactions", color="green")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.title("Évolution du volume des transactions du CAC 40")
    plt.legend()
    plt.grid()
    plt.show()

plot_cac40_data(df)

# ========== 4. AUTO ML AVEC TPOT ==========
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split

# Séparation X / y
X = df_scaled_df.drop(columns=["Price"])
y = df_scaled_df["Price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)

# Création du modèle TPOT
tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42, n_jobs=-1)

# Entraînement du modèle AutoML
print("\n⏳ Entraînement du modèle AutoML (TPOT)...")
tpot.fit(X_train, y_train)

# Score R²
score = tpot.score(X_test, y_test)
print(f"\n✅ R² score sur le test : {score:.4f}")

# Export du meilleur pipeline trouvé
tpot.export('meilleur_pipeline_cac40.py')
print("📦 Pipeline sauvegardé dans : meilleur_pipeline_cac40.py")
