import numpy as np
from numpy.polynomial import Polynomial

import pandas as pd

# Cette classe fournit plusieurs méthodes d'interpolation et d'extrapolation 
# basées sur un DataFrame contenant une série de données (x, y).

# Méthodes disponibles :

# 1. Spline cubique naturelle :
#    - Fournit une courbe lisse passant par tous les points.
#    - Adaptée aux données continues et régulières.


# 2. Moindres carrés (régression polynomiale) :
#    - Approxime les données avec un polynôme de degré choisi.
#    - Ne passe pas forcément par tous les points, mais suit la tendance globale.


# 3. Interpolation linéaire avec extrapolation :
#    - Simple et robuste.
#    - Interpole entre deux points par des segments droits.
#    - Permet une extrapolation linéaire en utilisant la pente aux extrémités.

class Interpolation:
    """
    Classe pour effectuer des interpolations sur des données contenues dans un DataFrame.
    df : Le DataFrame contenant les données.
    col : Le nom de la colonne contenant les valeurs y.

    """

    def __init__(self, df, col):
        self.df = df.sort_index()
        self.col = col
        self.x = self.df.index.to_numpy()
        self.y = self.df[col].to_numpy()

    def spline_cubique_naturelle(self, x_val=None):
        x_data = self.x
        y = self.y
        n = len(x_data) - 1

        if n < 1:
            raise ValueError("Au moins 2 points sont nécessaires pour l'interpolation.")

        # Étape 1 : Calcul des pas h et des valeurs alpha
        h = [x_data[i + 1] - x_data[i] for i in range(n)]
        alpha = [0.0] * (n + 1)
        for i in range(1, n):
            alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (
                y[i] - y[i - 1]
            )

        # Étape 2 : Résolution du système tridiagonal
        l = [1.0] + [0.0] * n
        mu = [0.0] * (n + 1)
        z = [0.0] * (n + 1)

        for i in range(1, n):
            l[i] = 2 * (x_data[i + 1] - x_data[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        l[n] = 1.0
        z[n] = 0.0

        # Étape 3 : Calcul des coefficients a, b, c, d
        c = [0.0] * (n + 1)
        b = [0.0] * n
        d = [0.0] * n
        a = y[:-1]

        for j in range(n - 1, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (2 * c[j] + c[j + 1]) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])

        # Fonction d'évaluation du spline sur l'intervalle correspondant
        def evaluer_spline(x_eval):
            for i in range(n):
                if x_data[i] <= x_eval <= x_data[i + 1]:
                    dx = x_eval - x_data[i]
                    return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
            if x_eval < x_data[0]:
                i = 0
                dx = x_eval - x_data[i]
                return a[i] + b[i] * dx
            else:
                i = n - 1
                dx = x_eval - x_data[i]
                return a[i] + b[i] * dx

        if x_val is not None:
            return evaluer_spline(x_val)

        return {
            "a": a,
            "b": b,
            "c": c[:-1],
            "d": d,
            "intervalles": list(zip(x_data[:-1], x_data[1:])),
        }
    def ajustement_polynomiale_moindres_carres(self, degre=34, x_val=None):

        x = self.x
        y = self.y

        # Normalisation des données pour la stabilité numérique
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        x_normalise = (x - x_min) / (x_max - x_min)
        y_normalise = (y - y_min) / (y_max - y_min)

        # Ajustement du polynôme sur les données normalisées
        poly_normalise = Polynomial.fit(x_normalise, y_normalise, degre)

        if x_val is not None:
            x_val_normalise = (x_val - x_min) / (x_max - x_min)
            resultat_normalise = poly_normalise(x_val_normalise)
            resultat = resultat_normalise * (y_max - y_min) + y_min
            return resultat

        raise ValueError("Veuillez fournir une valeur x_val pour l'évaluation.")
    def interpolation_lineaire_extrapolation(self, x_val):
        x = self.x
        y = self.y

        if x_val < x[0]:
            # Extrapolation à gauche
            slope = (y[1] - y[0]) / (x[1] - x[0])
            return y[0] + slope * (x_val - x[0])
        elif x_val > x[-1]:
            # Extrapolation à droite
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            return y[-1] + slope * (x_val - x[-1])
        else:
            # Interpolation linéaire classique
            for i in range(len(x) - 1):
                if x[i] <= x_val <= x[i + 1]:
                    slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                    return y[i] + slope * (x_val - x[i])



