import numpy as np
import pandas as pd

#  ------>
# Le phénomène de Runge pour lagrange et newton
# les division de Chebyshev reduire le prb mais en notre cas y'a pas une grande diff (overflow)
# les valeur tres proches => resultat inf en quelque cas et le nombre de points aussi favorise cela
# meilleur methose spline


class Interpolation:
    """
    df: Le DataFrame(csv) contenant les données.
    col: Le nom de la colonne contenant les valeurs y.
    h : le pas entre xi+1 et xi
    """

    def __init__(self, df, col):
        self.df = df
        self.col = col
        self.x = df.index.to_numpy()
        self.y = df[col].to_numpy()

    def lagrange_interpolation(self, x_val, use_chebyshev=False):
        n = len(self.x)
        a, b = self.x[0], self.x[-1]
        result = 0
        if use_chebyshev:
            print("cheb")
            # Génération des nœuds de Chebyshev dans [-1, 1], puis transformation vers [a, b]
            k = np.arange(n)
            theta = (2 * k + 1) * np.pi / (2 * n)
            x_cheb = (a + b) / 2 + (b - a) / 2 * np.cos(theta)

            # Évaluer les valeurs y aux nœuds de Chebyshev en interpolant les données originales
            y_cheb = np.interp(x_cheb, self.x, self.y)  # Interpolation linéaire

            # Remplacer les données originales par les nœuds de Chebyshev
            self.x, self.y = x_cheb, y_cheb

        for i in range(n):
            term = self.y[i]  # valeur f(i)
            for j in range(n):  # Calcul du produit de Lagrange : Π((x - k) / (i - k))
                if j != i:
                    term *= (x_val - self.x[j]) / (self.x[i] - self.x[j])
            result += term  # Ajout de la contribution du terme courant à l'interpolation finale

        return result

    def difference_divisee_newton(self):
        n = len(self.x)  # Utiliser l'index comme les valeurs x
        coefficients = list(self.y)  # Copie des valeurs de y

        for j in range(1, n):
            for i in range(
                n - 1, j - 1, -1
            ):  # Utiliser les valeurs d'index pour x_i et x_(i-j)
                x_i = self.x[i]
                x_i_j = self.x[i - j]
                coefficients[i] = (coefficients[i] - coefficients[i - 1]) / (
                    x_i - x_i_j
                )  # Calcul de la différence divisée de Newton

        return coefficients

    def interpolation_newton(self, x_val):
        n = len(self.x)  # Commencer avec le dernier coefficient
        coeffs = self.difference_divisee_newton()
        result = coeffs[-1]  # Utiliser l'index pour les valeurs x

        for i in range(n - 2, -1, -1):  # Évaluer le polynôme de Newton en x
            result = result * (x_val - self.x[i]) + coeffs[i]

        return result

    def natural_cubic_spline(self, x_val):
        x = self.x
        y = self.y
        n = len(x) - 1

        h = [x[i + 1] - x[i] for i in range(n)]  # Pas entre les points

        alpha = [0] * (n + 1)
        for i in range(1, n):  # Calcul des différences pour la diagonale principale
            alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (
                y[i] - y[i - 1]
            )

        l = [1] + [0] * n  # Diagonale principale de la matrice
        mu = [0] * (n + 1)  # Coefficients intermédiaires
        z = [0] * (n + 1)  # Second membre

        for i in range(1, n):
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        l[n] = 1
        z[n] = 0

        c = [0] * (
            n + 1
        )  # Coefficients c des splines (suite de polynomes)initialisation
        b = [0] * n  # Coefficients b des splines (suite de polynomes)initialisation
        d = [0] * n  # Coefficients d des splines (suite de polynomes)initialisation
        a = y[:-1]  # Coefficients a des splines (suite de polynomes)initialisation

        for j in range(
            n - 1, -1, -1
        ):  # Résolution  pour trouver c, b et d par algorithme de Thomas Matriciel
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (2 * c[j] + c[j + 1]) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])

        for i in range(n):
            if x[i] <= x_val <= x[i + 1]:
                dx = x_val - x[i]
                return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
