
import numpy as np
import pandas as pd


# data sous form csv
# l'ajout d un point car simpson necessite un nombre impaire de points

class Integration:
    """
        df: Le DataFrame(csv) contenant les données.
        col: Le nom de la colonne contenant les valeurs y.
        h : le pas entre xi+1 et xi
    """
    def __init__(self, df, col, h):
        self.df = df
        self.col = col
        self.h = h
        self.n = len(df)

    def trapeze(self):
        premiere_valeur = self.df.loc[0, self.col]    # Valeur au début de l’intervalle
        derniere_valeur = self.df.loc[self.n - 1, self.col]  # Valeur à la fin de l’intervalle
        somme_trapeze = 0
        # Somme des valeurs internes
        for i in range(1, self.n - 1):
            somme_trapeze += self.df.loc[self.df.index[i], self.col]
        # Application de la formule du trapèze
        integrale_trapeze = (self.h / 2) * (premiere_valeur + derniere_valeur + 2 * somme_trapeze)
        return integrale_trapeze

    def simpson(self):
        fist_value = self.df.loc[self.df.index[0], self.col]
        last_value = self.df.loc[self.df.index[-1], self.col]
        if self.n % 2 == 0:
            added_row =self.df.iloc[[-1]].copy()
            self.df = pd.concat([self.df.iloc[:877],added_row,self.df.iloc[877:]],ignore_index=True)
        somme_simpson = 0
        # Test pour coeff et somme
        for i in range(1, self.n - 1):
            coeff = 4 if i % 2 != 0 else 2
            somme_simpson += coeff * self.df.loc[self.df.index[i], self.col]



        simpson_integrale = (self.h / 3) * (fist_value + last_value + somme_simpson)
        return simpson_integrale

    def rect_gauche(self):
        somme_rectangle_gauche = 0
        # Somme des hauteurs des rectangles à gauche
        for i in range(0, self.n - 1):
            somme_rectangle_gauche += self.df.loc[self.df.index[i], self.col]
        # Calcul de l’intégrale par la méthode des rectangles à gauche
        return self.h * somme_rectangle_gauche

    def rect_droite(self):
        # Somme des hauteurs des rectangles à droite
        somme_rectangle_droite = 0
        for i in range(1, self.n):
            somme_rectangle_droite += self.df.loc[self.df.index[i], self.col]
        # Calcul de l’intégrale par la méthode des rectangles à droite
        return self.h * somme_rectangle_droite


