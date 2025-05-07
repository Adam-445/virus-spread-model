import numpy as np
import pandas as pd


class Derivation:
    def __init__(self, df, col, h):
        """
        df: Le DataFrame(csv) contenant les données.
        col: Le nom de la colonne contenant les valeurs y.
        h : le pas entre xi+1 et xi
        """
        self.df = df
        self.col = col
        self.x = df.index.to_numpy()
        self.y = df[col].to_numpy()
        self.h = h
        self.n = len(df)

    def premier_derivation_5point(self):
        fx = self.y
        n = self.n
        dfx = np.zeros(n)  # List vide pour la derivée première

        for i in range(2, n - 2):  ## derivation formule pour 5points centre
            dfx[i] = (-fx[i + 2] + 8 * fx[i + 1] - 8 * fx[i - 1] + fx[i - 2]) / (
                12 * self.h
            )

        # le probleme pour les bornes ou il n y a pas 5point autour de xi
        dfx[0] = (fx[1] - fx[0]) / self.h
        dfx[1] = (fx[2] - fx[0]) / (2 * self.h)
        dfx[-2] = (fx[-1] - fx[-3]) / (2 * self.h)
        dfx[-1] = (fx[-1] - fx[-2]) / self.h

        return dfx

    def second_derivation_5point(self):
        # List vide pour la derivée seconde
        # derivation formule pour 5points  centre
        fx = self.y
        n = self.n
        d2fx = np.zeros(n)

        for i in range(2, n - 2):
            d2fx[i] = (
                -fx[i + 2] + 16 * fx[i + 1] - 30 * fx[i] + 16 * fx[i - 1] - fx[i - 2]
            ) / (12 * self.h**2)

        # le probleme pour les bornes ou il n y a pas 5point autour de xi
        d2fx[0] = (fx[2] - 2 * fx[1] + fx[0]) / self.h**2
        d2fx[1] = (fx[2] - 2 * fx[1] + fx[0]) / self.h**2
        d2fx[-2] = (fx[-1] - 2 * fx[-2] + fx[-3]) / self.h**2
        d2fx[-1] = (fx[-1] - 2 * fx[-2] + fx[-3]) / self.h**2

        return d2fx
