import pandas as pd
import numpy as np


class ResolutionSIRD:
    def __init__(self, df: pd.DataFrame, parametres: dict[str, float]):
        """
        Initialisation du modèle SIRD avec des paramètres épidémiques.

        Args:
            df: DataFrame avec colonnes 'S', 'I', 'R', 'D' indexé par 'Jour'
            parametres: Dictionnaire avec clés 'r', 'a', 'b'
        """
        columns_necessaires = ['S', 'I', 'R', 'D']
        if not all(col in df.columns for col in columns_necessaires):
            raise ValueError(f"DataFrame doit contenir les colonnes : {columns_necessaires}")
        if "lits_par_mille" not in df.columns:
            raise ValueError("Colonne 'lits_par_mille' manquante pour calculer Imax")

        self.r = parametres["r"]
        self.a = parametres["a"]
        self.b = parametres["b"]
        self.Imax = df["lits_par_mille"].iloc[0] / 1000
        self._valider_parametres()
        self.df = df.copy()

    def _valider_parametres(self):
        """Validation des contraintes sur les paramètres."""
        if any(val < 0 for val in [self.r, self.a, self.b]):
            raise ValueError("Tous les paramètres doivent être positifs")
        if not 0 < self.r <= 1:
            raise ValueError("r doit être dans ]0, 1]")
        self.R0 = self.r / (self.a + self.b)
        if self.R0 < 1:
            raise ValueError(f"R0={self.R0:.2f} < 1 → Pas d'épidémie")

    def index_pic_epidimique(self):
        """
        Trouve l'indice du pic épidémique en détectant le changement de signe de dI/dt.
        """
        t = self.df.index.values
        S = self.df["S"].values
        I = self.df["I"].values
        dI_dt = self.r * I * S - (self.a + self.b) * I
        changements_signe = np.where(np.diff(np.sign(dI_dt)) < 0)[0]
        if len(changements_signe) == 0:
            return None
        return t[changements_signe[0] + 1]  # +1 car diff réduit la longueur

    def seuil_immunite(self):
        """Calcule le seuil d'immunité collective au moment du pic épidémique."""
        pic_index = self.index_pic_epidimique()
        if pic_index is None:
            return None
        return 1 - self.df.loc[pic_index, 'S']

    def temps_critique(self):
        """Trouve le premier indice où I dépasse la capacité hospitalière Imax."""
        I = self.df["I"].values
        for i, val in enumerate(I):
            if val > self.Imax:
                return i
        return None  # Aucun dépassement
        #nombre de reproduction
    def reproduction(self):
        r=self.r
        a=self.a
        b=self.b
        return r/(a+b)
