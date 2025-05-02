import numpy as np
import pandas as pd

from src.data import DataPipeline

from .solveur import SolveurNumerique


class SimulateurSIRD:
    """
    Classe pour simuler la dynamique d'une épidémie avec le modèle SIRD.

    Exemple d'utilisation:
    >>> simulateur = SimulateurSIRD({
    ...     "r": 0.35,
    ...     "a": 0.1,
    ...     "b": 0.02
    ... })
    >>> # Simulation sur 1 an avec pas quotidien
    >>> resultats = simulateur.resoudre(
    ...     df=donnees,
    ...     t_max=365,
    ...     dt=1.0,
    ...     methode="rk4"
    ... )
    """

    def __init__(self, parametres: dict[str, float]):
        """
        Initialise le modèle avec les paramètres épidémiologiques.
        Les paramètres sont stockés comme attributs de classe pour être accessibles
        dans toutes les méthodes.

        Args:
            parametres: Dictionnaire des paramètres contenant:
                - r: Taux de contagion (0 < r <= 1)
                - a: Taux de guérison (a > 0)
                - b: Taux de mortalité (b > 0)
        """
        self.r = parametres["r"]
        self.a = parametres["a"]
        self.b = parametres["b"]
        self._valider_parametres()

    def _valider_parametres(self) -> None:
        """Validation des contraintes sur les paramètres."""
        # Les taux ne peuvent pas être négatifs
        if any(val < 0 for val in [self.r, self.a, self.b]):
            raise ValueError("Tous les paramètres doivent être positifs")

        # Le taux de contagion est normalisé entre 0 et 1
        if not 0 < self.r <= 1:
            raise ValueError("r doit être dans ]0, 1]")

        # Calcul du nombre de reproduction de base (seuil épidémique)
        self.R0 = self.r / (self.a + self.b)
        if self.R0 < 1:
            raise ValueError(f"R0={self.R0:.2f} < 1 → Pas d'épidémie")

    def _modele_sird(self, etat: np.ndarray, t: float) -> np.ndarray:
        """
        Implémentation vectorielle des équations différentielles du modèle SIRD.
        Utilise la notation vectorielle pour faciliter l'intégration avec les solveurs numériques.

        Args:
            etat: Vecteur d'état [S, I, R, D]
            t: Temps (non utilisé mais requis par le solveur)

        Returns:
            Vecteur des dérivées [dS/dt, dI/dt, dR/dt, dD/dt]
        """
        # On déstructure le vecteur d'état
        S, I, _, _ = etat

        # Equations différentielles (version vectorielle)
        dS = -self.r * S * I  # Diminution des sains
        dI = self.r * S * I - (self.a + self.b) * I  # Variation des infectés
        dR = self.a * I  # Augmentation des guéris
        dD = self.b * I  # Augmentation des décédés
        return np.array([dS, dI, dR, dD])

    def resoudre(
        self, df: pd.DataFrame, t_max: int, dt: float = 1.0, methode: str = "rk4"
    ) -> pd.DataFrame:
        """
        Résout le système d'équations différentielles.
        Combine les conditions initiales avec la méthode numérique choisie.

        Args:
            df: DataFrame contenant les conditions initiales
            t_max: Durée de simulation (jours)
            dt: Pas de temps
            methode: 'euler' ou 'rk4'

        Returns:
            DataFrame avec les résultats de simulation
        """
        # Extraction des conditions initiales depuis le DataFrame
        # Note: Les colonnes doivent correspondre à ['S', 'I', 'R', 'D']
        y0 = np.array(
            [
                df["S"].iloc[0],  # Population saine initiale
                max(df["I"].iloc[0], 1e-5),  # Infectés initiaux
                df["R"].iloc[0],  # Guéris initiaux
                df["D"].iloc[0],  # Décédés initiaux
            ]
        )

        # Sélection de la méthode numérique (abstraction via SolveurNumerique)
        if methode == "euler":
            t, y = SolveurNumerique.euler(self._modele_sird, y0, t_max, dt)
        elif methode == "rk4":
            t, y = SolveurNumerique.rk4(self._modele_sird, y0, t_max, dt)
        else:
            raise ValueError(f"Méthode {methode} non supportée")

        return self._creer_dataframe(t, y)

    def _creer_dataframe(self, t: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Formatage des résultats en DataFrame pour faciliter l'analyse et la visualisation."""
        return pd.DataFrame(
            {"temps": t, "S": y[:, 0], "I": y[:, 1], "R": y[:, 2], "D": y[:, 3]}
        )

    def comparer_avec_donnees_reelles(
        self, pays: str, methode: str = "rk4"
    ) -> pd.DataFrame:
        """
        Méthode d'analyse comparative qui calcule trois métriques d'erreur :
        1. Erreur absolue (différence brute)
        2. Erreur relative (pourcentage d'erreur)

        Args:
            pays: Nom du pays pour les données réelles
            methode: Méthode de résolution numérique

        Returns:
            DataFrame avec comparaison et métriques d'erreur
        """
        # Chargement des données réelles depuis le pipeline
        donnees = DataPipeline(country=pays).run()

        # Simulation sur la même période que les données réelles
        simulation = self.resoudre(donnees, len(donnees), methode=methode)

        # Calcul des métriques d'erreur
        simulation["I_reel"] = donnees["I"].values  # Ajout des données réelles
        simulation["Erreur_absolue"] = abs(simulation["I"] - simulation["I_reel"])

        # Gestion des divisions par zéro pour l'erreur relative
        simulation["Erreur_relative"] = simulation["Erreur_absolue"] / simulation[
            "I_reel"
        ].replace(0, 1e-9)

        return simulation
