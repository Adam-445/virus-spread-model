import numpy as np
import pandas as pd

from src.analysis.integration.methodes import Integration


def estimer_parametres_rab(df: pd.DataFrame) -> dict[str, float]:
    """
    Estime les paramètres épidémiologiques r, a et b à partir des données SIRD.

    Args:
        df: DataFrame contenant les colonnes 'I', 'S', 'I_abs', 'R_abs', 'D_abs'

    Returns:
        Dictionnaire avec les paramètres:
            - r: Taux de contagion
            - a: Taux de guérison
            - b: Taux de mortalité
    """
    # Création d'un DataFrame temporaire pour l'intégration
    df_integration = pd.DataFrame({"I_abs": df["I_abs"].values}, index=df.index)

    # Initialisation de l'intégrateur personnalisé
    integrateur = Integration(df=df_integration, col="I_abs", h=1.0)

    # Calcul de l'intégrale avec notre méthode Simpson
    integral_I_abs = integrateur.simpson()

    # Calcul des paramètres a et b
    a = df["R_abs"].iloc[-1] / integral_I_abs
    b = df["D_abs"].iloc[-1] / integral_I_abs

    # Calcul de r avec validation des valeurs
    df["dI_dt"] = np.gradient(df["I"], 1)
    condition_valide = (df["S"] * df["I"] > 1e-9) & (df["I"] > 1e-6)

    r_values = (df["dI_dt"][condition_valide] + (a + b) * df["I"][condition_valide]) / (
        df["S"][condition_valide] * df["I"][condition_valide]
    )
    r = r_values.median()

    return {"r": float(r), "a": float(a), "b": float(b)}
