import numpy as np
import pandas as pd
from scipy.integrate import simpson


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
    # Calcul de a et b
    integral_I_abs = simpson(df["I_abs"], dx=1)
    a = df["R_abs"].iloc[-1] / integral_I_abs
    b = df["D_abs"].iloc[-1] / integral_I_abs

    # Calcul de r
    df["dI_dt"] = np.gradient(df["I"], 1)
    valid = (df["S"] * df["I"]) > 1e-9

    r_values = (df["dI_dt"][valid] + (a + b) * df["I"][valid]) / (
        df["S"][valid] * df["I"][valid]
    )
    r = np.nanmedian(r_values)

    return {"r": r, "a": a, "b": b}