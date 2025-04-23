from typing import Callable

import numpy as np


class SolveurNumerique:
    """Classe contenant des méthodes numériques pour résoudre des équations différentielles."""

    @staticmethod
    def euler(
        fonction_derivee: Callable,
        y0: np.ndarray,
        t_max: float,
        dt: float,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Résout un système d'EDO avec la méthode d'Euler.

        Args:
            fonction_derivee: Fonction calculant les dérivées (dy/dt = f(y, t))
            y0: Vecteur d'état initial
            t_max: Temps final de simulation
            dt: Pas de temps

        Returns:
            Tuple: (temps, états)
        """
        # Nombre total d'itérations
        n_steps = int(t_max / dt)

        # Grille temporelle
        t = np.linspace(0, t_max, n_steps + 1)

        # Historique des états
        y = [y0]

        for _ in range(n_steps):
            # Calcul de la dérivée à l'instant t
            dy = fonction_derivee(y[-1], t[_])
            # Formule d'Euler explicite
            y_new = y[-1] + dt * dy
            # Empêche les valeurs négatives
            y.append(np.maximum(y_new, 0))

        return t, np.array(y)

    @staticmethod
    def rk4(
        fonction_derivee: Callable[[np.ndarray, float], np.ndarray],
        y0: np.ndarray,
        t_max: float,
        dt: float,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Résout un système d'EDO avec la méthode de Runge-Kutta d'ordre 4.

        Args/Voir méthode Euler pour les paramètres
        """
        n_steps = int(t_max / dt)
        t = np.linspace(0, t_max, n_steps + 1)
        y = [y0]

        for _ in range(n_steps):
            # Calcul des 4 coefficients caractéristiques de RK4
            k1 = fonction_derivee(y[-1], t[_]) # Pente au début de l'intervalle
            k2 = fonction_derivee(y[-1] + dt / 2 * k1, t[_] + dt / 2) # Pente au milieu (utilisant k1)
            k3 = fonction_derivee(y[-1] + dt / 2 * k2, t[_] + dt / 2) # Pente au milieu (utilisant k2)
            k4 = fonction_derivee(y[-1] + dt * k3, t[_] + dt) # Pente à la fin de l'intervalle

            # Combinaison pondérée des coefficients
            y_new = y[-1] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            # Maintien des valeurs positives
            y.append(np.maximum(y_new, 0))

        return t, np.array(y)
