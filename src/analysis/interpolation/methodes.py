import numpy as np
import pandas as pd

class Interpolation:
    """
    df: Le DataFrame contenant les données.
    col: Le nom de la colonne contenant les valeurs y.
    """

    def __init__(self, df, col):
        self.df = df.sort_index()
        self.col = col
        self.x = self.df.index.to_numpy()
        self.y = self.df[col].to_numpy()

    def lagrange_interpolation(self, x_val, use_chebyshev=False):
        n = len(self.x)
        a, b = self.x[0], self.x[-1]
        result = 0
        x_data = self.x
        y_data = self.y

        if use_chebyshev:
            k = np.arange(n)
            theta = (2 * k + 1) * np.pi / (2 * n)
            x_cheb = (a + b) / 2 + (b - a) / 2 * np.cos(theta)
            y_cheb = np.interp(x_cheb, self.x, self.y)
            x_data, y_data = x_cheb, y_cheb

        for i in range(n):
            term = y_data[i]
            for j in range(n):
                if j != i:
                    term *= (x_val - x_data[j]) / (x_data[i] - x_data[j])
            result += term

        return result

    def difference_divisee_newton(self):
        n = len(self.x)
        coefficients = list(self.y)

        for j in range(1, n):
            for i in range(n - 1, j - 1, -1):
                x_i = self.x[i]
                x_i_j = self.x[i - j]
                coefficients[i] = (coefficients[i] - coefficients[i - 1]) / (x_i - x_i_j)

        return coefficients

    def interpolation_newton(self, x_val):
        n = len(self.x)
        coeffs = self.difference_divisee_newton()
        result = coeffs[-1]

        for i in range(n - 2, -1, -1):
            result = result * (x_val - self.x[i]) + coeffs[i]

        return result

    def natural_cubic_spline(self, x_val=None):
        x_data = self.x
        y = self.y
        n = len(x_data) - 1

        if n < 1:
            raise ValueError("Au moins 2 points sont nécessaires pour l'interpolation.")

        h = [x_data[i + 1] - x_data[i] for i in range(n)]

        alpha = [0.0] * (n + 1)
        for i in range(1, n):
            alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])

        l = [1.0] + [0.0] * n
        mu = [0.0] * (n + 1)
        z = [0.0] * (n + 1)

        for i in range(1, n):
            l[i] = 2 * (x_data[i + 1] - x_data[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        l[n] = 1.0
        z[n] = 0.0

        c = [0.0] * (n + 1)
        b = [0.0] * n
        d = [0.0] * n
        a = y[:-1]

        for j in range(n - 1, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (2 * c[j] + c[j + 1]) / 3
            d[j] = (c[j + 1] - c[j]) / (3 * h[j])

        def evaluate_spline(x_eval):
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
            return evaluate_spline(x_val)

        return {
            "a": a,
            "b": b,
            "c": c[:-1],
            "d": d,
            "intervals": list(zip(x_data[:-1], x_data[1:]))
        }


