import json
from pathlib import Path

import pandas as pd


class DataValidator:
    """
    Valide les fichiers CSV SIRD (train/test) pour un pays donné.

    Vérifie que :
    - Les colonnes S, I, R, D existent
    - Toutes les valeurs sont dans l’intervalle [0, 1]
    - Il n’y a pas de valeurs manquantes
    - La somme S + I + R + D ≈ 1 (tolérance configurable)
    - Génère un fichier de métadonnées .json
    """

    def __init__(
        self, country: str, processed_path: Path = None, tolerance: float = 0.01
    ):
        self.country = country.lower()
        self.tolerance = tolerance
        self.processed_path = (
            processed_path
            or Path(__file__).resolve().parents[2] / "data/processed" / self.country
        )

        if not self.processed_path.exists():
            raise FileNotFoundError(f"Dossier introuvable : {self.processed_path}")

    def validate(self) -> dict[str, pd.DataFrame]:
        """
        Valide les fichiers train/test et génère les métadonnées.

        Returns:
            dict : { "train": DataFrame, "test": DataFrame, "metadata": dict }
        """
        train_df = self._load(self._file_path("train"))
        test_df = self._load(self._file_path("test"))

        self._validate_df(train_df, "train")
        self._validate_df(test_df, "test")

        metadata = self._generate_metadata(pd.concat([train_df, test_df]))
        return {"train": train_df, "test": test_df, "metadata": metadata}

    def _file_path(self, split: str) -> Path:
        """Construit le chemin du fichier train/test"""
        return self.processed_path / f"sird_{self.country}_{split}.csv"

    def _load(self, path: Path) -> pd.DataFrame:
        """Charge un fichier CSV avec parsing de la date"""
        if not path.exists():
            raise FileNotFoundError(f"Fichier manquant : {path}")
        return pd.read_csv(path, parse_dates=["date"])

    def _validate_df(self, df: pd.DataFrame, label: str):
        """Applique toutes les règles de validation à un DataFrame"""
        required = ["date", "S", "I", "R", "D"]
        if not set(required).issubset(df.columns):
            raise ValueError(
                f"Colonnes manquantes dans {label} : {set(required) - set(df.columns)}"
            )

        if df.isnull().values.any():
            raise ValueError(f"Valeurs manquantes détectées dans {label}")

        for col in ["S", "I", "R", "D"]:
            if not df[col].between(0, 1).all():
                raise ValueError(f"Valeurs hors de [0,1] dans {col} ({label})")

        total = df[["S", "I", "R", "D"]].sum(axis=1)
        bad_rows = (total - 1).abs() > self.tolerance
        if bad_rows.any():
            print(f"{bad_rows.sum()} lignes invalides dans {label} (S+I+R+D ≠ 1)")

    def _generate_metadata(self, df: pd.DataFrame) -> dict:
        """Crée et sauvegarde un résumé des données"""
        metadata = {
            "pays": self.country,
            "nombre_total_jours": len(df),
            "plage_dates": {
                "debut": df["date"].min().isoformat(),
                "fin": df["date"].max().isoformat(),
            },
            "bornes_SIRD": {
                col: {"min": float(df[col].min()), "max": float(df[col].max())}
                for col in ["S", "I", "R", "D"]
            },
            "derniere_validation": pd.Timestamp.now().isoformat(),
        }

        # Sauvegarde du fichier metadata
        metadata_path = self.processed_path / f"metadata_{self.country}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return metadata
