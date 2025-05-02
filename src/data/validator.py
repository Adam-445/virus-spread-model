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
        """
        Initialise le validateur pour un pays spécifique.

        Args:
            country: Pays cible (format insensible à la casse)
            processed_path: Chemin personnalisé pour les données nettoyées
            tolerance: Écart maximal autorisé pour S+I+R+D autour de 1
        """
        self.country = country.lower()
        self.tolerance = tolerance

        # Configuration des chemins
        self.processed_path = (
            processed_path
            or Path(__file__).resolve().parents[2] / "data/processed" / self.country
        )

        # Validation initiale du dossier
        if not self.processed_path.is_dir():
            raise FileNotFoundError(
                f"Dossier de données introuvable: {self.processed_path}"
            )

    def validate(self) -> dict[str, pd.DataFrame]:
        """
        Exécute le pipeline complet de validation.

        Returns:
            Résultats avec métadonnées :
            - "train": DataFrame d'entraînement validé
            - "test": DataFrame de test validé
            - "metadata": Statistiques de validation

        Raises:
            ValueError: Si une validation échoue
        """
        # Chargement des données
        train_df = self._load_and_validate_split("train")
        test_df = self._load_and_validate_split("test")

        metadata = self._generate_metadata(pd.concat([train_df, test_df]))
        return {"train": train_df, "test": test_df, "metadata": metadata}

    def _load_and_validate_split(self, split_type: str) -> pd.DataFrame:
        """Pipeline de validation pour un ensemble (train/test)"""
        file_path = self.processed_path / f"sird_{self.country}_{split_type}.csv"

        # Chargement avec vérification d'existence
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier {split_type} manquant: {file_path}")

        df = pd.read_csv(file_path, parse_dates=["date"])

        # Contrôles de qualité
        required = ["date", "S", "I", "R", "D"]
        if not set(required).issubset(df.columns):
            raise ValueError(
                f"Colonnes manquantes dans {split_type} : {set(required) - set(df.columns)}"
            )

        if df.isnull().values.any():
            raise ValueError(f"Valeurs manquantes détectées dans {split_type}")

        for col in ["S", "I", "R", "D"]:
            if not df[col].between(0, 1).all():
                raise ValueError(f"Valeurs hors de [0,1] dans {col} ({split_type})")

        total = df[["S", "I", "R", "D"]].sum(axis=1)
        bad_rows = (total - 1).abs() > self.tolerance
        if bad_rows.any():
            print(f"{bad_rows.sum()} lignes invalides dans {split_type} (S+I+R+D ≠ 1)")

        return df

    def _generate_metadata(self, df: pd.DataFrame) -> dict:
        """Génère un rapport de qualité des données"""
        metadata = {
            "pays": self.country,
            "periode_jours": len(df),
            "date_min": df["date"].min().isoformat(),
            "date_max": df["date"].max().isoformat(),
            "metriques": {
                col: {
                    "moyenne": df[col].mean(),
                    "max": df[col].max(),
                    "min": df[col].min(),
                    "nan_count": df[col].isna().sum(),
                }
                for col in ["S", "I", "R", "D"]
            },
            "validation": {
                "tolerance": self.tolerance,
                "date_validation": pd.Timestamp.now().isoformat(),
                "status": "SUCCES" if not df.empty else "ECHEC",
            },
        }

        # Sauvegarde du rapport
        metadata_path = self.processed_path / f"metadata_{self.country}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        return metadata
