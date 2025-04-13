import json
from pathlib import Path

import pandas as pd


class DataValidator:
    """
    Classe de validation de l'intégrité des données brutes et traitées

    Attributes:
        population (int): Population totale de référence
        raw_path (Path): Chemin des données brutes
        processed_path (Path): Chemin des données traitées
    """

    def __init__(
        self, population: int, raw_path: Path = None, processed_path: Path = None
    ):
        """Initialise les chemins de validation"""
        project_root = Path(__file__).resolve().parents[2]

        # Définition des chemins par défaut
        self.raw_path = Path(raw_path) if raw_path else project_root / "data/raw"
        self.processed_path = (
            Path(processed_path) if processed_path else project_root / "data/processed"
        )
        self.population = population

    def validate_raw_data(self):
        """
        Vérifie la présence des fichiers sources essentiels
        Lève FileNotFoundError si un fichier manque
        """
        required_files = [
            "time_series_covid19_confirmed_global.csv",
            "time_series_covid19_deaths_global.csv",
            "time_series_covid19_recovered_global.csv",
        ]

        # Vérification de l'existence de chaque fichier
        for f in required_files:
            full_path = (
                self.raw_path
                / "COVID-19-master/csse_covid_19_data/csse_covid_19_time_series"
                / f
            )
            if not full_path.exists():
                raise FileNotFoundError(f"Fichier source manquant: {full_path}")

    def validate_processed_data(self, country: str = "France"):
        """
        Valide la cohérence des données transformées
        Args:
            country (str): Code pays pour le fichier à valider
        """
        # Chargement des données
        file_path = self.processed_path / f"sird_{country.lower()}.csv"
        df = pd.read_csv(file_path)
        
        # Vérification des valeurs négatives
        for col in ['S', 'I', 'R', 'D']:
            if (df[col] < 0).any():
                raise ValueError(f"Valeurs négatives détectées dans {col}")

        # Vérification de la cohérence démographique
        total = df[['S', 'I', 'R', 'D']].sum(axis=1)

        if (total > self.population).any():
            raise ValueError("Somme S+I+R+D dépasse la population totale")

        # Enregistrement des métadonnées
        self._save_metadata(df, country)
    
    def _save_metadata(self, df: pd.DataFrame, country: str):
        """Sauvegarde les métadonnées techniques du traitement"""
        metadata = {
            "pays": country,
            "derniere_maj": pd.Timestamp.now().isoformat(),
            "nombre_jours": len(df),
            "plage_dates": {
                "debut": df['date'].min(),
                "fin": df['date'].max()
            },
            "population_reference": self.population
        }

        with open(self.processed_path / f"metadata_{country}.json", 'w') as f:
            json.dump(metadata, f, indent=4)
