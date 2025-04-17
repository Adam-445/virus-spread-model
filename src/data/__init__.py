from pathlib import Path
from typing import Optional, Tuple, Dict

from .fetcher import DataFetcher
from .cleaner import DataCleaner
from .validator import DataValidator


class DataPipeline:
    """
    Pipeline simple pour charger, nettoyer et valider les données COVID-19 SIRD
    """

    def __init__(self, country: str = "France"):
        self.country = country.lower()
        self.project_root = Path(__file__).resolve().parents[2]
        self.raw_path = self.project_root / "data/raw"
        self.processed_path = self.project_root / "data/processed" / self.country

        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        start_date: Optional[str] = "2020-01-01",
        end_date: Optional[str] = "2023-12-31",
        tolerance: float = 0.01,
    ) -> Tuple[Dict, Dict]:
        """
        Exécute le pipeline : téléchargement, nettoyage, validation

        Returns:
            Tuple[Dict, Dict]: (datasets, metadata)
        """
        # Étape 1: Télécharger les données globales
        fetcher = DataFetcher(self.raw_path)
        global_df = fetcher.fetch_data()

        # Étape 2: Nettoyer les données pour le pays
        cleaner = DataCleaner(country=self.country, processed_path=self.processed_path)
        train_df = cleaner.clean_and_save(
            global_df=global_df, start_date=start_date, end_date=end_date
        )["train"]

        # Étape 3: Valider les données
        validator = DataValidator(
            country=self.country, processed_path=self.processed_path, tolerance=tolerance
        )
        validation = validator.validate()

        return train_df
