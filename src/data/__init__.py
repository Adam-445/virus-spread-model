from pathlib import Path

import pandas as pd

from .cleaner import DataCleaner
from .fetcher import DataFetcher
from .validator import DataValidator


class DataPipeline:
    """
    Pipeline complet de gestion des données épidémiologiques.

    Workflow:
    1. Téléchargement des données brutes (DataFetcher)
    2. Nettoyage et transformation SIRD (DataCleaner)
    3. Validation de qualité (DataValidator)
    4. Sortie des données prêtes pour l'analyse

    Exemple:
    >>> pipeline = DataPipeline(country="brazil")
    >>> data = pipeline.run(start_date="2020-03-01", end_date="2022-01-01")
    """

    def __init__(self, country: str = "France"):
        """
        Initialise le pipeline pour un pays spécifique.

        Args:
            country: Nom du pays ou code ISO3 (validation automatique)
        """
        self.country = country.strip().lower()
        self._init_paths()

    def _init_paths(self):
        """Configure les chemins de données avec création des répertoires"""
        self.project_root = Path(__file__).resolve().parents[2]

        # Configuration des chemins
        self.raw_path = self.project_root / "data/raw"
        self.processed_path = self.project_root / "data/processed" / self.country

        # Création des répertoires si nécessaire
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        start_date: str = None,
        end_date: str = None,
        split: str = "train",
        tolerance: float = 0.01,
        smoothing: bool = True,
        window_size: int = 7,
    ) -> pd.DataFrame:
        """
        Exécute le pipeline complet de traitement des données.

        Args:
            start_date: Date de début au format YYYY-MM-DD
            end_date: Date de fin au format YYYY-MM-DD
            tolerance: Tolérance pour la validation des données

        Returns:
            DataFrame d'entraînement validé

        Raises:
            RuntimeError: Si une étape du pipeline échoue
        """
        try:
            # Étape 1: Acquisition des données
            raw_data = DataFetcher(self.raw_path).fetch_data()

            # Étape 2: Nettoyage et transformation
            cleaner = DataCleaner(
                processed_path=self.processed_path,
                country=self.country,
                smoothing=smoothing,
                window_size=window_size,
            )
            cleaned_data = cleaner.clean_and_save(
                raw_data, start_date=start_date, end_date=end_date
            )
            self.population = cleaner.population

            # Étape 3: Validation de qualité
            validator = DataValidator(
                country=self.country,
                processed_path=self.processed_path,
                tolerance=tolerance,
            )
            validation_report = validator.validate()

            return validation_report[split]

        except Exception as e:
            raise RuntimeError(
                f"Échec du pipeline pour {self.country}: {str(e)}"
            ) from e
