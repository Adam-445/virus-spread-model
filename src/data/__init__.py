from pathlib import Path

from .cleaner import DataCleaner
from .fetcher import DataFetcher
from .validator import DataValidator


class DataPipeline:
    """
    Pipeline complet de traitement des données
    Sequence : Téléchargement -> Nettoyage -> Validation
    """

    def __init__(self, population: int, country: str = "France"):
        """
        Args:
            population (int): Population totale du pays cible
            country (str): Code pays ISO à traiter
        """
        self.population = population
        self.country = country
        self._init_paths()

    def _init_paths(self):
        """Initialise les chemins des dossiers"""
        project_root = Path(__file__).resolve().parents[2]
        self.raw_path = project_root / "data/raw"
        self.processed_path = project_root / "data/processed"

    def run(self):
        """Exécute le pipeline complet"""
        try:
            # Étape 1: Téléchargement
            fetcher = DataFetcher(self.raw_path)
            confirmed, deaths, recovered = fetcher.fetch_data()
            
            # Étape 2: Nettoyage
            cleaner = DataCleaner(self.population, self.processed_path)
            df = cleaner.clean_jhu_data(
                confirmed, deaths, recovered,
                country=self.country
            )
            
            # Étape 3: Validation
            validator = DataValidator(
                population=self.population,
                raw_path=self.raw_path,
                processed_path=self.processed_path
            )
            validator.validate_raw_data()
            validator.validate_processed_data(self.country)
            
            return df
            
        except Exception as e:
            print(f"Échec du pipeline: {str(e)}")
            raise