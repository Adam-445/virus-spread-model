from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


class DataFetcher:
    """
    Classe responsable du téléchargement et du chargement des données brutes COVID-19

    Attributes:
        raw_path (Path): Répertoire de stockage des données brutes
        url (str): URL source du dataset
        file_path (Path): Chemin complet vers le fichier CSV téléchargé
    """

    def __init__(self, raw_path: Path):
        # Par défaut : <project_root>/data/raw
        self.raw_path = raw_path or Path(__file__).resolve().parents[2] / "data/raw"
        self.raw_path.mkdir(parents=True, exist_ok=True)

        self.url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
        self.file_path = self.raw_path / "owid-covid-data.csv"

    def fetch_data(self) -> pd.DataFrame:
        """
        Télécharge (si nécessaire) et charge le dataset complet

        Returns:
            pd.DataFrame: DataFrame contenant toutes les données mondiales

        Raises:
            ConnectionError: Si le téléchargement échoue
            FileNotFoundError: Si le fichier local est introuvable après téléchargement
            pd.errors.ParserError: Si le parsing CSV échoue
        """
        try:
            if not self.file_path.exists():
                self._download_dataset()

            # Chargement avec vérification des dates
            return pd.read_csv(
                self.file_path,
                parse_dates=["date"],
            )

        except pd.errors.ParserError as e:
            raise pd.errors.ParserError(
                f"Erreur de parsing du fichier {self.file_path}: {str(e)}"
            ) from e

    def _download_dataset(self):
        """
        Télécharge le dataset avec gestion des erreurs et barre de progression

        Raises:
            ConnectionError: Pour les erreurs réseau
            requests.HTTPError: Pour les réponses HTTP non valides
        """
        try:
            # Configuration de la requête avec timeout
            response = requests.get(
                self.url,
                stream=True,
            )
            response.raise_for_status()

            # Sauvegarde avec barre de progression
            self._save_with_progress(response)

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Erreur de connexion: {str(e)}") from e

    def _save_with_progress(self, response: requests.Response):
        """
        Sauvegarde le contenu de la réponse avec barre de progression

        Args:
            response: Objet Response de requests

        Returns:
            None: Écrit le fichier sur le disque
        """
        total_size = int(response.headers.get("content-length", 0))

        with open(self.file_path, "wb") as f, tqdm(
            desc=f"Téléchargement {self.file_path.name}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=16 * 1024):  # 16KB chunks
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
