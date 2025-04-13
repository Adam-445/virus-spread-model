import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


class DataFetcher:
    """
    Classe pour le téléchargement et l'extraction des données brutes COVID-19 depuis JHU
    Attributes:
        raw_path (Path): Chemin vers le dossier de stockage des données brutes
        url (str): URL de la source des données
        zip_path (Path): Chemin complet du fichier zip téléchargé
        extracted_path (Path): Chemin d'extraction des données
    """

    def __init__(self, raw_path: Path = None):
        """Initialise les chemins des données brutes"""
        # Défaut: <racine_projet>/data/raw
        if not raw_path:
            project_root = Path(__file__).resolve().parents[2]
            raw_path = project_root / "data/raw"

        self.raw_path = Path(raw_path)

        # Crée le dossier si inexistant
        self.raw_path.mkdir(parents=True, exist_ok=True)

        self.url = (
            "https://github.com/CSSEGISandData/COVID-19/archive/refs/heads/master.zip"
        )
        self.zip_path = self.raw_path / "covid19.zip"
        self.extracted_path = (
            self.raw_path
            / "COVID-19-master/csse_covid_19_data/csse_covid_19_time_series"
        )

    def fetch_data(self):
        """
        Orchestre le processus complet :
        1. Téléchargement du zip (si absent)
        2. Extraction des fichiers (si absents)
        3. Chargement des CSV dans des DataFrames
        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (confirmed, deaths, recovered)
        """
        if not self.zip_path.exists():
            self._download()

        if not self.extracted_path.exists():
            self._extract()

        return self._load_csvs()

    def _download(self):
        """Télécharge le fichier zip avec barre de progression"""
        try:
            # Configuration de la requête HTTP en mode stream
            response = requests.get(self.url, stream=True, timeout=10)

            # Lève une exception pour les codes 4xx/5xx
            response.raise_for_status()

            # Récupération de la taille totale pour la barre de progression
            total = int(response.headers.get("content-length", 0))

            # Écriture par blocs avec mise à jour de la barre de progression
            with open(self.zip_path, "wb") as f, tqdm(
                desc="Téléchargement données JHU",
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    # Filtre les keep-alive chunks vides
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

        except requests.exceptions.RequestException as e:
            print(f"Échec du téléchargement: {e}")
            raise

    def _extract(self):
        """Extrait les fichiers du zip téléchargé"""
        print("Extraction des archives...")
        with zipfile.ZipFile(self.zip_path) as z:
            # Extraction dans le dossier raw
            z.extractall(self.raw_path)

    def _load_csvs(self):
        """Charge les données CSV extraites dans des DataFrames Pandas"""
        confirmed = pd.read_csv(
            self.extracted_path / "time_series_covid19_confirmed_global.csv"
        )
        deaths = pd.read_csv(
            self.extracted_path / "time_series_covid19_deaths_global.csv"
        )
        recovered = pd.read_csv(
            self.extracted_path / "time_series_covid19_recovered_global.csv"
        )
        return confirmed, deaths, recovered


# Exemple d'utilisation
if __name__ == "__main__":
    fetcher = DataFetcher()
    confirmed, deaths, recovered = fetcher.fetch_data()
    print("Aperçu des données confirmées:\n", confirmed.head())
