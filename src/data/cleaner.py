from pathlib import Path

import pandas as pd


class DataCleaner:
    """
    Classe pour le nettoyage et la transformation des données COVID-19 en format SIRD

    Attributes:
        population (int): Population totale du pays cible
        processed_path (Path): Chemin de sauvegarde des données traitées
    """

    def __init__(self, population: int, processed_path: Path = None):
        """Initialise les paramètres de nettoyage"""
        # Défaut: <racine_projet>/data/processed
        if not processed_path:
            project_root = Path(__file__).resolve().parents[2]
            processed_path = project_root / "data/processed"

        self.processed_path = Path(processed_path)
        self.processed_path.mkdir(parents=True, exist_ok=True)

        # Population pour le calcul des Susceptibles
        self.population = population

    def clean_jhu_data(
        self,
        confirmed: pd.DataFrame,
        deaths: pd.DataFrame,
        recovered: pd.DataFrame,
        country: str = "France",
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Transforme les données brutes JHU en format SIRD standardisé

        Args:
            confirmed (pd.DataFrame): Données des cas confirmés
            deaths (pd.DataFrame): Données des décès
            recovered (pd.DataFrame): Données des guérisons
            country (str): Pays à traiter
            save (bool): Sauvegarder le résultat si True

        Returns:
            DataFrame: Données au format SIRD avec colonnes [date, S, I, R, D]
        """
        # Agrégation des données pour le pays spécifié
        country_data = {
            "I": confirmed.loc[confirmed["Country/Region"] == country]
            .iloc[:, 4:]
            .sum(),
            "D": deaths.loc[deaths["Country/Region"] == country].iloc[:, 4:].sum(),
            "R": recovered.loc[recovered["Country/Region"] == country]
            .iloc[:, 4:]
            .sum(),
        }

        # Création du DataFrame et traitement des dates
        df = pd.DataFrame(country_data)
        df["date"] = pd.to_datetime(df.index, format="%m/%d/%y", errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        # Les valeurs de décès et de guérisons avant avril 2020 restent nulles (plateau à 0),
        # ce qui fausse les dynamiques SIRD. On commence donc à partir du 1er avril 2020.
        start_date = pd.to_datetime("2020-04-01")
        df = df[df["date"] >= start_date]

        # Les données de guérisons s'arrêtent brutalement à une certaine date,
        # donc on tronque aussi les données à la dernière date où une guérison est renseignée.
        valid_r = df[df["R"] > 0]
        if not valid_r.empty:
            last_valid_date = valid_r["date"].max()
            df = df[df["date"] <= last_valid_date]

        # Correction des anomalies
        df = self._filter_anomalies(df)

        # Calcul des Susceptibles (Population - Infectés - Guéris - Décédés)
        df["S"] = self.population - df[["I", "R", "D"]].sum(axis=1)

        # Reinitialisation de l'index
        df = df.reset_index(drop=True)

        # Sauvegarde automatique si activé
        if save:
            self._save(df, f"sird_{country.lower()}.csv")

        return df

    def _filter_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corrige les anomalies communes dans les données épidémiques
        1. Élimination des valeurs négatives
        2. Correction des diminutions illogiques (les compteurs ne doivent pas diminuer)
        """
        # Suppression des valeurs négatives par seuillage
        for col in ["I", "R", "D"]:
            df[col] = df[col].clip(lower=0)

        # Application d'un cumul maximum pour éviter les diminutions
        for col in ["I", "R", "D"]:
            df[col] = df[col].cummax()

        return df

    def _save(self, df: pd.DataFrame, filename: str):
        """Sauvegarde le DataFrame nettoyé au format CSV"""
        path = self.processed_path / filename
        df.to_csv(path, index=False)
