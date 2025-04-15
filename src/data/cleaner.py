from pathlib import Path

import pandas as pd


class DataCleaner:
    """
    Classe pour le nettoyage et la transformation des données COVID-19 en format SIRD (Susceptible-Infecté-Rétabli-Décédé)

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
        split: bool = True,
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
        df = self._aggregate_data(confirmed, deaths, recovered, country)

        # Traitement des dates
        df = self._process_dates(df)

        # Filtrage temporel
        df = self._filter_dates(df)

        # Correction des anomalies
        df = self._filter_anomalies(df)

        # Calcul du nombre de jours depuis le début
        df["Jour"] = (df["Date"] - df["Date"].min()).dt.days

        # Calcul des Susceptibles (Population - Infectés - Guéris - Décédés)
        df["Susceptibles"] = self.population - df[
            ["Infectes", "Retablis", "Deces"]
        ].sum(axis=1)

        # Formatage final
        df = df[["Jour", "Susceptibles", "Infectes", "Retablis", "Deces", "Date"]]

        # Sauvegarde et retour
        if save:
            self._save(df, f"sird_{country.lower()}.csv")

        if split:
            return self._split_and_save(df, country, save)

        return df

    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gestion complète des dates"""
        df["Date"] = pd.to_datetime(df.index, format="%m/%d/%y", errors="coerce")
        return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    def _filter_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Application des filtres temporels"""
        # Filtre de début
        # Les valeurs de décès et de guérisons avant avril 2020 restent nulles (plateau à 0),
        # ce qui fausse les dynamiques SIRD. On commence donc à partir du 1er avril 2020.
        start_date = pd.to_datetime("2020-04-01")
        df = df[df["Date"] >= start_date]

        # Filtre de fin
        # Les données de guérisons s'arrêtent brutalement à une certaine date,
        # donc on tronque aussi les données à la dernière date où une guérison est renseignée.
        valid_r = df[df["Retablis"] > 0]
        if not valid_r.empty:
            last_valid_date = valid_r["Date"].max()
            df = df[df["Date"] <= last_valid_date]

        return df

    def _aggregate_data(
        self,
        confirmed: pd.DataFrame,
        deaths: pd.DataFrame,
        recovered: pd.DataFrame,
        country: str,
    ):
        """Agrège les données brutes pour le pays spécifié"""
        return pd.DataFrame(
            {
                "Infectes": confirmed.loc[confirmed["Country/Region"] == country]
                .iloc[:, 4:]
                .sum(),
                "Deces": deaths.loc[deaths["Country/Region"] == country]
                .iloc[:, 4:]
                .sum(),
                "Retablis": recovered.loc[recovered["Country/Region"] == country]
                .iloc[:, 4:]
                .sum(),
            }
        )

    def _filter_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corrige les anomalies communes dans les données épidémiques
        1. Élimination des valeurs négatives
        2. Correction des diminutions illogiques (les compteurs ne doivent pas diminuer)
        """
        # Suppression des valeurs négatives par seuillage
        for col in ["Infectes", "Retablis", "Deces"]:
            df[col] = df[col].clip(lower=0)

        # Application d'un cumul maximum pour éviter les diminutions
        for col in ["Retablis", "Deces"]:
            df[col] = df[col].cummax()

        return df

    def _save(self, df: pd.DataFrame, filename: str):
        """Sauvegarde le DataFrame nettoyé au format CSV"""
        path = self.processed_path / filename
        df.to_csv(path, index=False)

    def _split_and_save(self, df: pd.DataFrame, country: str, save: bool = True):
        """Crée une séparation temporelle 80/20 (train/test) et les sauvegarde"""
        split_idx = int(0.8 * len(df))
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]

        if save:
            self._save(df_train, f"sird_{country.lower()}_train.csv")
            self._save(df_test, f"sird_{country.lower()}_test.csv")

        return df_train, df_test
