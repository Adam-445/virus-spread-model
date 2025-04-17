from pathlib import Path

import pandas as pd


class DataCleaner:
    """
    Classe de transformation des données brutes en format SIRD normalisé

    Attributes:
        country (str): Nom du pays ou code ISO3
        use_iso_code (bool): True pour utiliser le code ISO3 au lieu du nom
        processed_path (Path): Répertoire de sortie des données nettoyées
    """

    def __init__(
        self,
        processed_path: Path,
        country: str = "France",
        use_iso_code: bool = False,
    ):
        self.processed_path = (
            processed_path
            or Path(__file__).resolve().parents[2]
            / "data/processed"
            / self.country.lower()
        )
        self.processed_path.mkdir(parents=True, exist_ok=True)

        self.country = country.capitalize()
        self.use_iso_code = use_iso_code

    def clean_and_save(
        self,
        global_df: pd.DataFrame,
        save: bool = True,
        start_date: str = None,
        end_date: str = None,
    ) -> dict:
        """
        Pipeline complet de nettoyage.

        Args:
            global_df (DataFrame): Données OWID complètes
            save (bool): Sauvegarder les fichiers CSV
            start_date (str): Filtre temporel (YYYY-MM-DD)
            end_date (str): Filtre temporel (YYYY-MM-DD)

        Returns:
            dict: { "train": DataFrame, "test": DataFrame }
        """
        # 1. Filtrage pays
        df = self._filter_country_data(global_df)

        # 2. Filtrage date
        df = self._filter_dates(df, start_date, end_date)

        # 3. Calcul des compartiments SIRD
        df = self._calculate_sird(df)

        # 4. Split train/test
        train, test = self._split_data(df)

        # 5. Sauvegarde optionnelle
        if save:
            self._save(train, test)

        return {"train": train, "test": test}

    def _filter_country_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtre les données pour un pays spécifique (par nom ou ISO3)"""
        if self.use_iso_code:
            filtered = df[df["iso_code"] == self.country].copy()
        else:
            filtered = df[df["location"] == self.country].copy()

        if filtered.empty:
            raise ValueError(f"Aucune donnée trouvée pour le pays : {self.country}")

        return filtered

    def _filter_dates(
        self, df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Filtrage temporel optionnel"""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        if len(df) < 50:
            raise ValueError(
                f"Période trop courte ({len(df)} jours) pour {self.country}"
            )

        return df

    def _calculate_sird(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les compartiments S, I, R, D"""
        population = df["population"].iloc[0]

        # Nettoyage des colonnes d'intérêt
        df["total_cases"] = df["total_cases"].ffill().clip(lower=0)
        df["total_deaths"] = df["total_deaths"].ffill().clip(lower=0)

        # Calcul des infectés : somme glissante des nouveaux cas
        new_cases = df["total_cases"].diff().fillna(0).clip(lower=0)
        df["infected"] = new_cases.rolling(window=14, min_periods=1).sum()

        # Recovered = total_cases - infected - deaths
        df["recovered"] = (
            df["total_cases"] - df["infected"] - df["total_deaths"]
        ).clip(lower=0)

        # Susceptibles = population - tout le reste
        df["susceptible"] = (
            population - df["infected"] - df["recovered"] - df["total_deaths"]
        ).clip(lower=0)

        # Normalisation
        df["S"] = (df["susceptible"] / population).clip(0, 1)
        df["I"] = (df["infected"] / population).clip(0, 1)
        df["R"] = (df["recovered"] / population).clip(0, 1)
        df["D"] = (df["total_deaths"] / population).clip(0, 1)

        # Ajout du jour relatif
        df["Jour"] = (df["date"] - df["date"].min()).dt.days + 1

        return df[["date", "Jour", "S", "I", "R", "D"]].fillna(0)

    def _split_data(self, df: pd.DataFrame) -> tuple:
        """Découpe 80/20 pour train/test"""
        split_idx = max(int(len(df) * 0.8), 50)
        return df.iloc[:split_idx], df.iloc[split_idx:]

    def _save(self, train: pd.DataFrame, test: pd.DataFrame):
        """Sauvegarde des fichiers CSV"""
        base = f"sird_{self.country.lower()}"
        train.to_csv(self.processed_path / f"{base}_train.csv", index=False)
        test.to_csv(self.processed_path / f"{base}_test.csv", index=False)
