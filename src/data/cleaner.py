from pathlib import Path

import pandas as pd


class DataCleaner:
    """
    Transforme les données épidémiologiques brutes en format SIRD normalisé
    avec gestion des valeurs manquantes et validation des entrées.
    """

    def __init__(
        self,
        processed_path: Path,
        country: str = "France",
        use_iso_code: bool = False,
        smoothing: bool = True,
        window_size: int = 7,
    ):
        """
        Initialise le nettoyeur de données.

        Args:
            processed_path: Répertoire de sortie pour les données nettoyées
            country: Nom du pays ou code ISO3 (case insensitive)
            use_iso_code: True si le pays est spécifié par code ISO3
            smoothing: Si True, applique un lissage aux données
            window_size: Taille de la fenêtre pour le lissage (par défaut 7 jours)

        Raises:
            ValueError: Si le répertoire de sortie ne peut être créé
        """
        self.country = country.strip().capitalize()
        self.use_iso_code = use_iso_code
        self.smoothing = smoothing
        self.window_size = window_size

        # Configuration des chemins
        self.processed_path = processed_path
        try:
            self.processed_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(
                f"Erreur de création du répertoire {self.processed_path}: {e}"
            )

    def clean_and_save(
        self,
        global_df: pd.DataFrame,
        save: bool = True,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
    ) -> dict[str, pd.DataFrame]:
        """
        Pipeline complet de nettoyage.

        Étapes:
        1. Filtrage des données par pays
        2. Filtrage temporel
        3. Calcul des métriques SIRD
        4. Lissage des données (optionnel)
        5. Découpe des données
        6. Sauvegarde (optionnelle)

        Args:
            global_df: DataFrame brut de l'OWID
            save: Sauvegarde les résultats en CSV si True
            start_date: Date de début au format YYYY-MM-DD
            end_date: Date de fin au format YYYY-MM-DD

        Returns:
            dict: { "train": DataFrame, "test": DataFrame }
        """
        # Pipeline de traitement
        try:
            # Filtrage pays
            df = self._filter_country_data(global_df)
            # Filtrage date
            df = self._filter_dates(df, start_date, end_date)
            # Calcul des compartiments SIRD
            df = self._calculate_sird(df)
            # Lissage des données si nécessaire
            if self.smoothing:
                df = self._smooth_data(df)
            # Split train/test
            train, test = self._split_data(df)
        except KeyError as e:
            raise ValueError(f"Colonne manquante dans les données: {e}")

        # Sauvegarde des résultats
        if save:
            self._save(train, test)

        return {"train": train, "test": test}

    def _filter_country_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtre les données pour le pays spécifié avec gestion d'erreur améliorée."""
        col_name = "iso_code" if self.use_iso_code else "location"
        country_id = self.country.upper() if self.use_iso_code else self.country

        filtered = df[df[col_name] == country_id].copy()

        if filtered.empty:
            available = df[col_name].unique().tolist()
            raise ValueError(
                f"Pays '{country_id}' non trouvé. Disponibles: {available[:5]}..."
                f"\nAstuce: Vérifiez les codes ISO3"
            )

        return filtered

    def _filter_dates(
        self, df: pd.DataFrame, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Filtre les données selon une plage temporelle."""
        MIN_DAYS = 50  # Minimum requis pour l'analyse temporelle

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Application des filtres de date
        if start_date:
            start = pd.to_datetime(start_date)
            df = df[df["date"] >= start]
        if end_date:
            end = pd.to_datetime(end_date)
            df = df[df["date"] <= end]

        # Validation de la période résultante
        if len(df) < MIN_DAYS:
            raise ValueError(
                f"Période trop courte ({len(df)} jours). Minimum requis: {MIN_DAYS} jours"
            )

        return df

    def _calculate_sird(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les compartiments SIRD à partir des données brutes.

        Logique:
        - Sains (S): Population - (Infectés + Guéris + Décédés)
        - Infectés (I): Somme glissante sur 14 jours des nouveaux cas
        - Guéris (R): Total des cas - Infectés actifs - Décédés
        - Décédés (D): Total des décès cumulés

        Les valeurs sont stockées en absolu et en proportion de la population.
        """
        self.population = df["population"].iloc[0]

        # Nettoyage des colonnes d'intérêt
        df["total_cases"] = df["total_cases"].ffill().clip(lower=0)
        df["D_abs"] = df["total_deaths"].ffill().clip(lower=0)

        # Calcul des infectés : somme glissante des nouveaux cas
        new_cases = df["total_cases"].diff().fillna(0).clip(lower=0)
        df["I_abs"] = new_cases.rolling(window=14, min_periods=1).sum()

        # Recovered = total_cases - infected - deaths
        df["R_abs"] = (df["total_cases"] - df["I_abs"] - df["D_abs"]).clip(lower=0)

        # Susceptibles = population - tout le reste
        df["S_abs"] = (self.population - df["I_abs"] - df["R_abs"] - df["D_abs"]).clip(
            lower=0
        )

        # Doses absolues
        df["V_abs"] = df["people_fully_vaccinated"].ffill().diff().fillna(0)

        # Normalisation par la population
        for col in ["S", "I", "R", "D", "V"]:
            df[col] = (df[f"{col}_abs"] / self.population).clip(0, 1)

        # Moyenne des lits disponibles par 1000 personnes (valeur constante pour tout le pays)
        beds = df["hospital_beds_per_thousand"].dropna()
        beds_value = beds.iloc[0] if not beds.empty else 0
        df["lits_par_mille"] = beds_value

        # Ajout du jour relatif
        df["Jour"] = (df["date"] - df["date"].min()).dt.days + 1
        df = df.set_index("Jour").sort_index()

        return df[
            [
                "date",
                "S",
                "I",
                "R",
                "D",
                "S_abs",
                "I_abs",
                "R_abs",
                "D_abs",
                "V",
                "V_abs",
                "lits_par_mille",
            ]
        ].fillna(0)

    def _smooth_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique un lissage aux données en utilisant une moyenne mobile.

        Args:
            df: DataFrame contenant les données à lisser

        Returns:
            DataFrame avec les données lissées
        """
        # Lissage pour les colonnes S, I, R, D
        columns_to_smooth = ["S", "I", "R", "D", "V"]
        for col in columns_to_smooth:
            df[col] = df[col].rolling(window=self.window_size, min_periods=1).mean()
            df[f"{col}_abs"] = (
                df[f"{col}_abs"].rolling(window=self.window_size, min_periods=1).mean()
            )
        return df

    def _split_data(self, df: pd.DataFrame) -> tuple:
        """Découpe les données en ensembles d'entraînement (80%) et de test (20%)."""
        split_ratio = 0.8
        split_idx = int(len(df) * split_ratio)
        return df.iloc[:split_idx], df.iloc[split_idx:]

    def _save(self, train: pd.DataFrame, test: pd.DataFrame):
        """Sauvegarde les données au format CSV avec nommage standardisé."""
        base_name = f"sird_{self.country.lower()}"
        train.to_csv(self.processed_path / f"{base_name}_train.csv", index=True)
        test.to_csv(self.processed_path / f"{base_name}_test.csv", index=True)
