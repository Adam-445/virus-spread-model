import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def fetch_epidemic_data():
    """Télécharge et extrait les données COVID-19 de JHU"""
    # Get the project root directory (where setup.py is located)
    project_root = Path(__file__).resolve().parent.parent.parent

    # All paths are now relative to project root
    data_dir = project_root / "data/raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    # URL des données Johns Hopkins University
    url = "https://github.com/CSSEGISandData/COVID-19/archive/refs/heads/master.zip"
    zip_path = data_dir / "covid19.zip"

    # Téléchargement
    if not zip_path.exists():
        try:
            # Send a request to download the file in "streaming" mode
            response = requests.get(url, stream=True, timeout=10)
            # Check if the request was successful (status code 200)
            response.raise_for_status()

            # Get total file size (for progress bar)
            total = int(response.headers.get("content-length", 0))

            # Download the file in chunks (useful for big files)
            with open(zip_path, "wb") as f, tqdm(
                desc="Downloading",
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    # if chunk is not None (to avoid writing empty chunks)
                    if chunk:
                        # write the chunk to the file
                        f.write(chunk)

                        # update the progress bar
                        bar.update(len(chunk))

        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            return None, None, None

    # Extraction
    csv_path = data_dir / "COVID-19-master/csse_covid_19_data/csse_covid_19_time_series"
    if not csv_path.exists():
        print("Extracting files...")
        with zipfile.ZipFile(zip_path) as z:
            # Unzip everything into data/raw/
            z.extractall(data_dir)

    # Chargement des données
    confirmed = pd.read_csv(csv_path / "time_series_covid19_confirmed_global.csv")
    deaths = pd.read_csv(csv_path / "time_series_covid19_deaths_global.csv")
    recovered = pd.read_csv(csv_path / "time_series_covid19_recovered_global.csv")

    # Return the dataframes
    return confirmed, deaths, recovered


# if the script is run directly, do a quick test
if __name__ == "__main__":
    confirmed, deaths, recovered = fetch_epidemic_data()
    print("Confirmed cases:\n", confirmed.head())
