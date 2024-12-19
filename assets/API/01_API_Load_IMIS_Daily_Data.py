import requests
import pandas as pd

# URL mit Datum als Filterparameter
url = 'https://measurement-api.slf.ch/public/api/imis/measurements'

# API-Request mit Filter
response = requests.get(url)

# Überprüfung und Verarbeitung
if response.status_code == 200:
    data = response.json()
    df_imis_d = pd.DataFrame(data)
    df_imis_d.name = '04_SLF_daily_imis_measurements'
    csv_file_path = f"./daily/{df_imis_d.name}.csv"

    # Daten an CSV-Datei anhängen
    df_imis_d.to_csv(csv_file_path, sep=';', index=True, mode='a', header=False)
    print(df_imis_d.head())
else:
    print(f"Error during request: {response.status_code}")
