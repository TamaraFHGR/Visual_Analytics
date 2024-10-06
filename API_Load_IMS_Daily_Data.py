import requests
import pandas as pd

# URL der API
url = 'https://measurement-api.slf.ch/public/api/imis/measurements'

# API-Aufruf
response = requests.get(url)

# Überprüfen, ob die Anfrage erfolgreich war
if response.status_code == 200:
    # JSON-Antwort verarbeiten
    data = response.json()

    # In ein Pandas DataFrame umwandeln
    df = pd.DataFrame(data)

    # Speichern unter dem Namen '00_SLF_IMS_stations'
    df.name = 'test_SLF_ims_measurements'

    # Ausgabe des DataFrames
    print(df.head())  # zeigt die ersten Zeilen an
else:
    print(f"Fehler bei der Anfrage: {response.status_code}")

# Datensatz speichern
df.to_csv(f"assets/{df.name}.csv", sep=';', index=True)

