import requests
import pandas as pd

# URL and API-request:
url = 'https://measurement-api.slf.ch/public/api/imis/measurements'
response = requests.get(url)

# Check if the request was successful:
if response.status_code == 200:
    data = response.json()
    df_imis_d = pd.DataFrame(data) # Create a DataFrame from the JSON data
    df_imis_d.name = '04_SLF_daily_imis_measurements'
    csv_file_path = f"./daily/{df_imis_d.name}.csv"

    # Append new data to CSV-File
    df_imis_d.to_csv(csv_file_path, sep=';', index=True, mode='a',
              header=False)  # header=False, to avoid duplicate headers
    print(df_imis_d.head())
else:
    print(f"Error during request: {response.status_code}")