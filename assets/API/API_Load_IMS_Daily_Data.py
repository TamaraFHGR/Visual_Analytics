import requests
import pandas as pd

# URL and API-request:
url = 'https://measurement-api.slf.ch/public/api/imis/measurements'
response = requests.get(url)

# Check if the request was successful:
if response.status_code == 200:
    data = response.json()
    df_ims_d = pd.DataFrame(data) # Create a DataFrame from the JSON data
    df_ims_d.name = '03_SLF_daily_ims_measurements'
    csv_file_path = f"assets/{df_ims_d.name}.csv"

    # Append new data to CSV-File
    df_ims_d.to_csv(csv_file_path, sep=';', index=True, mode='a',
              header=True)  # header=False, to avoid duplicate headers
    print(df_ims_d.head())
else:
    print(f"Error during request: {response.status_code}")
