import requests
import pandas as pd
import time

start_time = time.time()

# URL and API-request:
url = 'https://measurement-api.slf.ch/public/api/imis/daily-snow'
response = requests.get(url)

# Check if the request was successful:
if response.status_code == 200:
    data = response.json()
    df_imis_d = pd.DataFrame(data) # Create a DataFrame from the JSON data
    df_imis_d.name = '05_SLF_daily_imis_snow'
    csv_file_path = f"./daily/{df_imis_d.name}.csv"

    # Append new data to CSV-File
    df_imis_d.to_csv(csv_file_path, sep=';', index=True, mode='a',
              header=False)  # header=False, to avoid duplicate headers
    print(df_imis_d.head())
else:
    print(f"Error during request: {response.status_code}")

end_time = time.time()
time = end_time - start_time
print(f'Time: {time} seconds')     # Time: 0.5381348133087158 seconds