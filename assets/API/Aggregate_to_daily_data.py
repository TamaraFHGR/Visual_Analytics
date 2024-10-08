import pandas as pd

# Read the CSV file
df = pd.read_csv('daily/04_SLF_daily_imis_measurements_clean.csv')

# Convert the 'measure_date' column to datetime
df['date_only'] = df['measure_date'].str[:10]

# Optional: display the DataFrame to verify the new column
print(df.head())


# aggregate the data:
# group by station_code and measure_date