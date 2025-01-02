import pandas as pd

# Read the CSV file
df = pd.read_csv('daily/04_SLF_daily_imis_measurements.csv', sep=';', skiprows=0, low_memory=False)

# Extract date from 'measure_date':
df['date'] = df['measure_date'].str[:10]

# Aggregate to daily data:
df = df.groupby(['date', 'station_code']).agg({
    'station_code': 'first',
    'date': 'first',
    'measure_date': 'first',
    'HS': 'median',
    'TA_30MIN_MEAN': 'mean',
    'RH_30MIN_MEAN': 'mean',
    'TSS_30MIN_MEAN': 'mean',
    'TS0_30MIN_MEAN': 'mean',
    'TS25_30MIN_MEAN': 'mean',
    'TS50_30MIN_MEAN': 'mean',
    'TS100_30MIN_MEAN': 'mean',
    'RSWR_30MIN_MEAN': 'mean',
    'VW_30MIN_MEAN': 'mean',
    'VW_30MIN_MAX': 'max',
    'DW_30MIN_MEAN': 'mean',
    'DW_30MIN_SD': 'mean'
})

# Save the DataFrame to a new CSV file
df.to_csv('daily/04_SLF_daily_imis_measurements_daily.csv', sep=';', index=False)

# Optional: display the DataFrame to verify the new column
print(df.head())
