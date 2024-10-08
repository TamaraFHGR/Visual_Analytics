import pandas as pd
import glob
import os

"""
#Part 1: Aggregate daily data from multiple CSV files:
input_folder = 'C:/Users/tamar/alle_csv/'
output_folder = 'C:/Users/tamar/alle_csv/agg_daily/'

expected_columns = [
    'station_code', 'measure_date', 'HS', 'TA_30MIN_MEAN', 'RH_30MIN_MEAN',
    'TSS_30MIN_MEAN', 'TS0_30MIN_MEAN', 'TS25_30MIN_MEAN', 'TS50_30MIN_MEAN',
    'TS100_30MIN_MEAN', 'RSWR_30MIN_MEAN', 'VW_30MIN_MEAN', 'VW_30MIN_MAX',
    'DW_30MIN_MEAN', 'DW_30MIN_SD'
]

csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

# Loop over all CSV files
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file, sep=',', skiprows=0)

    # Add missing columns with NA values
    for col in expected_columns:
        if col not in df.columns:
            df[col] = pd.NA  # oder np.nan, falls du NumPy verwenden möchtest

    # Extract date from 'measure_date':
    df['date'] = df['measure_date'].str[:10]

    # Aggregate to daily data:
    df_agg = df.groupby(['date', 'station_code']).agg({
        'station_code': 'first',
        'date': 'first',
        'measure_date': 'first',
        'hyear': 'first',
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

    output_file = os.path.join(output_folder, os.path.basename(file).replace('.csv', '_agg.csv'))

    # Save the aggregated DataFrame to a new CSV file
    df_agg.to_csv(output_file, sep=';', index=False)
"""

# Part 2: Combine all aggregated CSV files into one file:
agg_folder = 'C:/Users/tamar/alle_csv/agg_daily/'
output_file = 'C:/Users/tamar/alle_csv/combined_agg_file.csv'

# Alle aggregierten CSV-Dateien im Ordner erfassen
agg_files = glob.glob(os.path.join(agg_folder, '*_agg.csv'))

# Liste für DataFrames
df_list = []

# Schleife über alle aggregierten Dateien
for file in agg_files:
    df = pd.read_csv(file, sep=';')
    df_list.append(df)

# Alle DataFrames zu einem großen DataFrame zusammenführen
combined_df = pd.concat(df_list, ignore_index=True)

# Den kombinierten DataFrame in eine einzelne CSV-Datei speichern
combined_df.to_csv(output_file, sep=';', index=False)

