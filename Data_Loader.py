import pandas as pd
import gdown
"""
------------------------------------------------------------------------------------------------------------------------
Part 1 - Load static Data:
In this part, the historical (static) data are loaded from CSV files.
The data are divided into three datasets:
"""
# 1.1) Load Geodata of 204 IMIS stations (some only collect wind and temperature data, others also snow data):
def load_imis_stations():
    imis_df = pd.read_csv('assets/00_SLF_imis_stations.csv', sep=';', skiprows=0)
    return imis_df

imis_df = load_imis_stations()
#print(imis_df.head())


# 1.2) Load historical data of avalanche accidents (3'301 records since 1998):
def load_accidents():
    acc_df = pd.read_csv('assets/01_SLF_hist_avalanche_accidents.csv', sep=';',skiprows=0)
    return acc_df

acc_df = load_accidents()
#print(acc_df.head())


# 1.3) Load historical IMIS wind and temperature data:
# Dataset >100MB and too large for GitHub. Download from Google Drive instead:
def load_hist_measurements():
    hist_measure_df = pd.read_csv('assets/02_SLF_hist_daily_measurements.csv', sep=';',skiprows=0)
    #url = 'https://drive.google.com/uc?id=1LwGMAvYekeEeD2f37E3YyMidP-nxlyzp'
    #output = 'assets/02_SLF_hist_daily_measurements.csv'
    #gdown.download(url, output, quiet=False)
    #hist_measure_df = pd.read_csv('assets/02_SLF_hist_daily_measurements.csv', sep=';',skiprows=0)
    return hist_measure_df

hist_measure_df = load_hist_measurements()
# print(hist_measure_df.head())

# 1.4) Load historical IMIS snow data:
def load_hist_snow():
    hist_snow_df = pd.read_csv('assets/03_SLF_hist_daily_snow.csv', sep=';',skiprows=0)
    return hist_snow_df

hist_snow_df = load_hist_snow()
# print(hist_snow_df.head())

"""
------------------------------------------------------------------------------------------------------------------------
Part 2 - Update live Data:
In this part, the realtime data are loaded.
The collection of the data is done by connecting to the respective APIs (see scripts in asset folder).
"""
# 2.1) Load daily IMIS measurement data (updated every 30 minutes) and aggregate to daily data:
# New data collected every day at 16:00 UTC with '01_API_Load_IMIS_Daily_Data.py'
def load_measurements():
    measure_df = pd.read_csv('assets/API/daily/04_SLF_daily_imis_measurements_daily.csv', sep=';',skiprows=0)
    return measure_df

measure_df = load_measurements()
# print(measure_df.head())

# 2.2) Load daily IMIS snow data (updated once a day):
# New data collected every day at 16:00 UTC with '02_API_Load_IMIS_Daily_Snow'
def load_snow():
    snow_df = pd.read_csv('assets/API/daily/05_SLF_daily_imis_snow_clean.csv', sep=';',skiprows=0)
    return snow_df

snow_df = load_snow()
# print(snow_df.head())

# Data cleaning is done in the scripts '03_Delete_Duplicates.py' and '04_Aggregate_to_daily_data.py'
