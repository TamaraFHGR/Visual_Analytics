import pandas as pd
"""
------------------------------------------------------------------------------------------------------------------------
Part 1 - Load historical (static) Data:
"""
def load_hist_data():
    acc_df = pd.read_csv('assets/01_SLF_hist_statistical_avalanche_data.csv', sep=',', skiprows=0)
    return acc_df

acc_df = load_hist_data()
#print(acc_df.head())

def load_imis_stations():
    station_df = pd.read_csv('assets/Raw_Data/00_SLF_imis_stations.csv', sep=';', skiprows=0)
    return station_df

stations_df = load_imis_stations()
#print(stations_df.head())

"""
------------------------------------------------------------------------------------------------------------------------
Part 2 - Load real-time (daily, dynamic) Data:
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