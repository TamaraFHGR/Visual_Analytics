import pandas as pd
import requests
import plotly.graph_objs as go
import plotly.io as pio
#---------------------------------------------------

# Part 1 - Load static Data:
# 1.1) Load Geodata of IMS stations:
ims_df = pd.read_csv('assets/01_SLF_ims_stations.csv', sep=';', skiprows=0)
print(ims_df.head())

# 1.2) Load historical data of avalanche accidents:
acc_df = pd.read_csv('assets/02_SLF_hist_avalanche_accidents.csv', sep=';',skiprows=3)
#print(avalanche_df.head())

# 1.3) Load historical data of Snow height (HS) and New Snowfall (HN_1D):
hist_snow_df = pd.read_csv('assets/03_SLF_hist_daily_snow.csv', sep=';',skiprows=0)
#print(snow_df.head())

