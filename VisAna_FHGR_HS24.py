import pandas as pd
import requests
import plotly.graph_objs as go
import plotly.io as pio
#---------------------------------------------------

# Part 1 - Load Data:
# 1.1) Load Geodata of IMS stations (static data):
ims_df = pd.read_csv('assets/00_SLF_ims_stations.csv', sep=';',skiprows=0)
print(ims_df.head())

# 1.2) Load Data of Snow height (HS) and New Snowfall (HN_1D):
#snow_df = pd.read_csv('assets/01_SLF_daily_snow.csv', sep=';',skiprows=0)
#print(snow_df.head())

# 3) Load Data of historical avalanche accidents:
#avalanche_df = pd.read_csv('assets/02_SLF_avalanche_accidents_all_switzerland_since_1970.csv', sep=';',skiprows=3)
#print(avalanche_df.head())