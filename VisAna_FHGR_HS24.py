import pandas as pd
import plotly.graph_objs as go
import gdown

"""
------------------------------------------------------------------------------------------------------------------------
Part 1 - Load static Data:
In this part, the historical (static) data are loaded from CSV files.
The data are divided into three datasets:
"""

# 1.1) Load Geodata of 204 IMIS stations (some only collect wind and temperature data, others also snow data):
imis_df = pd.read_csv('assets/00_SLF_imis_stations.csv', sep=';', skiprows=0)
print(imis_df.head())

# 1.2) Load historical data of avalanche accidents (3'301 records since 1998):
acc_df = pd.read_csv('assets/01_SLF_hist_avalanche_accidents.csv', sep=';',skiprows=0)
print(acc_df.head())

# 1.3) Load historical IMIS wind and temperature data:
# Dataset >100MB and too large for GitHub. Download from Google Drive instead:
url = 'https://drive.google.com/uc?id=1LwGMAvYekeEeD2f37E3YyMidP-nxlyzp'
output = 'assets/02_SLF_hist_daily_measurements.csv'
gdown.download(url, output, quiet=False)

hist_measure_df = pd.read_csv('assets/02_SLF_hist_daily_measurements.csv', sep=';',skiprows=0)
print(hist_measure_df.head())

# 1.4) Load historical IMIS snow data:
hist_snow_df = pd.read_csv('assets/03_SLF_hist_daily_snow.csv', sep=';',skiprows=0)
print(hist_snow_df.head())

"""
------------------------------------------------------------------------------------------------------------------------
Part 2 - Update live Data:
In this part, the realtime data are loaded.
The collection of the data is done by connecting to the respective APIs (see scripts in asset folder).
"""

# 2.1) Load daily IMIS measurement data (updated every 30 minutes) and aggregate to daily data:
# New data collected every day at 16:00 UTC with '01_API_Load_IMIS_Daily_Data.py'
measure_df = pd.read_csv('assets/API/daily/04_SLF_daily_imis_measurements_daily.csv', sep=';',skiprows=0)
print(measure_df.head())

# 2.2) Load daily IMIS snow data (updated once a day):
# New data collected every day at 16:00 UTC with '02_API_Load_IMIS_Daily_Snow'
snow_df = pd.read_csv('assets/API/daily/05_SLF_daily_imis_snow_clean.csv', sep=';',skiprows=0)
print(snow_df.head())

# Data cleaning is done in the scripts '03_Delete_Duplicates.py' and '04_Aggregate_to_daily_data.py'

"""
------------------------------------------------------------------------------------------------------------------------
Part 3 - Visualize IMIS stations and accident locations on a map with Plotly:
In this part, a geo map is created to visualize the the IMIS stations
and the locations of the avalanche accidents.
"""

def imis_accident_map(imis_df, acc_df):
    latitudes_imis = imis_df['lat']
    latitudes_acc = acc_df['start_zone_coordinates_latitude']
    longitudes_imis = imis_df['lon']
    longitudes_acc = acc_df['start_zone_coordinates_longitude']
    station_names = imis_df['station_name']
    accident_names = acc_df['municipality']

    # Plot IMIS locations:
    fig = go.Figure(go.Scattermapbox(
        lat=latitudes_imis,
        lon=longitudes_imis,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=9,
            color='black'),
        text=station_names)
    )

    # Plot accident locations:
    fig.add_trace(go.Scattermapbox(
        lat=latitudes_acc,
        lon=longitudes_acc,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=5,
            color='red'),
        text=accident_names)
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=7,  # Adjust zoom level depending on your data
        mapbox_center={"lat": latitudes_imis.mean(), "lon": longitudes_imis.mean()},  # Center map on the average location
        margin={"r":0,"t":0,"l":0,"b":0}  # Remove margins around the map
    )

    fig.write_html('imis_stations_and_accidents.html', auto_open=True)


imis_accident_map(imis_df, acc_df)
