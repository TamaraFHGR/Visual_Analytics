import pandas as pd
import plotly.graph_objs as go

"""
------------------------------------------------------------------------------------------------------------------------
Part 1 - Load static Data:
In this part, the historical (static) data are loaded. The data are divided into three datasets:
"""

# 1.1) Load Geodata of IMIS stations:
imis_df = pd.read_csv('assets/00_SLF_imis_stations.csv', sep=';', skiprows=0)
print(imis_df.head())

# 1.2) Load historical data of avalanche accidents:
acc_df = pd.read_csv('assets/01_SLF_hist_avalanche_accidents.csv', sep=';',skiprows=3)
print(acc_df.head())

# 1.3) Load historical IMIS measurement data:
hist_measure_df = pd.read_csv('assets/02_SLF_hist_daily_measurements.csv', sep=';',skiprows=0)
print(hist_measure_df.head())

# 1.4) Load historical data of snow height (HS) and new snowfall (HN_1D):
hist_snow_df = pd.read_csv('assets/03_SLF_hist_daily_snow.csv', sep=';',skiprows=0)
print(hist_snow_df.head())

"""
------------------------------------------------------------------------------------------------------------------------
Part 2 - Update live Data:
In this part, the realtime data are loaded.
The collection of the data is done by connecting to the respective APIs (in asset folder).
"""

# 2.1) Load daily IMIS measurement data (updated every 30 minutes):
# New data collected every day at 14:00 with 'API_Load_IMIS_Daily_Data.py'
measure_df = pd.read_csv('assets/API/daily/04_SLF_daily_imis_measurements_daily.csv', sep=';',skiprows=0)
print(measure_df.head())

# 2.2) Load daily IMIS snow data (updated once a day):
# New data collected every day at 14:00 with 'API_Load_IMIS_Daily_Snow'
snow_df = pd.read_csv('assets/API/daily/05_SLF_daily_imis_snow_clean.csv', sep=';',skiprows=0)
print(snow_df.head())

"""
------------------------------------------------------------------------------------------------------------------------
Part 3 - Visualize IMIS stations and accident locations on a map with Plotly:
In this part, a geo map is created to visualize the the IMIS stations
and the locations of the avalanche accidents.
"""

def imis_accident_map(imis_df, acc_df):
    # filter data on 2022 - 2023:
    acc_df = acc_df[(acc_df['hydrological.year'] == '2021/22') | (acc_df['hydrological.year'] == '2022/23')]
    latitudes_imis = imis_df['lat']
    latitudes_acc = acc_df['start.zone.coordinates.latitude']
    longitudes_imis = imis_df['lon']
    longitudes_acc = acc_df['start.zone.coordinates.longitude']
    station_names = imis_df['label']
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

#imis_accident_map(imis_df, acc_df)

"""
------------------------------------------------------------------------------------------------------------------------
Part 4 - Visualize snow height and new snowfall data:
In this part, the daily snow height and new snowfall data by stations are visualized.
"""
def snow_data_plot(snow_df):
    fig = go.Figure()

    for station in snow_df['station_code'].unique():
        station_data = snow_df[hist_snow_df['station_code'] == station]
        # Snow height as bar chart:
        fig.add_trace(go.Bar(x=station_data['measure_date'], y=station_data['HS'], name=f'{station} Snow Height'))
        # New snowfall as line chart:
        fig.add_trace(go.Scatter(x=station_data['measure_date'], y=station_data['HN_1D'], mode='lines', name=f'{station} New Snowfall'))

    fig.update_layout(
        title='Snow Height and New Snowfall Data',
        xaxis_title='Date',
        yaxis_title='Snow Height / New Snowfall',
        xaxis_tickformat='%d-%m-%Y',
        legend_title='Station',
        height=600,
        width=1200
    )

    fig.write_html('snow_height_new_snowfall.html', auto_open=True)

#snow_data_plot(hist_snow_df)

