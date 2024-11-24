from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import os
from datetime import datetime
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Data_Loader import (load_hist_data, load_imis_stations, load_measurements, load_snow,
                         load_measurements_trend, load_snow_trend,
                         load_kmeans_training, load_pca_training, load_pca_live, load_kmeans_centers)

acc_df = load_hist_data()
imis_df = load_imis_stations()
daily_weather_df = load_measurements()
daily_snow_df = load_snow()
trend_measure_df = load_measurements_trend()
trend_snow_df = load_snow_trend()
k_means_training_df = load_kmeans_training()
pca_training_df = load_pca_training()
pca_live_df = load_pca_live()
k_centers_df = load_kmeans_centers()

# Define color map for the K-Means clusters:
color_map = {0: 'orange', #3 - moderate
             1: 'lightcoral', #4 - considerable
             2: 'yellow', #2 - low
             3: 'lightgreen', #1 - very low
             4: 'darkred'} #5 - high
last_update = datetime.fromtimestamp(os.path.getmtime('assets/API/daily/05_SLF_daily_imis_snow_clean.csv')).strftime('%Y-%m-%d-%H:%M')

# Create the Dash app:
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, '/assets/custom_style.css'])

app.layout = html.Div([
    # Hidden Div for URL storage
    dcc.Location(id='url', refresh=True),
    html.Div(id='selected-url', style={'display': 'none'}),

    # Header:
    html.Div([
        html.H1('Visual Analytics Tool on Snow Avalanches'),
    ], className='header'),

    # Column 1 of 2 (left):
    html.Div([
        html.Div([
            html.Div([
                html.H3(f'Last Update: {last_update}', style={'color': 'grey'}),
                html.H4('Current Snow and Weather Conditions in the Swiss Alps'),
                html.Div([
                    html.Button('Stations', id='default_button', n_clicks=0),
                    html.Button('Temperature', id='temp_button', n_clicks=0),
                    html.Button('Wind', id='wind_button', n_clicks=0),
                    html.Button('Snow Height', id='snow_height_button', n_clicks=0),
                    html.Button('New Snow', id='new_snow_button', n_clicks=0)
                ], className='weather_buttons'),
                html.Div([
                    dcc.Dropdown(
                        id='station_dropdown',
                        options=[{'label': station, 'value': station} for station in imis_df['code'].unique()],
                        multi=True,
                        placeholder="Select stations",
                        value=[])
                ], className='station_dropdown'),
            ], className='sub_header'),

            # Live Weather Data and Trend:
            html.Div([
                html.Div([
                    html.H2('Live Weather Monitoring'),
                    dcc.Graph(id='live_geomap')], className='map_graph'),
                html.Div([
                    html.H2('Weather Trend Analysis'),
                    dcc.Graph(id='trend_analysis')], className='trend_graph')
                ], className='weather_row'),

            # Risk Assessment Clusters and Matrix:
            html.Div([
                html.Div([
                    html.H2('Risk-Assessment Clustering'),
                    dcc.Dropdown(
                        id='date_dropdown',
                        options=[{'label': date, 'value': date} for date in sorted(pca_live_df['measure_date'].unique())],
                        placeholder="Select a Measure Date...",
                        value=pca_live_df['measure_date'].max()
                    ),
                    dcc.Graph(id='k_cluster_map')], className='k_cluster_scatter'),
                html.Div([
                    html.H2('Risk-Assessment Matrix'),
                    dcc.Dropdown(
                        id='region_dropdown',
                        options=[{'label': region, 'value': region} for region in
                                 sorted(pca_live_df['alpine_region'].unique())],
                        placeholder="Select an Alpine Region...",
                        value=pca_live_df['alpine_region'].unique()[0]
                    ),
                    dcc.Graph(id='cluster_matrix')], className='k_cluster_matrix')
                ], className='k_cluster_row'),
        ], style={'display': 'inline-block'}, className='left_column'),

        # Column 2 of 2 (right):
        html.Div([
            html.H4('Historical Avalanche Observations'),
            html.Div([
                html.H2('Avalanche Accident Data since 1998'),
                dcc.RangeSlider(id='altitude', min=1000, max=4000, step=500),
                dcc.Graph(id='training_geomap')
            ], className='hover_cursor'),
            html.Div([
                html.H2('Statistics on Accident Data'),
                dcc.Graph(id='accidents_stats')
            ], className='training_data'),
        ], style={'display': 'inline-block'}, className='right_column')
    ], className='main_container')
    ])


if __name__ == '__main__':
    app.run_server(debug=True)