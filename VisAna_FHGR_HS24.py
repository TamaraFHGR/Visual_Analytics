from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import os
from datetime import datetime
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from jedi.api.refactoring import inline
from plotly.subplots import make_subplots
from Data_Loader import (load_imis_stations, load_hist_data, load_daily,
                         load_kmeans_centers, load_pca_training, load_pca_live)
"""
-------------------------------------------------------------------------------------------------------------------
-> Load Data:
"""
imis_df = load_imis_stations()
acc_df = load_hist_data()
daily_snow_df = load_daily()
k_centers_df = load_kmeans_centers()
pca_training_df = load_pca_training()
pca_live_df = load_pca_live()
last_update = datetime.fromtimestamp(os.path.getmtime('assets/API/daily/05_SLF_daily_imis_snow_clean.csv')).strftime('%Y-%m-%d-%H:%M')
"""
-------------------------------------------------------------------------------------------------------------------
-> Define filters:
"""
imis_df['combined'] = (
    imis_df['canton_code'].fillna('') + ' - ' +
    imis_df['code'].fillna('') + ' - ' +
    imis_df['station_name'].fillna('') + ' - ' +
    imis_df['elevation'].fillna(0).astype(int).astype(str) + ' m'
)
station_filter = imis_df['combined'].unique()
#print(station_filter)

all_regions = {
    'Entire Alpine Region': ['BE', 'FR', 'VD', 'TI', 'VS', 'OW', 'SZ', 'UR', 'FL', 'GL', 'GR', 'SG'],
    'Western Alps': ['BE', 'FR', 'VD'],
    'Southern Alps': ['TI', 'VS'],
    'Central Alps': ['OW', 'SZ', 'UR'],
    'Eastern Alps': ['FL', 'GL', 'GR', 'SG'],
}

button_ids = ['default_button', 'temp_button', 'wind_button', 'snow_height_button', 'new_snow_button']

color_map = {0: 'orange', #3 - moderate risk
             1: 'lightcoral', #4 - considerable risk
             2: 'yellow', #2 - low risk
             3: 'lightgreen', #1 - very low risk
             4: 'darkred'} #5 - high risk

month_mapping = {
    0: 10,  # October
    1: 11,  # November
    2: 12,  # December
    3: 1,  # January
    4: 2,  # February
    5: 3,  # March
    6: 4  # April
}
"""
-------------------------------------------------------------------------------------------------------------------
-> Create Dash App:
"""
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, '/assets/custom_style.css'])

app.layout = html.Div([
    # Hidden Div for URL storage
    dcc.Location(id='url', refresh=True),
    html.Div(id='selected-url', style={'display': 'none'}),

    # State Storage for Active Button
    dcc.Store(id='active_button', data='default_button'),

    # Title:
    html.Div([ # header
        html.H1('Visual Analytics on Avalanche Accidents and Snow Conditions'),
        html.H3(f'(last Update: {last_update})')
    ], className='header'),

    # Column 1 of 2 (left):
    html.Div([
        html.Div([ # left_column
            html.Div([ # sub_header
                html.Div([ # weather_buttons
                    html.Button('Station Overview', id='default_button', n_clicks=0, className='active'),
                    html.Button('Temperature', id='temp_button', n_clicks=0, className=''),
                    html.Button('Wind', id='wind_button', n_clicks=0, className=''),
                    html.Button('Snow Height', id='snow_height_button', n_clicks=0, className=''),
                    html.Button('New Snow', id='new_snow_button', n_clicks=0, className=''),
                ], className='weather_buttons'),
                html.Div([ # radio_group
                    html.Hr(),
                    dcc.RadioItems(
                        id='region_radio',
                        options=[{'label': region, 'value': region} for region in all_regions.keys()],
                        value='Entire Alpine Region',
                        inline=True,
                        className='region_radio'),
                    dcc.RadioItems(
                        id='canton_radio',
                        options=[],
                        value='None',
                        inline=True,
                        className='canton_radio'),
                ], className='radio_group'),
                html.Div([ # station_dropdown
                    dcc.Dropdown(
                        id='station_dropdown',
                        options=[{'label': combined, 'value': code}
                                 for combined, code in zip(imis_df['combined'], imis_df['code'])],
                        multi=True,
                        placeholder="Select any weather station...",
                        value=[])
                    ], className='station_dropdown'),
            ], className='sub_header'),

            # Live Weather Data and Trend:
            html.Div([ # weather_row
                html.Div([ # map_graph
                    html.H2('Snow and Weather Monitoring'),
                    dcc.Graph(id='live_geomap')
                ], className='map_graph'),
                html.Div([ # trend_graph
                    html.H2('Trend Analysis'),
                    dcc.Graph(id='trend_analysis'),
                    dcc.Slider(
                        id='moving_avg_window',
                        min=3,
                        max=9,
                        step=None,
                        value=7,
                        marks={3: '3 Days', 5: '5 Days', 7: '7 Days', 9: '9 Days'},
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                ], className='trend_graph')
            ], className='weather_row'),

            # Risk Assessment Clusters and Matrix:
            html.Div([
                html.Div([ # k_cluster_scatter
                    html.H2('Daily Risk Clustering)'),
                    dcc.Graph(id='k_cluster_map'),
                    dcc.Dropdown(
                        id='date_dropdown',
                        options=[{'label': date, 'value': date} for date in sorted(pca_live_df['measure_date'].unique())],
                        placeholder="Select a Measure Date...",
                        value=pca_live_df['measure_date'].max())
                    ], className='k_cluster_scatter'),
                html.Div([ # k_cluster_matrix
                    html.H2('Risk-Assessment Matrix'),
                    dcc.Graph(id='cluster_matrix'),
                    dcc.Slider(
                        id='month_slider',
                        min=0,
                        max=6,
                        step=None,
                        value=2,
                        marks={0: 'Oct 24',
                               1: 'Nov 24',
                               2: 'Dec 24',
                               3: 'Jan 25',
                               4: 'Feb 25',
                               5: 'Mar 25',
                               6: 'Apr 25'},
                        tooltip={"placement": "bottom", "always_visible": False})
                    ], className='k_cluster_matrix'),
                ], className='k_cluster_row'),
        ], style={'display': 'inline-block'}, className='left_column'),

        # Column 2 of 2 (right):
        html.Div([ # right_column
            html.Div([ # training_map
                html.H2('Historical Avalanche Accident since 1998'),
                dcc.Graph(id='training_geomap'),
                dcc.RangeSlider(id='altitude', min=1000, max=4000, step=500),
            ], className='training_map'),
            html.Div([ # training_data
                dcc.Graph(id='accidents_stats')
            ], className='training_data'),
        ], style={'display': 'inline-block'}, className='right_column')
    ], className='main_container')
    ])

"""
-------------------------------------------------------------------------------------------------------------------
-> Update Filter and Buttons:
"""
# Update Radio Button for Cantons:
@app.callback(
    Output('canton_radio', 'options'),
    Input('region_radio', 'value')
)
def update_canton_radio(selected_region):
    if selected_region:
        canton_codes = all_regions[selected_region]
        return [{'label': 'All Cantons', 'value': 'None'}] + [
            {'label': canton, 'value': canton} for canton in canton_codes
        ]
    else:
        return [{'label': 'All Cantons', 'value': 'None'}]

# Update Dropdown Menu for IMIS-Stations:
@app.callback(
    Output('station_dropdown', 'options'),
    [Input('region_radio', 'value'),
    Input('canton_radio', 'value')]
)
def update_station_dropdown(selected_region, selected_canton):
    filtered_df = imis_df.copy()
    # Filter by Alpine Region:
    if selected_region and selected_region != 'Entire Alpine Region':
        valid_regions = all_regions.get(selected_region, [])
        filtered_df = filtered_df[filtered_df['canton_code'].isin(valid_regions)]

    # Filter by Canton:
    if selected_canton and selected_canton != 'None':
        filtered_df = filtered_df[filtered_df['canton_code'] == selected_canton]

    options = [{'label': combined, 'value': code}
               for combined, code in zip(filtered_df['combined'], filtered_df['code'])]

    return options

# Update the stored active button in dcc.Store:
@app.callback(
    Output('active_button', 'data'),
    [Input(button_id, 'n_clicks') for button_id in button_ids],
    prevent_initial_call=True
)
def store_active_button(*args):
    ctx = callback_context
    if not ctx.triggered:
        return 'default_button'

    clicked_button = ctx.triggered[0]['prop_id'].split('.')[0]
    return clicked_button

# Update the button classes (active/inactive):
@app.callback(
    [Output(button_id, 'className') for button_id in button_ids],
    Input('active_button', 'data')
)
def update_button_classes(active_button):
    return ['active' if button_id == active_button else '' for button_id in button_ids]

# Callback to open the station document in a new tab:
@app.callback(
    Output('url', 'href'),
    [Input('live_geomap', 'clickData')]
)
def open_station_document(clickData):
    if clickData:
        # Extract URL from the click data
        url = clickData['points'][0]['customdata']
        return url
    return no_update

"""
-----------------------------------------------------------------------------------------
Section 1.1: left column - Map Visualization
"""
@app.callback(
    Output('live_geomap', 'figure'),
    [Input('active_button', 'data'),
     Input('region_radio', 'value'),
     Input('canton_radio', 'value'),
     Input('station_dropdown', 'value')],
)
def imis_live_map(active_button, selected_region, selected_canton, selected_stations):
    filtered_data = imis_df.copy()
    # URL for the IMIS station Details:
    filtered_data['data_url'] = filtered_data['code'].apply(lambda code: f"https://stationdocu.slf.ch/pdf/IMIS_{code}_DE.pdf")

    # a) Filter data based on selected region:
    if selected_region and selected_region != 'Entire Alpine Region':
        valid_regions = all_regions.get(selected_region, [])
        filtered_data = filtered_data[filtered_data['canton_code'].isin(valid_regions)]

    # b) Filter data based on selected canton:
    if selected_canton and selected_canton != 'None':
        filtered_data = filtered_data[filtered_data['canton_code'] == selected_canton]

    # c) Filter data based on selected stations:
    if selected_stations:
        filtered_data = filtered_data[filtered_data['code'].isin(selected_stations)]

    # Plot IMIS locations as "Station Overview":
    if active_button == 'default_button':
        fig = go.Figure(go.Scattermap(
            lat=filtered_data['lat'],
            lon=filtered_data['lon'],
            mode='markers',
            marker=go.scattermap.Marker(
                size=12,
                symbol='mountain',
                color='black'),
            text=filtered_data['combined'],
            hoverinfo='text',
            customdata=filtered_data['data_url'])
        )
        fig.update_layout(
            map_style="light",
            map_zoom=6,
            map_center={"lat": filtered_data['lat'].mean(), "lon": filtered_data['lon'].mean()},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=230
        )

    # Plot Heat Map "Temperature":
    elif active_button == 'temp_button':
        daily_snow = daily_snow_df[daily_snow_df['measure_date'] == daily_snow_df['measure_date'].max()]
        fig = go.Figure(go.Densitymap(
            lat=daily_snow['lat'],
            lon=daily_snow['lon'],
            z=daily_snow['air_temp_mean_stations'],
            radius=30,
            colorscale='blugrn',
            reversescale=True)
        )
        fig.update_layout(
            map_style="light",
            map_zoom=6,
            map_center={"lat": daily_snow['lat'].mean(), "lon": daily_snow['lon'].mean()},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=230
        )

    # Plot Heat Map "Wind":
    elif active_button == 'wind_button':
        daily_snow = daily_snow_df[daily_snow_df['measure_date'] == daily_snow_df['measure_date'].max()]
        fig = go.Figure(go.Densitymap(
            lat=daily_snow['lat'],
            lon=daily_snow['lon'],
            z=daily_snow['wind_speed_max_stations'],
            radius=30,
            colorscale='brwnyl',
            reversescale=False)
        )
        fig.update_layout(
            map_style="light",
            map_zoom=6,
            map_center={"lat": daily_snow['lat'].mean(), "lon": daily_snow['lon'].mean()},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=230
        )

    # Plot Heat Map "Snow Height":
    elif active_button == 'snow_height_button':
        daily_snow = daily_snow_df[daily_snow_df['measure_date'] == daily_snow_df['measure_date'].max()]
        fig = go.Figure(go.Densitymap(
            lat=daily_snow['lat'],
            lon=daily_snow['lon'],
            z=daily_snow['snow_height_mean_stations'],
            radius=30,
            colorscale='blues',
            reversescale=False)
        )
        fig.update_layout(
            map_style="light",
            map_zoom=6,
            map_center={"lat": daily_snow['lat'].mean(), "lon": daily_snow['lon'].mean()},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=230
        )

    # Plot Heat Map "New Snow":
    elif active_button == 'new_snow_button':
        daily_snow = daily_snow_df[daily_snow_df['measure_date'] == daily_snow_df['measure_date'].max()]
        fig = go.Figure(go.Densitymap(
            lat=daily_snow['lat'],
            lon=daily_snow['lon'],
            z=daily_snow['new_snow_mean_stations'],
            radius=30,
            colorscale='dense',
            reversescale=False)
        )
        fig.update_layout(
            map_style="light",
            map_zoom=6,
            map_center={"lat": daily_snow['lat'].mean(), "lon": daily_snow['lon'].mean()},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=230
        )

    return fig


"""
-----------------------------------------------------------------------------------------
Section 1.2: left column - Trend Visualization
"""
@app.callback(
    Output('trend_analysis', 'figure'),
    [Input('active_button', 'data'),
     Input('region_radio', 'value'),
     Input('canton_radio', 'value'),
     Input('station_dropdown', 'value'),
     Input('moving_avg_window', 'value')],
)
def weather_trend(active_button, selected_region, selected_canton, selected_stations, moving_window):
    # Filter data based on region, canton, and station
    filtered_data = daily_snow_df.copy()

    if selected_region and selected_region != 'Entire Alpine Region':
        valid_regions = all_regions.get(selected_region, [])
        filtered_data = filtered_data[filtered_data['canton_code'].isin(valid_regions)]

    if selected_canton and selected_canton != 'None':
        filtered_data = filtered_data[filtered_data['canton_code'] == selected_canton]

    if selected_stations:
        filtered_data = filtered_data[filtered_data['station_code'].isin(selected_stations)]

    # Function to calculate mean for any given value column
    def calculate_mean(df, value_column):
        return df.groupby('measure_date')[value_column].mean()

    # Function to plot the trend data
    def plot_trend(df, value_column, y_axis_label, is_bar=False, with_moving_avg=False):
        mean_data = calculate_mean(df, value_column)

        if with_moving_avg:
            moving_avg = mean_data.rolling(window=moving_window, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=mean_data.index, y=mean_data.values, mode='lines', name=y_axis_label))
            fig.add_trace(go.Scatter(x=moving_avg.index, y=moving_avg.values, mode='lines', line=dict(dash='dash'),
                                     name=f'{moving_window}-Day Moving Average'))
        else:
            fig.add_trace(go.Scatter(x=mean_data.index, y=mean_data.values, mode='lines', name=y_axis_label))

        if is_bar:
            fig.add_trace(go.Bar(x=mean_data.index, y=mean_data.values, name=y_axis_label))

    # Function to plot individual station trends
    def plot_station_trends(df, value_column, y_axis_label):
        if selected_stations:  # Only plot individual stations if selected stations exist
            for station in selected_stations:
                station_data = df[df['station_code'] == station]
                fig.add_trace(go.Scatter(x=station_data['measure_date'], y=station_data[value_column], mode='lines',
                                         name=f"{station}"))
                moving_avg = station_data[value_column].rolling(window=moving_window, min_periods=1).mean()
                fig.add_trace(
                    go.Scatter(x=station_data['measure_date'], y=moving_avg, mode='lines', line=dict(dash='dash'),
                               name=f"{station} ({moving_window}-day MA)"))
        else:  # Plot aggregated data if no stations are selected
            plot_trend(df, value_column, y_axis_label, with_moving_avg=True)

    fig = go.Figure()

    # Handle the active button conditions
    if active_button == 'default_button':
        plot_trend(filtered_data, 'air_temp_mean_stations', 'Mean Temperature [°C]', with_moving_avg=False)
        plot_trend(filtered_data, 'wind_speed_max_stations', 'Mean Wind Speed [m/s]', with_moving_avg=False)
        plot_trend(filtered_data, 'snow_height_mean_stations', 'Mean Snow Height [cm]', is_bar=True,
                   with_moving_avg=False)
        plot_trend(filtered_data, 'new_snow_mean_stations', 'Mean New Snow [cm]', is_bar=True, with_moving_avg=False)

    elif active_button == 'temp_button':
        plot_station_trends(filtered_data, 'air_temp_mean_stations', "Air Temperature [°C]")
    elif active_button == 'wind_button':
        plot_station_trends(filtered_data, 'wind_speed_max_stations', "Wind Speed [m/s]")
    elif active_button == 'snow_height_button':
        plot_station_trends(filtered_data, 'snow_height_mean_stations', "Height of Snowpack [cm]")
    elif active_button == 'new_snow_button':
        plot_station_trends(filtered_data, 'new_snow_mean_stations', "Height of New Snow [cm]")

    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        template="plotly_white",
        height=200
    )

    return fig


"""
-----------------------------------------------------------------------------------------
Section 2.1: left column - Risk Clustering (PCA)
"""
@app.callback(
    Output('k_cluster_map', 'figure'),
         Input('date_dropdown', 'value')
)
def k_means_clusters(selected_date):
    if selected_date:
        filtered_pca_live_df = pca_live_df[pca_live_df['measure_date'] == selected_date]
    else:
        filtered_pca_live_df = pca_live_df[pca_live_df['measure_date'] == pca_live_df['measure_date'].max()]

    # Plot the K-Means clusters, based on PCA data:
    fig = go.Figure()
    for cluster in pca_training_df['k_cluster'].unique():
        pca_data = pca_training_df[pca_training_df['k_cluster'] == cluster]
        cluster_color = color_map.get(cluster, 'gray')  # Default to gray if not found in color_map

        fig.add_trace(go.Scatter(
            x=pca_data['PCA1'],
            y=pca_data['PCA2'],
            mode='markers',
            marker=dict(size=4, color=cluster_color, opacity=0.7),
            name=f'Risk: {cluster}',
            showlegend=False)
        )

        # Add the live PCA data from the latest update:
        fig.add_trace(go.Scatter(
            x=filtered_pca_live_df['PCA1'],
            y=filtered_pca_live_df['PCA2'],
            mode='markers',
            marker=dict(size=4, color='black', opacity=0.7),
            name='Live Data',
            showlegend=False)
        )

        # Add the cluster centers from k_centers_df
        fig.add_trace(go.Scatter(
            x=k_centers_df['PCA1'],
            y=k_centers_df['PCA2'],
            mode='markers',
            marker=dict(size=8, color='darkblue', symbol='x'),
            name='Cluster Centers',
            showlegend=False)
        )

        # Update layout for better visualization
        fig.update_layout(
            xaxis_title="PCA-1",
            yaxis_title="PCA-2",
            template="plotly_white",
            height=200,
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )

    return fig
"""
-----------------------------------------------------------------------------------------
Section 2.2: left column - Risk Assessment Matrix
"""

@app.callback(
    Output('cluster_matrix', 'figure'),
    [Input('region_radio', 'value'),
     Input('canton_radio', 'value'),
     Input('month_slider', 'value')]
)
def update_cluster_matrix(selected_region, selected_canton, selected_month_idx):
    filtered_data = pca_live_df.copy()

    if not np.issubdtype(filtered_data['measure_date'].dtype, np.datetime64):
        filtered_data['measure_date'] = pd.to_datetime(filtered_data['measure_date'], errors='coerce')

    if selected_region and selected_region != 'Entire Alpine Region':
        valid_regions = all_regions.get(selected_region, [])
        filtered_data = filtered_data[filtered_data['canton_code'].isin(valid_regions)]

    if selected_canton and selected_canton != 'None':
        filtered_data = filtered_data[filtered_data['canton_code'] == selected_canton]
        group_by_column = 'station_code'
    else:
        group_by_column = 'canton_code'

    if selected_month_idx is not None:
        selected_month = month_mapping[selected_month_idx]
        filtered_data = filtered_data[filtered_data['measure_date'].dt.month == month_mapping[selected_month_idx]]

    # Create a matrix of k_cluster values for each station and measure date
    aggregated_data = filtered_data.groupby([group_by_column, 'measure_date'])['k_cluster'].mean().reset_index()
    y_values = aggregated_data[group_by_column].unique()
    measure_dates = aggregated_data['measure_date'].unique()
    y_idx_map = {value: idx for idx, value in enumerate(y_values)}
    date_idx_map = {date: idx for idx, date in enumerate(measure_dates)}

    # Add Values:
    cluster_matrix = np.full((len(y_values), len(measure_dates)), np.nan)
    for _, row in aggregated_data.iterrows():
        i = y_idx_map[row[group_by_column]]
        j = date_idx_map[row['measure_date']]
        cluster_matrix[i, j] = row['k_cluster']

    # Umwandlung der Cluster-Matrix in Farbcodes, mit Bedingung für NaN-Werte:
    def get_color(cluster_value):
        if np.isnan(cluster_value):
            return 'white'
        else:
            return color_map.get(int(cluster_value), 'white')

    color_matrix = np.vectorize(get_color)(cluster_matrix)

    fig = go.Figure(data=go.Heatmap(
        z=cluster_matrix,
        x=measure_dates,
        y=y_values,
        colorscale=[[0, 'white'], [1, 'white']],
        showscale=False,
        hovertemplate="Cluster: %{z}",
    ))

    for i in range(len(y_values)):
        for j in range(len(measure_dates)):
            if color_matrix[i, j] != 'white':
                fig.add_trace(go.Scatter(
                    x=[measure_dates[j]], y=[y_values[i]], mode='markers',
                    marker=dict(size=8, color=color_matrix[i, j], opacity=1),
                    showlegend=False
                ))

    # Add custom legend:
    sorted_clusters = [
        (3, "1 - very low risk", "lightgreen"),
        (2, "2 - low risk", "yellow"),
        (0, "3 - moderate risk", "orange"),
        (1, "4 - considerable risk", "lightcoral"),
        (4, "5 - high risk", "darkred")
    ]

    for k_value, label, color in sorted_clusters:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color),
            name=label
        ))

    fig.update_layout(
        template="plotly_white",
        height=200,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    return fig



"""
-----------------------------------------------------------------------------------------
Section 3.1: right column - Training Map
"""
@app.callback(
    Output('training_geomap', 'figure'),
    [Input('altitude', 'value')]
)
def imis_accident_map(altitude):
    if altitude:
        min_altitude, max_altitude = altitude
        filtered_acc_df = acc_df[(acc_df['start_zone_elevation'] >= min_altitude) & (acc_df['start_zone_elevation'] <= max_altitude)]
    else:
        filtered_acc_df = acc_df

    lat_acc = filtered_acc_df['start_zone_coordinates_latitude']
    long_acc = filtered_acc_df['start_zone_coordinates_longitude']

    # Plot accident locations:
    fig = go.Figure(go.Densitymap(
        lat=lat_acc,
        lon=long_acc,
        z=[1] * len(lat_acc),  # value to calculate the density, adjustable
        radius=15,  # radius of the density heatmap, adjustable
        colorscale="YlOrRd",  # Color scale: Yellow to Red
        opacity=0.6)
    )

    fig.update_layout(
        map_style="light",
        map_zoom=6.4,  # Adjust zoom level
        map_center={"lat": lat_acc.mean(), "lon": long_acc.mean()},  # Center map on the average location
        margin={"r":0,"t":0,"l":0,"b":0}, # Remove margins around the map
        height=300
    )
    return fig

"""
-----------------------------------------------------------------------------------------
Section 3.2: right column - Training Data 
"""


@app.callback(
    Output('accidents_stats', 'figure'),
    [Input('default_button', 'n_clicks'),
     Input('temp_button', 'n_clicks'),
     Input('wind_button', 'n_clicks'),
     Input('new_snow_button', 'n_clicks'),
     Input('snow_height_button', 'n_clicks')]
)
def accidents_stats(default_button, temp_button, wind_button, new_snow_button, snow_height_button):
    # Determine which button was last clicked, or use 'default_button' if none was clicked:
    ctx = callback_context
    if not ctx.triggered:
        button_id = 'default_button'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Filter data by altitude -> to be implemented:
    filtered_acc_df = acc_df

    # Convert dates to datetime and add year and month columns:
    filtered_acc_df['date'] = pd.to_datetime(filtered_acc_df['date'])
    filtered_acc_df['year'] = filtered_acc_df['date'].dt.year
    filtered_acc_df['month'] = filtered_acc_df['date'].dt.month

    # Define winter months and corresponding names:
    winter_months = [11, 12, 1, 2, 3, 4]
    month_names = {11: 'November', 12: 'December', 1: 'January', 2: 'February', 3: 'March', 4: 'April'}

    # Create subplot structure for two small multiples:
    fig = make_subplots(rows=3, cols=3, subplot_titles=[month_names[m] for m in winter_months], vertical_spacing=0.05, horizontal_spacing=0.05)

    # Set the font size for subplot titles specifically
    for i, month in enumerate(winter_months):
        fig['layout']['annotations'][i].update(font=dict(size=8))

    # Check which plot to display based on the button clicked
    if button_id == 'temp_button':
        # Temperature histogram by month:
        for idx, month in enumerate(winter_months):
            monthly_data = filtered_acc_df[filtered_acc_df['month'] == month]
            row, col = (idx // 3) + 1, (idx % 3) + 1
            fig.add_trace(
                go.Histogram(x=monthly_data['air_temp_mean_stations'], nbinsx=20, marker_color='blue'),
                row=row, col=col
            )
            # Set y-axis range for the subplot
            fig.update_yaxes(range=[0, 100], row=row, col=col)

    elif button_id == 'wind_button':
        # Wind speed histogram by month:
        for idx, month in enumerate(winter_months):
            monthly_data = filtered_acc_df[filtered_acc_df['month'] == month]
            row, col = (idx // 3) + 1, (idx % 3) + 1
            fig.add_trace(
                go.Histogram(x=monthly_data['wind_speed_mean_stations'], nbinsx=20, marker_color='green'),
                row=row, col=col
            )
            # Set axis range for the subplot
            fig.update_yaxes(range=[0, 200], row=row, col=col)
            #fig.update_xaxes(range=[0, 10], row=row, col=col)

    elif button_id == 'snow_height_button':
        # Snow height histogram by month:
        for idx, month in enumerate(winter_months):
            monthly_data = filtered_acc_df[filtered_acc_df['month'] == month]
            row, col = (idx // 3) + 1, (idx % 3) + 1
            fig.add_trace(
                go.Histogram(x=monthly_data['snow_height_mean_stations'], nbinsx=20, marker_color='orange'),
                row=row, col=col
            )
            # Set y-axis range for the subplot
            #fig.update_yaxes(range=[0, 100], row=row, col=col)

    elif button_id == 'new_snow_button':
        # New snow histogram by month:
        for idx, month in enumerate(winter_months):
            monthly_data = filtered_acc_df[filtered_acc_df['month'] == month]
            row, col = (idx // 3) + 1, (idx % 3) + 1
            fig.add_trace(
                go.Histogram(x=monthly_data['new_snow_mean_stations'], nbinsx=20, marker_color='red'),
                row=row, col=col
            )
            # Set y-axis range for the subplot
            #fig.update_yaxes(range=[0, 100], row=row, col=col)

    elif button_id == 'default_button':
        # Plot accident counts as bar charts (default):
        for idx, month in enumerate(winter_months):
            monthly_data = filtered_acc_df[filtered_acc_df['month'] == month]
            yearly_counts = monthly_data.groupby('year').size()

            # Calculate row and column for the subplot
            row = (idx // 3) + 1
            col = (idx % 3) + 1

            # Add bar chart to the respective subplot
            fig.add_trace(
                go.Bar(
                    x=yearly_counts.index,
                    y=yearly_counts.values,
                    name=month_names[month],
                    marker_color='darkblue'
                ),
                row=row, col=col
            )
            # Set y-axis range for the subplot
            fig.update_yaxes(range=[0, 80], row=row, col=col)

    # Update layout
    fig.update_layout(
        #title='Number of accidents by year for each winter month',
        showlegend=False,
        height=500,
        margin=dict(t=15, b=0, l=0, r=0),
        font=dict(size=8)
    )

    return fig




if __name__ == '__main__':
    app.run_server(debug=True)