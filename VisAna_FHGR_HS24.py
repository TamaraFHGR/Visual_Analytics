from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import os
from datetime import datetime
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Data_Loader import (load_hist_data, load_imis_stations, load_measurements, load_snow, load_combined,
                         load_measurements_trend, load_snow_trend,
                         load_kmeans_training, load_pca_training, load_pca_live, load_kmeans_centers)

# Load the data:
acc_df = load_hist_data()
imis_df = load_imis_stations()
daily_weather_df = load_measurements()
daily_snow_df = load_snow()
daily_combined_df = load_combined()
trend_measure_df = load_measurements_trend()
trend_snow_df = load_snow_trend()
k_means_training_df = load_kmeans_training()
pca_training_df = load_pca_training()
pca_live_df = load_pca_live()
k_centers_df = load_kmeans_centers()

# Define color map for the K-Means clusters:
color_map = {0: 'orange', #3 - moderate risk
             1: 'lightcoral', #4 - considerable risk
             2: 'yellow', #2 - low risk
             3: 'lightgreen', #1 - very low risk
             4: 'darkred'} #5 - high risk

last_update = datetime.fromtimestamp(os.path.getmtime('assets/API/daily/05_SLF_daily_imis_snow_clean.csv')).strftime('%Y-%m-%d-%H:%M')

# Combined filter values for IMIS stations:
imis_df['combined'] = (
    imis_df['canton_code'].fillna('') + ' - ' +
    imis_df['code'].fillna('') + ' - ' +
    imis_df['station_name'].fillna('') + ' - ' +
    imis_df['elevation'].fillna(0).astype(int).astype(str) + ' m'
)

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
                        options=[{'label': combined, 'value': code}
                                 for combined, code in zip(imis_df['combined'], imis_df['code'])],
                        multi=True,
                        placeholder="Select IMIS-Stations...",
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
                    dcc.Dropdown(id='moving_avg_window',
                                 options=[
                                    {'label': '3 Days', 'value': 3},
                                    {'label': '5 Days', 'value': 5},
                                    {'label': '7 Days', 'value': 7},
                                    {'label': '10 Days', 'value': 10}],
                                 value=7,
                                 clearable=False),
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


"""
-----------------------------------------------------------------------------------------
Section 1: Training Data Visualization -> Geo Map (right column)
"""

# Callback to update the training data map based on the selected altitude range
@app.callback(
    Output('training_geomap', 'figure'),
    [Input('altitude', 'value')]
)
def imis_accident_map(altitude):
    # filter data by altitude:
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
Section 2: Training Data Visualization -> Accident Statistics (right column)
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

"""
-----------------------------------------------------------------------------------------
Section 3: Live Weather Data Visualization -> Geo Map (left column)
"""

# Callback to open the station document in a new tab
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

@app.callback(
    Output('live_geomap', 'figure'),
    [Input('default_button', 'n_clicks'),
     Input('temp_button', 'n_clicks'),
     Input('wind_button', 'n_clicks'),
     Input('new_snow_button', 'n_clicks'),
     Input('snow_height_button', 'n_clicks')],
)
def imis_live_map(default_button, temp_button, wind_button, new_snow_button, snow_height_button):
    # URL for the IMIS station Details:
    imis_df['data_url'] = imis_df['code'].apply(lambda code: f"https://stationdocu.slf.ch/pdf/IMIS_{code}_DE.pdf")
    imis_df['hover_text'] = imis_df.apply(
        lambda row: f"{row['code']} - {row['station_name']}<br>"
                    f"Elevation: {row['elevation']} m<br>"
                    f"(click to open data sheet)", axis=1
    )

    # Determine which button was last clicked, or use 'default_button' if none was clicked:
    ctx = callback_context
    button_id = 'default_button'  # Default fallback

    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger in ['default_button', 'temp_button', 'wind_button', 'new_snow_button', 'snow_height_button']:
            button_id = trigger

    lat_imis = imis_df['lat']
    long_imis = imis_df['lon']
    day_weather = daily_weather_df.merge(imis_df[['code', 'lon', 'lat']], left_on='code', right_on='code')
    day_snow = daily_snow_df.merge(imis_df[['code', 'lon', 'lat']], left_on='code', right_on='code')

    # Plot IMIS locations:
    if button_id == 'default_button':
        fig = go.Figure(go.Scattermap(
            lat=lat_imis,
            lon=long_imis,
            mode='markers',
            marker=go.scattermap.Marker(
                size=12,
                symbol='mountain',
                color='black'),
            text=imis_df['hover_text'],
            hoverinfo='text',
            hovertemplate="<span class='hover_cursor'>%{text}</span>",
            customdata=imis_df['data_url'])
        )

    # Add heatmap for temperature:
    elif button_id == 'temp_button':
        fig = go.Figure(go.Densitymap(
                lat=day_weather['lat'],
                lon=day_weather['lon'],
                z=day_weather['air_temp_daily_mean'],
                radius=30,  # radius of the density heatmap, adjustable
                colorscale='blugrn',
                reversescale=True)
            )

    # Add heatmap for wind:
    elif button_id == 'wind_button':
        fig = go.Figure(go.Densitymap(
            lat=day_weather['lat'],
            lon=day_weather['lon'],
            z=day_weather['wind_speed_daily_mean'],
            radius=30,  # radius of the density heatmap, adjustable
            colorscale='brwnyl')
        )

    # Add heatmap for snow height:
    elif button_id == 'snow_height_button':
        fig = go.Figure(go.Densitymap(
            lat=day_snow['lat'],
            lon=day_snow['lon'],
            z=day_snow['snow_height_daily_mean'],
            radius=30,  # radius of the density heatmap, adjustable
            colorscale='blues')
        )

    # Add heatmap for new snow:
    elif button_id == 'new_snow_button':
        fig = go.Figure(go.Densitymap(
            lat=day_weather['lat'],
            lon=day_weather['lon'],
            z=day_snow['new_snow_daily_mean'],
            radius=30,  # radius of the density heatmap, adjustable
            colorscale='dense',
            showscale=True)
        )

    fig.update_layout(
        map_style="light",
        map_zoom=6,  # Adjust zoom level
        map_center={"lat": lat_imis.mean(), "lon": long_imis.mean()},  # Center map on the average location
        margin={"r":0,"t":0,"l":0,"b":0}, # Remove margins around the map
        height=270
    )
    return fig



"""
-----------------------------------------------------------------------------------------
Section 4: Live Weather Data Visualization -> Trend Analysis (left column)
"""
@app.callback(
    Output('trend_analysis', 'figure'),
    [Input('default_button', 'n_clicks'),
     Input('temp_button', 'n_clicks'),
     Input('wind_button', 'n_clicks'),
     Input('new_snow_button', 'n_clicks'),
     Input('snow_height_button', 'n_clicks'),
     Input('station_dropdown', 'value'),
     Input('moving_avg_window', 'value')]
)
def weather_trend(default_button, temp_button, wind_button, new_snow_button, snow_height_button, selected_stations,window_size):
    # Determine which button was last clicked
    ctx = callback_context
    button_id = 'default_button'  # Default fallback

    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    fig = go.Figure()

    # Function to get mean data for the selected or all stations
    def calculate_mean(df, value_column):
        if selected_stations:  # If stations are selected, filter by these stations
            df = df[df['code'].isin(selected_stations)]
        return df.groupby('date')[value_column].mean()

    # Plot for each station
    def plot_per_station(df, value_column, y_axis_label):
        if not selected_stations:  # If no stations are selected, calculate mean
            mean_data = calculate_mean(df, value_column)
            moving_avg = mean_data.rolling(window=window_size, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=mean_data.index,
                y=mean_data.values,
                mode='lines',
                name='Mean Value'
            ))
            fig.add_trace(go.Scatter(
                x=moving_avg.index,
                y=moving_avg.values,
                mode='lines',
                line=dict(dash='dash'),
                name='Moving Average'
            ))
        else:  # Plot data for selected stations
            for station in selected_stations:
                station_data = df[df['code'] == station]
                fig.add_trace(go.Scatter(
                    x=station_data['date'],
                    y=station_data[value_column],
                    mode='lines',
                    name=f"{station}"
                ))
                # Add moving average line
                station_data_sorted = station_data.sort_values(by='date')
                moving_avg = station_data_sorted[value_column].rolling(window=window_size, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=station_data_sorted['date'],
                    y=moving_avg,
                    mode='lines',
                    line=dict(dash='dash'),
                    name=f"{station} ({window_size}-day MA)"
                ))
        fig.update_layout(yaxis_title=y_axis_label)

    # Plot for the default button
    if button_id == 'default_button':
        temp_data = calculate_mean(trend_measure_df, 'air_temp_daily_mean')
        wind_data = calculate_mean(trend_measure_df, 'wind_speed_daily_mean')
        snow_height_data = calculate_mean(trend_snow_df, 'snow_height_daily_mean')

        # Add line plot for mean temperature
        fig.add_trace(go.Scatter(
            x=temp_data.index,
            y=temp_data.values,
            mode='lines',
            name='Mean Temperature'
        ))

        # Add line plot for mean wind speed
        fig.add_trace(go.Scatter(
            x=wind_data.index,
            y=wind_data.values,
            mode='lines',
            name='Mean Wind Speed'
        ))

        # Add bar chart for mean snow height
        fig.add_trace(go.Bar(
            x=snow_height_data.index,
            y=snow_height_data.values,
            name='Mean Snow Height'
        ))

    # Plot for temperature button
    elif button_id == 'temp_button':
        plot_per_station(trend_measure_df, 'air_temp_daily_mean', "Air Temperature [°C]")

    # Plot for wind speed button
    elif button_id == 'wind_button':
        plot_per_station(trend_measure_df, 'wind_speed_daily_mean', "Wind Speed [m/s]")

    # Plot for snow height button
    elif button_id == 'snow_height_button':
        plot_per_station(trend_snow_df, 'snow_height_daily_mean', "Height of Snowpack [cm]")

    # Plot for new snow button
    elif button_id == 'new_snow_button':
        plot_per_station(trend_snow_df, 'new_snow_daily_mean', "Height of New Snow [cm]")

    # Update layout
    fig.update_layout(
        xaxis_title="Measure Date",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=300,
        legend_title="Station Code"
    )

    return fig

"""
-----------------------------------------------------------------------------------------
Section 5: Live Weather Data Visualization -> PCA Data (left column)
"""
@app.callback(
    Output('k_cluster_map', 'figure'),
        [Input('default_button', 'n_clicks'),
         Input('date_dropdown', 'value')],
         [State('default_button', 'n_clicks')]
)
def k_means_clusters(default_button, selected_date, last_default):
    # Determine which button was last clicked
    ctx = callback_context
    button_id = 'default_button'  # Default fallback

    if ctx.triggered:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger in ['default_button']:
            button_id = trigger
        else:
        # No new button click; retain previous button
            button_id = 'default_button'

    # Create a scatter plot for the K-Means clusters
    fig = go.Figure()

    # Filter the data based on the selected date:
    if selected_date:
        filtered_pca_live_df = pca_live_df[pca_live_df['measure_date'] == selected_date]
    else:
        #take max date from the data:
        filtered_pca_live_df = pca_live_df[pca_live_df['measure_date'] == pca_live_df['measure_date'].max()]

    if button_id == 'default_button':
        # Plot the K-Means clusters, based on PCA data:
        for cluster in pca_training_df['risk'].unique():
            pca_data = pca_training_df[pca_training_df['risk'] == cluster]
            fig.add_trace(go.Scatter(
                x=pca_data['PCA1'],
                y=pca_data['PCA2'],
                mode='markers',
                marker=dict(size=5),
                name=f'Risk: {cluster}')
            )

        # Add the live PCA data from the latest update:
        fig.add_trace(go.Scatter(
            x=filtered_pca_live_df['PCA1'],
            y=filtered_pca_live_df['PCA2'],
            mode='markers',
            marker=dict(size=5, color='black'),
            name='Live Data')
        )

        # Add the cluster centers from k_centers_df
        fig.add_trace(go.Scatter(
            x=k_centers_df['PCA1'],
            y=k_centers_df['PCA2'],
            mode='markers',
            marker=dict(size=12, color='black', symbol='x'),
            name='Cluster Centers')
        )

        # Update layout for better visualization
        fig.update_layout(
            xaxis_title="Principal Component 1 (PCA1)",
            yaxis_title="Principal Component 2 (PCA2)",
            legend_title="Cluster Risk Levels",
            template="plotly_white",
            height=300,
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )

    return fig
"""
-----------------------------------------------------------------------------------------
Section 6: Live Weather Data Visualization -> Cluster Matrix (left column)
"""

@app.callback(
    Output('cluster_matrix', 'figure'),
    [Input('region_dropdown', 'value')]
)
def update_cluster_matrix(selected_region):
    # Filtere Daten nach ausgewählter Region
    filtered_pca_live_df = pca_live_df[pca_live_df['alpine_region'] == selected_region]

    # Erstelle eine Liste aller eindeutigen Stationscodes und Messdaten
    station_codes = filtered_pca_live_df['station_code'].unique()
    measure_dates = filtered_pca_live_df['measure_date'].unique()

    # Initialisiere eine leere Matrix mit NaN-Werten
    cluster_matrix = np.full((len(station_codes), len(measure_dates)), np.nan)

    # Mapping von Stationscodes und Messdaten zu Indizes
    station_idx_map = {code: idx for idx, code in enumerate(station_codes)}
    date_idx_map = {date: idx for idx, date in enumerate(measure_dates)}

    # Fülle die Matrix mit K-Cluster-Werten
    for _, row in filtered_pca_live_df.iterrows():
        i = station_idx_map[row['station_code']]
        j = date_idx_map[row['measure_date']]
        cluster_matrix[i, j] = row['k_cluster']

    # Erstelle eine Liste der Farben basierend auf der K-Cluster-Zuordnung
    color_matrix = np.empty(cluster_matrix.shape, dtype=object)
    for i in range(len(station_codes)):
        for j in range(len(measure_dates)):
            cluster_value = cluster_matrix[i, j]
            if np.isnan(cluster_value):
                color_matrix[i, j] = 'white'  # Setze fehlende Werte auf weiß
            else:
                # Sicherstellen, dass der k_cluster-Wert im color_map existiert
                color_matrix[i, j] = color_map.get(int(cluster_value), 'white')  # 'white' als Fallback für unbekannte Werte

    # Erstelle das Heatmap-Plot mit benutzerdefinierten Farben
    fig = go.Figure(data=go.Heatmap(
        z=cluster_matrix,  # Numerische Werte für Heatmap
        x=measure_dates,
        y=station_codes,
        colorscale=[[0, 'white'], [1, 'white']],  # Erstellt eine Farbskala basierend auf der festen Zuordnung
        showscale=False,  # Keine Farblegende anzeigen
        hovertemplate="Cluster: %{z}",
    ))

    # Hinzufügen eines Scatters für die benutzerdefinierte Farbzuordnung
    for i in range(len(station_codes)):
        for j in range(len(measure_dates)):
            if color_matrix[i, j] != 'white':  # Nur hinzufügen, wenn es eine gültige Farbe gibt
                fig.add_trace(go.Scatter(
                    x=[measure_dates[j]], y=[station_codes[i]], mode='markers',
                    marker=dict(size=10, color=color_matrix[i, j], opacity=1),
                    showlegend=False  # Nur Marker ohne Legende
                ))

    # Benutzerdefinierte Legende hinzufügen
    for k_value, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color),
            name=f'{k_value}: {color}'
        ))

    # Layout konfigurieren
    fig.update_layout(
        xaxis_title="Measure Date",
        yaxis_title="Station Code",
        template="plotly_white",
        height=300,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)