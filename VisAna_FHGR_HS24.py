from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import os
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Data_Loader import (load_imis_stations, load_hist_data, load_daily,
                         load_kmeans_centers, load_pca_training, load_pca_live, load_kmeans_training)
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
kmeans_df = load_kmeans_training()
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

last_update = datetime.fromtimestamp(os.path.getmtime('assets/API/daily/05_SLF_daily_imis_snow_clean.csv')).strftime('%Y-%m-%d-%H:%M')
latest_date = pd.to_datetime(daily_snow_df['measure_date'].max())
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
                html.Div([ # dropdown_group
                html.Span([ # station_dropdown
                    dcc.Dropdown(
                        id='station_dropdown',
                        options=[{'label': combined, 'value': code}
                                 for combined, code in zip(imis_df['combined'], imis_df['code'])],
                        multi=True,
                        placeholder="Select any weather station... (ideally not more than 3)",
                        value=[])
                    ], className='station_dropdown'),
                html.Span(html.H5('Date range for Trend Analysis:')),
                html.Span([ # date_range
                    dcc.DatePickerRange(
                        id='date_range',
                        min_date_allowed=daily_snow_df['measure_date'].min(),
                        max_date_allowed=daily_snow_df['measure_date'].max(),
                        initial_visible_month=date.today() - timedelta(days=1),
                        start_date=pd.to_datetime(daily_snow_df['measure_date']).max() - timedelta(days=21),
                        end_date=daily_snow_df['measure_date'].max(),
                        display_format='DD-MM-YYYY')
                    ], className='date_range'),
                ], className='dropdown_group'),
            ], className='sub_header'),
            # Live Weather Data and Trend:
            html.Div([ # weather_row
                html.Div([ # map_graph
                    html.H2(['Snow and Weather Conditions on ',
                    html.Span(latest_date.strftime("%d-%b-%Y"),
                           style={'color': 'blue'})]),
                    dcc.Graph(id='live_geomap')
                ], className='map_graph'),
                html.Div([ # trend_graph
                    html.H2('Trend Analysis'),
                    dcc.Graph(id='trend_analysis'),
                    html.Div(
                    dcc.Slider(
                        id='moving_avg_window',
                        min=3,
                        max=9,
                        step=None,
                        value=7,
                        marks={3: '3 Days', 5: '5 Days', 7: '7 Days', 9: '9 Days'},
                        tooltip={"placement": "bottom", "always_visible": False}),
                    id='slider_container')
                ], className='trend_graph')
            ], className='weather_row'),

            # Risk Assessment Clusters and Matrix:
            html.Div([
                html.Div([ # k_cluster_scatter
                    html.Div([
                    html.H2('Daily Risk Clustering'),
                    dcc.DatePickerSingle(
                        id='date_dropdown',
                        min_date_allowed=date(2024, 10, 5),
                        max_date_allowed=pca_live_df['measure_date'].max(),
                        initial_visible_month=date.today() - timedelta(days=1),
                        date=pca_live_df['measure_date'].max(),
                        display_format='DD-MM-YYYY')
                        ], className='date_dropdown'), # flexbox
                    dcc.Graph(id='k_cluster_map'),
                    ], className='k_cluster_scatter'),
                html.Div([ # k_cluster_matrix
                    html.H2('Risk Assessment Matrix'),
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
                html.Div([ # feature_dropdown
                    html.H2('Historical Avalanche Accidents'),
                    dcc.Dropdown(
                            id='feature_dropdown',
                            options=['Elevation','Snow Height','New Snow','Temperature','Wind'],
                            multi=False,
                            clearable=False,
                            value='Elevation'),], className='feature_dropdown'),
                    dcc.Graph(id='training_geomap'),
                dcc.Slider(
                    id='year_slider',
                    min=1998,
                    max=2023,
                    step=None,
                    value=2021,
                    marks={
                        year: {'label': str(year), 'style': {'transform': 'rotate(45deg)', 'whiteSpace': 'nowrap'}}
                        for year in range(1998, 2024)
                    },
                    tooltip={"placement": "bottom", "always_visible": False}),
                html.Span([ # region_dropdown
                    dcc.Dropdown(
                        id='month_dropdown',
                        options=[
                            {'label': 'January', 'value': 1},
                            {'label': 'February', 'value': 2},
                            {'label': 'March', 'value': 3},
                            {'label': 'April', 'value': 4},
                            {'label': 'November', 'value': 11},
                            {'label': 'December', 'value': 12}],
                            multi=False,
                            placeholder="Select a specific Month..."),

                    ], className='month_dropdown')
            ], className='training_map'),
            html.Div([ # training_data
                dcc.Graph(id='accidents_stats'),
                html.Div(
                    [dcc.Checklist(
                        id='risk_group_checklist',
                        options=[{'label': group, 'value': group} for group in sorted(kmeans_df['risk_group'].unique())],
                        value=['1 - very low', '2 - low', '4 - considerable'],
                        inline=True)
                    ], className='risk_group_checklist')
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

# Update Move_Avg_Window Sliders
@app.callback(
    Output('slider_container', 'style'),
    Input('default_button', 'className')
)
def toggle_slider_visibility(default_button_class):
    # Wenn der Default Button aktiv ist, wird der Slider ausgeblendet, d.h. wenn die className = 'active' ist:
    if default_button_class == 'active':
        return {'display': 'none'}
    return {}


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
            map_zoom=5.7,
            map_center={"lat": filtered_data['lat'].mean(), "lon": filtered_data['lon'].mean()},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=200
        )

    # Plot Heat Map "Temperature":
    elif active_button == 'temp_button':
        daily_snow = daily_snow_df[daily_snow_df['measure_date'] == daily_snow_df['measure_date'].max()]
        fig = go.Figure(go.Densitymap(
            lat=daily_snow['lat'],
            lon=daily_snow['lon'],
            z=daily_snow['air_temp_mean_stations'],
            radius=25,
            colorscale='blugrn',
            reversescale=True)
        )
        fig.update_layout(
            map_style="light",
            map_zoom=5.7,
            map_center={"lat": daily_snow['lat'].mean(), "lon": daily_snow['lon'].mean()},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=200
        )

    # Plot Heat Map "Wind":
    elif active_button == 'wind_button':
        daily_snow = daily_snow_df[daily_snow_df['measure_date'] == daily_snow_df['measure_date'].max()]
        fig = go.Figure(go.Densitymap(
            lat=daily_snow['lat'],
            lon=daily_snow['lon'],
            z=daily_snow['wind_speed_max_stations'],
            radius=25,
            colorscale='brwnyl',
            reversescale=False)
        )
        fig.update_layout(
            map_style="light",
            map_zoom=5.7,
            map_center={"lat": daily_snow['lat'].mean(), "lon": daily_snow['lon'].mean()},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=200
        )

    # Plot Heat Map "Snow Height":
    elif active_button == 'snow_height_button':
        daily_snow = daily_snow_df[daily_snow_df['measure_date'] == daily_snow_df['measure_date'].max()]
        fig = go.Figure(go.Densitymap(
            lat=daily_snow['lat'],
            lon=daily_snow['lon'],
            z=daily_snow['snow_height_mean_stations'],
            radius=25,
            colorscale='blues',
            reversescale=False)
        )
        fig.update_layout(
            map_style="light",
            map_zoom=5.7,
            map_center={"lat": daily_snow['lat'].mean(), "lon": daily_snow['lon'].mean()},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=200
        )

    # Plot Heat Map "New Snow":
    elif active_button == 'new_snow_button':
        daily_snow = daily_snow_df[daily_snow_df['measure_date'] == daily_snow_df['measure_date'].max()]
        fig = go.Figure(go.Densitymap(
            lat=daily_snow['lat'],
            lon=daily_snow['lon'],
            z=daily_snow['new_snow_mean_stations'],
            radius=25,
            colorscale='dense',
            reversescale=False)
        )
        fig.update_layout(
            map_style="light",
            map_zoom=5.7,
            map_center={"lat": daily_snow['lat'].mean(), "lon": daily_snow['lon'].mean()},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=200
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
     Input('moving_avg_window', 'value'),
     Input('date_range', 'start_date'),
     Input('date_range', 'end_date')],
)
def weather_trend(active_button, selected_region, selected_canton, selected_stations, moving_window, start_date, end_date):
    filtered_data = daily_snow_df.copy()
    # Convert start and end date to datetime:
    start_date = pd.to_datetime(start_date) if start_date else None
    end_date = pd.to_datetime(end_date) if end_date else None

    # Filter data based on selected date range:
    if start_date and end_date:
        filtered_data = filtered_data[
            (pd.to_datetime(filtered_data['measure_date']) >= start_date) &
            (pd.to_datetime(filtered_data['measure_date']) <= end_date)
            ]

    # Filter data based on selected region:
    if selected_region and selected_region != 'Entire Alpine Region':
        valid_regions = all_regions.get(selected_region, [])
        filtered_data = filtered_data[filtered_data['canton_code'].isin(valid_regions)]
    # Filter data based on selected canton:
    if selected_canton and selected_canton != 'None':
        filtered_data = filtered_data[filtered_data['canton_code'] == selected_canton]
    # Filter data based on selected stations:
    if selected_stations:
        filtered_data = filtered_data[filtered_data['station_code'].isin(selected_stations)]

    # Function to calculate mean for any given value column:
    def calculate_mean(df, value_column):
        return df.groupby('measure_date')[value_column].mean()

    # Aggregate the data for the plots:
    snow_height_mean = calculate_mean(filtered_data, 'snow_height_mean_stations')
    new_snow_mean = calculate_mean(filtered_data, 'new_snow_mean_stations')
    air_temp_mean = calculate_mean(filtered_data, 'air_temp_mean_stations')
    wind_speed_max = calculate_mean(filtered_data, 'wind_speed_max_stations')

    color_scales = {
        'temp_button': ['#16745C', '#0D4235', '#15BF94', '#9CBFB6'],  # Green tones (blugrn)
        'wind_button': ['#5F3515', '#FF6C00', '#FFC497', '#9CBFB6'],  # Brown tones (brwnyl)
        'snow_height_button': ['#003741', '#2ADDFF', '#0000AB', '#9CBFB6'],  # Blue tones (Blues)
        'new_snow_button': ['#78A8F0', '#601860', '#A8D8D8', '#9CBFB6'],  # Blue-Violet tones (Dense)
    }

    # 1) Plot Trend Analysis for "Station Overview":
    if active_button == 'default_button':
        fig = go.Figure()
        if active_button == 'default_button':
            fig = go.Figure()

            # Add Bar Chart for Snow Height:
            fig.add_trace(go.Bar(
                x=snow_height_mean.index,
                y=snow_height_mean.values,
                name="Ø Snow Height cm",
                marker=dict(color='#0000AB')
            ))

            # Add Stacked Bar Chart for New Snow:
            fig.add_trace(go.Bar(
                x=new_snow_mean.index,
                y=new_snow_mean.values,
                name="Ø New Snow cm",
                marker=dict(color='#78A8F0')
            ))

            # Add Line Plot for Temperature:
            fig.add_trace(go.Scatter(
                x=air_temp_mean.index,
                y=air_temp_mean.values,
                name="Ø Air Temperature °C",
                mode='lines',
                line=dict(color='#15BF94', width=2)
            ))

            # Add Line Plot for Wind:
            fig.add_trace(go.Scatter(
                x=wind_speed_max.index,
                y=wind_speed_max.values,
                name="Ø Wind Speed m/s",
                mode='lines',
                line=dict(color='#FF6C00', width=2)
            ))

            # Update layout
            fig.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                template="plotly_white",
                height=200,
                barmode='stack',
                legend=dict(
                    font=dict(color='gray', size=8, family='Arial, sans-serif'),
                    bgcolor='rgba(0, 0, 0, 0)'),
                showlegend=True,
            )
            fig.update_xaxes(
                autorange=True,
                showgrid=False,
                showticklabels=True,
                tickformat="%d-%b",
                tickfont=dict(color='gray', size=8, family='Arial, sans-serif'))

            fig.update_yaxes(
                autorange=True,
                showgrid=False,
                showticklabels=True,
                tickfont=dict(color='gray', size=8, family='Arial, sans-serif')
            )
            return fig

    # 2) Plot Trend Analysis for "Temperature":
    elif active_button == 'temp_button':
        fig = go.Figure()
        if selected_stations:
            # Plot individual station trends if stations are selected:
            colors = color_scales['temp_button']
            for i, station in enumerate(selected_stations):
                station_data = filtered_data[filtered_data['station_code'] == station]
                fig.add_trace(go.Scatter(
                    x=station_data['measure_date'],
                    y=station_data['air_temp_mean_stations'],
                    name=f"{station}",
                    mode='lines',
                    marker=dict(color=colors[i % len(colors)])
                ))
                fig.add_trace(go.Scatter(
                    x=station_data['measure_date'],
                    y=station_data['air_temp_mean_stations'].rolling(window=moving_window).mean(),
                    name=f"{station} ({moving_window}-day Avg)",
                    mode='lines',
                    line=dict(dash='dot', color=colors[i % len(colors)], width=1),
                ))
        else:  # Global aggregation, if no stations are selected:
            fig.add_trace(go.Scatter(
                x=air_temp_mean.index,
                y=air_temp_mean.values,
                name="Ø Air Temperature °C",
                mode='lines',
                marker=dict(color='#16745C')
            ))
            fig.add_trace(go.Scatter(
                x=air_temp_mean.index,
                y=air_temp_mean.rolling(window=moving_window).mean(),
                name=f"{moving_window}-day Avg",
                mode='lines',
                line=dict(dash='dot', color='#16745C', width=1),
            ))
        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            template="plotly_white",
            height=160,
            legend=dict(
                font=dict(color='gray', size=8, family='Arial, sans-serif'),
                bgcolor='rgba(0, 0, 0, 0)',
                orientation='h',
                yanchor='bottom',
                y=-0.4,
                xanchor='center',
                x=0.5),
            showlegend=True,
        )
        fig.update_xaxes(
                autorange=True,
                showgrid=False,
                showticklabels=True,
                tickformat="%d-%b",
                tickfont=dict(color='gray', size=8, family='Arial, sans-serif'))

        fig.update_yaxes(
            autorange=True,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color='gray', size=8, family='Arial, sans-serif')
        )
        return fig

    # 3) Plot Trend Analysis for "Wind":
    elif active_button == 'wind_button':
        fig = go.Figure()
        if selected_stations:
            # Plot individual station trends if stations are selected:
            colors = color_scales['wind_button']
            for i, station in enumerate(selected_stations):
                station_data = filtered_data[filtered_data['station_code'] == station]
                fig.add_trace(go.Scatter(
                    x=station_data['measure_date'],
                    y=station_data['wind_speed_max_stations'],
                    name=f"{station}",
                    mode='lines',
                    marker=dict(color=colors[i % len(colors)]),
                ))
                fig.add_trace(go.Scatter(
                    x=station_data['measure_date'],
                    y=station_data['wind_speed_max_stations'].rolling(window=moving_window).mean(),
                    name=f"{station} ({moving_window}-day Avg)",
                    mode='lines',
                    line=dict(dash='dot', color=colors[i % len(colors)], width=1),
                ))
        else:  # Global aggregation, if no stations are selected:
            fig.add_trace(go.Scatter(
                x=wind_speed_max.index,
                y=wind_speed_max.values,
                name="Ø Wind Speed m/s",
                mode='lines',
                marker=dict(color='#5F3515'),
            ))
            fig.add_trace(go.Scatter(
                x=wind_speed_max.index,
                y=wind_speed_max.rolling(window=moving_window).mean(),
                name=f"{moving_window}-day Avg",
                mode='lines',
                line=dict(dash='dot', color='#5F3515', width=1)
            ))
        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            template="plotly_white",
            height=160,
            legend=dict(
                font=dict(color='gray', size=8, family='Arial, sans-serif'),
                bgcolor='rgba(0, 0, 0, 0)',
                orientation='h',
                yanchor='bottom',
                y=-0.4,
                xanchor='center',
                x=0.5),
            showlegend=True,
        )
        fig.update_xaxes(
            autorange=True,
            showgrid=False,
            showticklabels=True,
            tickformat="%d-%b",
            tickfont=dict(color='gray', size=8, family='Arial, sans-serif'))

        fig.update_yaxes(
            autorange=True,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color='gray', size=8, family='Arial, sans-serif')
        )
        return fig

    # 4) Plot Trend Analysis for "Snow Height":
    elif active_button == 'snow_height_button':
        fig = go.Figure()
        if selected_stations:
            # Plot individual station trends if stations are selected:
            colors = color_scales['snow_height_button']
            for i, station in enumerate(selected_stations):
                station_data = filtered_data[filtered_data['station_code'] == station]
                fig.add_trace(go.Scatter(
                    x=station_data['measure_date'],
                    y=station_data['snow_height_mean_stations'],
                    name=f"{station}",
                    mode='lines',
                    marker=dict(color=colors[i % len(colors)]),
                ))
                fig.add_trace(go.Scatter(
                    x=station_data['measure_date'],
                    y=station_data['snow_height_mean_stations'].rolling(window=moving_window).mean(),
                    name=f"{station} ({moving_window}-day Avg)",
                    mode='lines',
                    line=dict(dash='dot', color=colors[i % len(colors)], width=1),
                ))
        else:  # Global aggregation, if no stations are selected:
            fig.add_trace(go.Scatter(
                x=snow_height_mean.index,
                y=snow_height_mean.values,
                name="Ø Snow Height cm",
                mode='lines',
                marker=dict(color='#003741'),
            ))
            fig.add_trace(go.Scatter
                          (x=snow_height_mean.index,
                           y=snow_height_mean.rolling(window=moving_window).mean(),
                           name=f"{moving_window}-day Avg",
                           mode='lines',
                           line=dict(dash='dot', color='#003741', width=1),
                           ))

        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            template="plotly_white",
            height=160,
            legend=dict(
                font=dict(color='gray', size=8, family='Arial, sans-serif'),
                bgcolor='rgba(0, 0, 0, 0)',
                orientation='h',
                yanchor='bottom',
                y=-0.4,
                xanchor='center',
                x=0.5),
            showlegend=True,
        )
        fig.update_xaxes(
            autorange=True,
            showgrid=False,
            showticklabels=True,
            tickformat="%d-%b",
            tickfont=dict(color='gray', size=8, family='Arial, sans-serif'))

        fig.update_yaxes(
            autorange=True,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color='gray', size=8, family='Arial, sans-serif')
        )
        return fig

    # 5) Plot Trend Analysis for "New Snow":
    elif active_button == 'new_snow_button':
        fig = go.Figure()
        if selected_stations:
            # Plot individual station trends if stations are selected:
            colors= color_scales['new_snow_button']
            for i, station in enumerate(selected_stations):
                station_data = filtered_data[filtered_data['station_code'] == station]
                fig.add_trace(go.Scatter(
                    x=station_data['measure_date'],
                    y=station_data['new_snow_mean_stations'],
                    name=f"{station}",
                    mode='lines',
                    marker=dict(color=colors[i % len(colors)]),
                ))
                fig.add_trace(go.Scatter(
                    x=station_data['measure_date'],
                    y=station_data['new_snow_mean_stations'].rolling(window=moving_window).mean(),
                    name=f"{station} ({moving_window}-day Avg)",
                    mode='lines',
                    line=dict(dash='dot', color=colors[i % len(colors)], width=1),
                ))
        else:  # Global aggregation, if no stations are selected:
            fig.add_trace(go.Scatter(
                x=new_snow_mean.index,
                y=new_snow_mean.values,
                name="Ø New Snow cm",
                mode='lines',
                marker=dict(color='#78A8F0'),
            ))
            fig.add_trace(go.Scatter
                            (x=new_snow_mean.index,
                             y=new_snow_mean.rolling(window=moving_window).mean(),
                             name=f"{moving_window}-day Avg",
                             mode='lines',
                             line=dict(dash='dot', color='#78A8F0', width=1),
                             ))
        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            template="plotly_white",
            height=160,
            legend=dict(
                font=dict(color='gray', size=8, family='Arial, sans-serif'),
                bgcolor='rgba(0, 0, 0, 0)',
                orientation='h',
                yanchor='bottom',
                y=-0.4,
                xanchor='center',
                x=0.5),
            showlegend=True,
        )
        fig.update_xaxes(
            autorange=True,
            showgrid=False,
            showticklabels=True,
            tickformat="%d-%b",
            tickfont=dict(color='gray', size=8, family='Arial, sans-serif'))

        fig.update_yaxes(
            autorange=True,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color='gray', size=8, family='Arial, sans-serif')
        )
        return fig

"""
-----------------------------------------------------------------------------------------
Section 2.1: left column - Risk Clustering (PCA)
"""
@app.callback(
    Output('k_cluster_map', 'figure'),
         Input('date_dropdown', 'date')
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
            xaxis_title={
                'text': 'PCA-1',
                'font': {
                    'size': 8,
                    'family': 'Arial, sans-serif',
                    'color': 'gray'}
                },
            yaxis_title={
                'text': 'PCA-2',
                'font': {
                    'size': 8,
                    'family': 'Arial, sans-serif',
                    'color': 'gray'}
                },
            template="plotly_white",
            height=190,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )

        fig.update_xaxes(
            autorange=True,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color='gray', size=8, family='Arial, sans-serif'))

        fig.update_yaxes(
            autorange=True,
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color='gray', size=8, family='Arial, sans-serif')
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
    y_values = sorted(aggregated_data[group_by_column].unique(), reverse=True)
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
        (3, "very low risk", "lightgreen"),
        (2, "low risk", "yellow"),
        (0, "moderate risk", "orange"),
        (1, "considerable risk", "lightcoral"),
        (4, "high risk", "darkred")
    ]

    for k_value, label, color in sorted_clusters:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=6, color=color),
            name=label
        ))

    fig.update_layout(
        template="plotly_white",
        height=190,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend = dict(
            font=dict(color='gray', size=8, family='Arial, sans-serif'),
            bgcolor='rgba(0, 0, 0, 0)',
            orientation='h',
            yanchor='bottom',
            y=-0.6,
            xanchor='center',
            x=0.5),
        showlegend = True,
    )

    fig.update_xaxes(
        autorange=True,
        showgrid=False,
        showticklabels=True,
        tickformat="%d-%b",
        tickfont=dict(color='gray', size=8, family='Arial, sans-serif'))

    fig.update_yaxes(
        autorange=True,
        showgrid=False,
        showticklabels=True,
        tickfont=dict(color='gray', size=8, family='Arial, sans-serif')
    )

    return fig

"""
-----------------------------------------------------------------------------------------
Section 3.1: right column - Training Heat Map
"""
@app.callback(
    Output('training_geomap', 'figure'),
    [Input('year_slider', 'value'),
     Input('feature_dropdown', 'value')]
)
def imis_accident_map(selected_year, selected_feature):
    filtered_df = acc_df.copy()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

    filtered_df['month'] = pd.to_datetime(filtered_df['date']).dt.month
    filtered_df = filtered_df[filtered_df['year'] == selected_year]

    if selected_feature == 'Elevation':
        filtered_df['rounded_elevation'] = (filtered_df['start_zone_elevation'] // 100) * 100
        heatmap_data = filtered_df.groupby(['rounded_elevation', 'month']).size().unstack(fill_value=0)
        y_axis_label = 'Elevation  [m]'
        y_axis_values = heatmap_data.index

    elif selected_feature == 'Snow Height':
        filtered_df['snow_height_mean_stations'] = (filtered_df['snow_height_mean_stations'] // 10) * 10
        heatmap_data = filtered_df.groupby(['snow_height_mean_stations', 'month']).size().unstack(fill_value=0)
        y_axis_label = 'Snow Height  [cm]'
        y_axis_values = heatmap_data.index

    elif selected_feature == 'New Snow':
        filtered_df['new_snow_mean_stations'] = (filtered_df['new_snow_mean_stations'] // 5) * 5
        heatmap_data = filtered_df.groupby(['new_snow_mean_stations', 'month']).size().unstack(fill_value=0)
        y_axis_label = 'New Snow  [cm]'
        y_axis_values = heatmap_data.index

    elif selected_feature == 'Temperature':
        filtered_df['air_temp_mean_stations'] = (filtered_df['air_temp_mean_stations'] // 5) * 5
        heatmap_data = filtered_df.groupby(['air_temp_mean_stations', 'month']).size().unstack(fill_value=0)
        y_axis_label = 'Temperature  [°C]'
        y_axis_values = heatmap_data.index

    elif selected_feature == 'Wind':
        filtered_df['wind_speed_max_stations'] = (filtered_df['wind_speed_max_stations'] // 5) * 5
        heatmap_data = filtered_df.groupby(['wind_speed_max_stations', 'month']).size().unstack(fill_value=0)
        y_axis_label = 'Wind Speed  [m/s]'
        y_axis_values = heatmap_data.index

    # Reindex to ensure all months are represented, even if no data is available
    heatmap_data = heatmap_data.reindex(columns=range(1, 13), fill_value=0)

    # Create the Heatmap figure
    fig = go.Figure(
        go.Heatmap(
            z=heatmap_data.values,  # Number of Accidents
            x=month_names,          # Month
            y=y_axis_values,        # Elevation or Snow Height
            colorscale='ice',
            reversescale=True,
            colorbar=dict(
                title='# Accidents',
                titlefont=dict(
                    color='gray',
                    size=8,
                    family='Arial, sans-serif'),
                tickfont=dict(
                    color='gray',
                    size=8,
                    family='Arial, sans-serif'))
        )
    )
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 12)),
            ticktext=month_names),
        yaxis=dict(
            title=y_axis_label,
            titlefont=dict(color='gray', size=8, family='Arial, sans-serif'),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        template='plotly_white',
        height=220
    )
    fig.update_yaxes(
        tickfont=dict(color='gray', size=8, family='Arial, sans-serif')
    )
    fig.update_xaxes(
        tickfont=dict(color='gray', size=8, family='Arial, sans-serif')
    )

    return fig

"""
-----------------------------------------------------------------------------------------
Section 3.2: right column - Training Data 
"""
@app.callback(
    Output('accidents_stats', 'figure'),
        [Input('month_dropdown', 'value'),
        Input('risk_group_checklist', 'value')]
)
def accidents_stats(selected_month, selected_risk_groups):
    filtered_df = kmeans_df.copy()

    # Convert dates to datetime and add year and month columns:
    filtered_df['date'] = pd.to_datetime(filtered_df['date'])
    filtered_df['month'] = filtered_df['date'].dt.month
    filtered_df['year'] = filtered_df['date'].dt.year

    # Filter by selected month, if any:
    if selected_month:
        filtered_df = filtered_df[filtered_df['month'] == selected_month]

    if selected_risk_groups:
        filtered_df = filtered_df[filtered_df['risk_group'].isin(selected_risk_groups)]

    # Define alpine regions for splitting subplots
    alpine_regions = filtered_df['alpine_region'].unique()
    region_names = {region: region for region in alpine_regions}  # Map for titles

    # Color mapping for clusters
    color_map = {
        3: 'lightgreen',  # 1 - very low risk
        2: 'yellow',  # 2 - low risk
        0: 'orange',  # 3 - moderate risk
        1: 'lightcoral',  # 4 - considerable risk
        4: 'darkred'  # 5 - high risk
    }

    max_accidents = 0
    for region in alpine_regions:
        region_data = filtered_df[filtered_df['alpine_region'] == region]
        yearly_counts = region_data.groupby(['year', 'k_cluster']).size().unstack(fill_value=0)
        max_accidents = max(max_accidents, yearly_counts.sum(axis=1).max())

    # Create 2x2 subplot structure
    fig = make_subplots(rows=2, cols=2, subplot_titles=[region_names[r] for r in alpine_regions],
                        vertical_spacing=0.1, horizontal_spacing=0.1)

    # Set the font size for subplot titles specifically
    for i, region in enumerate(alpine_regions):
        fig['layout']['annotations'][i].update(font=dict(size=10, family='Arial, sans serif', color='black'))

    # Plot stacked bar charts:
    for idx, region in enumerate(alpine_regions):
        region_data = filtered_df[filtered_df['alpine_region'] == region]
        yearly_counts = region_data.groupby(['year', 'k_cluster']).size().unstack(fill_value=0)

        # Calculate row and column for the subplot
        row = (idx // 2) + 1
        col = (idx % 2) + 1

        # Add a trace for each cluster to stack them
        for cluster, color in color_map.items():
            if cluster in yearly_counts.columns:
                fig.add_trace(
                    go.Bar(
                        x=yearly_counts.index,
                        y=yearly_counts[cluster],
                        name=f"Cluster {cluster}",
                        marker_color=color,
                        showlegend=(idx == 0)  # Only show legend for the first subplot
                    ),
                    row=row, col=col
                )

        # Set y-axis range for the subplot
        fig.update_yaxes(range=[0, max_accidents],
                         row=row,
                         col=col,
                         tickfont=dict(size=8, family='Arial, sans-serif', color='gray')),

        fig.update_xaxes(
            tickfont=dict(size=8, family='Arial, sans-serif', color='gray'),
            row=row,
            col=col,
            #show every fifth year on the x-axis
            tickvals=yearly_counts.index[::6],
        )

    # Update layout
    fig.update_layout(
        showlegend=False,
        height=275,
        margin=dict(t=15, b=0, l=0, r=0),
        font=dict(size=8),
        template='plotly_white',
        barmode='stack'  # Stacking bars
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)