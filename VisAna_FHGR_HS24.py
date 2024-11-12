from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import os
from datetime import datetime
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Data_Loader import load_hist_data, load_imis_stations, load_measurements, load_snow, load_measurements_trend, load_snow_trend

acc_df = load_hist_data()
imis_df = load_imis_stations()
daily_weather_df = load_measurements()
daily_snow_df = load_snow()
trend_measure_df = load_measurements_trend()
trend_snow_df = load_snow_trend()

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

    #Column 1 of 2 (left):
    html.Div([
        html.Div([
            html.H2('Current Weather Conditions in the Swiss Alps'),
            html.H3(f'Last Update: {last_update}', style={'color': 'grey'}),
            html.Div([
                html.Button('Stations', id='default_button', n_clicks=0),
                html.Button('Temperature', id='temp_button', n_clicks=0),
                html.Button('Wind', id='wind_button', n_clicks=0),
                html.Button('Snow Height', id='snow_height_button', n_clicks=0),
                html.Button('New Snow', id='new_snow_button', n_clicks=0)
            ], className='weather_buttons'),
            html.Div([
                dcc.Graph(id='live_geomap'),
                html.Div([
                    html.H2('Trend Analysis'),
                    dcc.Dropdown(id='station_dropdown',
                                 options=[{'label': station, 'value': station} for station in imis_df['code'].unique()],
                                 multi=True,
                                 placeholder="Select stations",
                                 value=[]),
                    dcc.Graph(id='trend_analysis')
                    ], className='trend_plot'),
                #dcc.Graph(id='risk_map')
            ], className='live_weather'),
        ], style={'display': 'inline-block'}, className='left_column'),

        #Column 2 of 2 (right):
        html.Div([
            html.H2('Historical Avalanche Accident Data'),
            html.Div([
                dcc.RangeSlider(id='altitude', min=1000, max=4000, step=500),
                dcc.Graph(id='training_geomap', className='hover_cursor')
            ]),
            html.H2('Statistics on Accident Data'),
            html.Div([
                dcc.Graph(id='accidents_stats'),
            ], className='training_data'),
        ], style={'display': 'inline-block'}, className='right_column')
    ], className='main_container'),
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
        height=250
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
     Input('snow_height_button', 'n_clicks')]
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
    if not ctx.triggered:
        button_id = 'default_button'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

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
            colorscale="RdBu")
        )

    # Add heatmap for wind:
    elif button_id == 'wind_button':
        fig = go.Figure(go.Densitymap(
            lat=day_weather['lat'],
            lon=day_weather['lon'],
            z=day_weather['wind_speed_daily_mean'],
            radius=30,  # radius of the density heatmap, adjustable
            colorscale="Viridis")
        )

    # Add heatmap for snow height:
    elif button_id == 'snow_height_button':
        fig = go.Figure(go.Densitymap(
            lat=day_snow['lat'],
            lon=day_snow['lon'],
            z=day_snow['snow_height_daily_mean'],
            radius=30,  # radius of the density heatmap, adjustable
            colorscale="Blues")
        )

    # Add heatmap for new snow:
    elif button_id == 'new_snow_button':
        fig = go.Figure(go.Densitymap(
            lat=day_snow['lat'],
            lon=day_snow['lon'],
            z=day_snow['new_snow_daily_mean'],
            radius=30,  # radius of the density heatmap, adjustable
            colorscale='gray')
        )

    fig.update_layout(
        map_style="light",
        map_zoom=7,  # Adjust zoom level
        map_center={"lat": lat_imis.mean(), "lon": long_imis.mean()},  # Center map on the average location
        margin={"r":0,"t":0,"l":0,"b":0}, # Remove margins around the map
        height=400
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
     Input('station_dropdown', 'value')]
)

def weather_trend(default_button, temp_button, wind_button, new_snow_button, snow_height_button, selected_stations):
    # Determine which button was last clicked, or use 'default_button' if none was clicked:
    ctx = callback_context
    if not ctx.triggered:
        button_id = 'default_button'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    fig = go.Figure()

    # Add Line Plot for temperature:
    if button_id == 'temp_button' and selected_stations:
        for station in selected_stations:
            station_data = trend_measure_df[trend_measure_df['code'] == station]
            fig.add_trace(go.Scatter(
                x=station_data['date'],
                y=station_data['air_temp_daily_mean'],
                mode='lines', name=station)
            )
            fig.update_layout(xaxis_title="Measure Date",
                              yaxis_title="Air Temperature [°C]",
                              legend_title="Station Code",
                              margin={"r":0,"t":0,"l":0,"b":0},
                              height=300)

    # Add Line Plot for wind:
    elif button_id == 'wind_button' and selected_stations:
        for station in selected_stations:
            station_data = trend_measure_df[trend_measure_df['code'] == station]
            fig.add_trace(go.Scatter(
                x=station_data['date'],
                y=station_data['wind_speed_daily_mean'],
                mode='lines', name=station)
            )
            fig.update_layout(xaxis_title="Measure Date",
                              yaxis_title="Wind speed [m/s]",
                              legend_title="Station Code",
                              margin={"r": 0, "t": 0, "l": 0, "b": 0},
                              height=300)

    # Add Line Plot for snow height:
    elif button_id == 'snow_height_button' and selected_stations:
        for station in selected_stations:
            station_data = trend_snow_df[trend_snow_df['code'] == station]
            fig.add_trace(go.Scatter(
                x=station_data['date'],
                y=station_data['snow_height_daily_mean'],
                mode='lines', name=station)
            )
            fig.update_layout(xaxis_title="Measure Date",
                              yaxis_title="Height of Snowpack [cm]",
                              legend_title="Station Code",
                              margin={"r":0,"t":0,"l":0,"b":0},
                              height=300)

    # Add Line Plot for new snow:
    elif button_id == 'new_snow_button' and selected_stations:
        for station in selected_stations:
            station_data = trend_snow_df[trend_snow_df['code'] == station]
            fig.add_trace(go.Scatter(
                x=station_data['date'],
                y=station_data['new_snow_daily_mean'],
                mode='lines', name=station)
            )
            fig.update_layout(xaxis_title="Measure Date",
                              yaxis_title="Height of new Snow [cm]",
                              legend_title="Station Code",
                              margin={"r": 0, "t": 0, "l": 0, "b": 0},
                              height=300)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)