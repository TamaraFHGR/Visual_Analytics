from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Data_Loader import load_imis_stations
#from Data_Loader import load_hist_measurements, load_hist_snow, load_measurements, load_snow

# Load data:
imis_df = load_imis_stations()
acc_df = pd.read_csv('assets/01_3_SLF_hist_mapped_avalanche_accidents.csv', sep=',', skiprows=0)
#hist_measure_df = load_hist_measurements()
#hist_snow_df = load_hist_snow()
#measure_df = load_measurements()
#snow_df = load_snow()

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
            html.H3('Last Update: ....'),
            html.Div([
                html.Button('Temperature', id='temp_button', n_clicks=0),
                html.Button('Wind', id='wind_button', n_clicks=0),
                html.Button('New Snow', id='new_snow_button', n_clicks=0),
                html.Button('Snow Height', id='snow_height_button', n_clicks=0),
            ], className='weather_buttons'),
            html.Div([
                dcc.Graph(id='live_geomap'),
                dcc.Graph(id='risk_map')
            ], className='live_weather'),
        ], style={'width': '50%', 'display': 'inline-block'}, className='left_column'),

        #Column 2 of 2 (right):
        html.Div([
            html.H2('Historical Avalanche Accident Data'),
            html.Div([
                dcc.Graph(id='training_geomap', className='hover_cursor'),
                dcc.RangeSlider(id='altitude', min=1000, max=4000, step=500)
            ]),
            html.H2('Statistics on Accident Data'),
            html.Div([
                dcc.Graph(id='accidents_count'),
                dcc.Graph(id='weather_trend')
            ], className='training_data'),
        ], style={'width': '50%', 'display': 'inline-block'}, className='right_column')
    ], className='main_container'),
])

"""
-----------------------------------------------------------------------------------------
Section 1: Training Data Visualization -> Geo Map
"""

# Callback to open the station document in a new tab
@app.callback(
    Output('url', 'href'),
    [Input('training_geomap', 'clickData')]
)
def open_station_document(clickData):
    if clickData:
        # Extrahiere die URL aus customdata
        url = clickData['points'][0]['customdata']
        return url  # Setzt die URL des dcc.Location-Elements und öffnet es im Browser
    return no_update

# Callback to update the training data map based on the selected altitude range
@app.callback(
    Output('training_geomap', 'figure'),
    [Input('altitude', 'value')]
)
def imis_accident_map(altitude):
    # filter data by altitude:
    if altitude:
        min_altitude, max_altitude = altitude
        filtered_imis_df = imis_df[(imis_df['elevation'] >= min_altitude) & (imis_df['elevation'] <= max_altitude)]
        filtered_acc_df = acc_df[(acc_df['start_zone_elevation'] >= min_altitude) & (acc_df['start_zone_elevation'] <= max_altitude)]
    else:
        filtered_imis_df = imis_df
        filtered_acc_df = acc_df

    # URL for the IMIS station Details:
    filtered_imis_df['data_url'] = filtered_imis_df['code'].apply(lambda code: f"https://stationdocu.slf.ch/pdf/IMIS_{code}_DE.pdf")
    filtered_imis_df['hover_text'] = filtered_imis_df.apply(
        lambda row: f"{row['code']} - {row['station_name']}<br>"
                    f"Elevation: {row['elevation']} m<br>"
                    f"(click to open data sheet)", axis=1
    )

    lat_imis = filtered_imis_df['lat']
    long_imis = filtered_imis_df['lon']
    lat_acc = filtered_acc_df['start_zone_coordinates_latitude']
    long_acc = filtered_acc_df['start_zone_coordinates_longitude']

    # Plot IMIS locations:
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

    # Plot accident locations:
    fig.add_trace(go.Densitymap(
        lat=lat_acc,
        lon=long_acc,
        z=[1] * len(lat_acc),  # value to calculate the density, adjustable
        radius=15,  # radius of the density heatmap, adjustable
        colorscale="YlOrRd",  # Color scale: Yellow to Red
        opacity=0.6)
    )

    fig.update_layout(
        map_style="light",
        map_zoom=8,  # Adjust zoom level
        map_center={"lat": lat_imis.mean(), "lon": long_imis.mean()},  # Center map on the average location
        margin={"r":0,"t":0,"l":0,"b":0}  # Remove margins around the map
    )
    #fig.write_html('imis_stations_and_accidents.html', auto_open=True)
    return fig

"""
-----------------------------------------------------------------------------------------
Section 2: Training Data Visualization -> Accident Count
"""

@app.callback(
    Output('accidents_count', 'figure'),
    [Input('altitude', 'value')]
)
def accidents_count(altitude):
    # Filter data by altitude
    if altitude:
        min_altitude, max_altitude = altitude
        filtered_acc_df = acc_df[
            (acc_df['start_zone_elevation'] >= min_altitude) & (acc_df['start_zone_elevation'] <= max_altitude)]
    else:
        filtered_acc_df = acc_df

    # Convert dates to datetime and add year and month columns
    filtered_acc_df['date'] = pd.to_datetime(filtered_acc_df['date'])
    filtered_acc_df['year'] = filtered_acc_df['date'].dt.year
    filtered_acc_df['month'] = filtered_acc_df['date'].dt.month

    # Define winter months and corresponding names
    winter_months = [11, 12, 1, 2, 3, 4]
    month_names = {11: 'November', 12: 'December', 1: 'January', 2: 'February', 3: 'March', 4: 'April'}

    # Create subplot structure for two small multiples
    fig = make_subplots(rows=3, cols=3, subplot_titles=[month_names[m] for m in winter_months], vertical_spacing=0.05)

    # Loop over each winter month to create a separate bar chart for each:
    for idx, month in enumerate(winter_months):
        # Filter data for the month across all years
        monthly_data = filtered_acc_df[filtered_acc_df['month'] == month]

        # Count accidents per year within the month
        yearly_counts = monthly_data.groupby('year').size()

        # Calculate row and column for the subplot (3x3 grid)
        row = (idx // 3) + 1
        col = (idx % 3) + 1

        # Add bar chart to the respective subplot
        fig.add_trace(
            go.Bar(
                x=yearly_counts.index,  # Years
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
        title='Number of accidents by year for each winter month',
        showlegend=False,
        height=800
    )

    return fig

"""
-----------------------------------------------------------------------------------------
Section 3: Training Data Visualization -> Weather Trend
"""
@app.callback(
    Output('weather_trend', 'figure'),
    [Input('altitude', 'value')]
)
def weather_trend(altitude):
    # Filter data by altitude
    if altitude:
        min_altitude, max_altitude = altitude
        filtered_acc_df = acc_df[
            (acc_df['start_zone_elevation'] >= min_altitude) & (acc_df['start_zone_elevation'] <= max_altitude)]
    else:
        filtered_acc_df = acc_df

    # Create Distribution of Snow Height:
    fig = go.Figure()
    fig.add_trace(go.Histogram
                  (x=filtered_acc_df['mean_air_temp'], nbinsx=20, marker_color='blue'))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)