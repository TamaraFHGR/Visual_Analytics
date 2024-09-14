import requests
import plotly.graph_objs as go
import plotly.io as pio

# Define the API key and the city
API_KEY = 'your_openweathermap_api_key'
CITY = 'London'
API_URL = f'http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric'

# Fetch real-time data from OpenWeatherMap API
response = requests.get(API_URL)
weather_data = response.json()

# Extract relevant data
city_name = weather_data['name']
temperature = weather_data['main']['temp']
humidity = weather_data['main']['humidity']
pressure = weather_data['main']['pressure']
description = weather_data['weather'][0]['description']

# Print the data to check
print(f"City: {city_name}")
print(f"Temperature: {temperature} °C")
print(f"Humidity: {humidity} %")
print(f"Pressure: {pressure} hPa")
print(f"Weather description: {description}")

# Create a Plotly bar chart for visualization
fig = go.Figure(data=[
    go.Bar(name='Temperature (°C)', x=['Temperature'], y=[temperature], marker_color='blue'),
    go.Bar(name='Humidity (%)', x=['Humidity'], y=[humidity], marker_color='green'),
    go.Bar(name='Pressure (hPa)', x=['Pressure'], y=[pressure], marker_color='orange'),
])

# Customize layout
fig.update_layout(
    title=f'Real-Time Weather Data for {city_name}',
    xaxis_title='Weather Metrics',
    yaxis_title='Values',
    barmode='group'
)

# Render the figure
pio.show(fig)