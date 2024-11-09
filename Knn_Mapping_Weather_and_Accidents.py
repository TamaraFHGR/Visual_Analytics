import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from Data_Loader import load_hist_measurements
from timeit import default_timer as timer

start = timer()
acc_df = pd.read_csv('assets/01_2_SLF_hist_avalanche_accidents_mapped.csv', sep=',', skiprows=0)
hist_measure_df = load_hist_measurements()

# K-NN Model to map historical weather data to accidents based on 3 nearest neighbors
# (output: acc_mapped_weather_df):
def find_closest_weather_stations(acc_df, hist_measure_df, lat_lon_weight, elevation_weight):
    # Define the weather columns:
    weather_columns = {
        'mean_air_temp': 'air_temp_day_mean',
        'mean_wind_speed': 'wind_speed_day_mean',
        'max_wind_speed': 'wind_speed_day_max',
        'mean_snow_surf_temp': 'snow_surf_temp_day_mean',
        'mean_snow_ground_temp': 'snow_ground_temp_day_mean'
    }

    # Create new DataFrame for weather data and stations:
    weather_data = pd.DataFrame(index=acc_df.index, columns=weather_columns.keys())
    used_station_codes = pd.Series(index=acc_df.index, dtype=object)

    # Iterate over all accident data:
    for i, accident in acc_df.iterrows():
        accident_date = accident['date']

        # Filter weather data by accident date:
        same_day_measurements = hist_measure_df[hist_measure_df['measure_date'] == accident_date]

        if same_day_measurements.empty:
            continue

        # Scaling of latitude, longitude, and elevation:
        scaled_lat_lon = same_day_measurements[['imis_longitude', 'imis_latitude']] * lat_lon_weight
        scaled_elevation = same_day_measurements['imis_elevation'] * elevation_weight
        scaled_coords = pd.concat([scaled_lat_lon, scaled_elevation], axis=1)

        # Train the K-NN model with the geographical data (Longitude, Latitude, Elevation):
        knn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
        knn.fit(scaled_coords)

        # Define the accident point (Longitude, Latitude, Elevation):
        accident_point = pd.DataFrame([[accident['start_zone_coordinates_longitude'] * lat_lon_weight,
                                        accident['start_zone_coordinates_latitude'] * lat_lon_weight,
                                        accident['start_zone_elevation'] * elevation_weight
                                        ]],
                                      columns=['imis_longitude', 'imis_latitude', 'imis_elevation'])

        # Find the nearest neighbors:
        distances, indices = knn.kneighbors(accident_point)

        # Filter valid neighbors:
        valid_indices = [idx for idx in indices[0] if idx < len(same_day_measurements)]
        num_neighbors = min(3, len(valid_indices))

        if num_neighbors == 0:
            continue

        neighbor_indices = valid_indices[:num_neighbors]

        # Calculate the mean values of the weather data:
        neighbor_data = same_day_measurements.iloc[neighbor_indices]
        mean_values = neighbor_data[list(weather_columns.values())].mean()

        # If NaN values are present, try with the next 6 neighbors:
        if np.isnan(mean_values.mean()) and len(valid_indices) >= 6:
            num_neighbors = min(6, len(valid_indices))
            neighbor_indices = valid_indices[:num_neighbors]
            neighbor_data = same_day_measurements.iloc[neighbor_indices]
            mean_values = neighbor_data[list(weather_columns.values())].mean()

        # Assign the mean values to the corresponding column:
        for col, source_col in weather_columns.items():
            weather_data.at[i, col] = mean_values.get(source_col, np.nan)

        # Save the used stations:
        used_station_codes.at[i] = ', '.join(neighbor_data['station_code'].astype(str))

    # Add the column for the used stations to the weather data DataFrame:
    weather_data['used_station_codes'] = used_station_codes

    # Combine the original accident data with the weather data:
    acc_mapped_weather_df = pd.concat([acc_df, weather_data], axis=1)
    return acc_mapped_weather_df

acc_mapped_weather_df = find_closest_weather_stations(acc_df, hist_measure_df, 1.0, 0.1)
acc_mapped_weather_df.to_csv('assets/01_3_SLF_hist_mapped_avalanche_accidents.csv', index=False)
print(acc_mapped_weather_df.head())
print(f"Runtime: {timer()-start:.2f} s")
