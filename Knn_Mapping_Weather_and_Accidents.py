import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from Data_Loader import load_imis_stations, load_accidents, load_hist_measurements, load_hist_snow

# Load data
imis_df = load_imis_stations()
acc_df = load_accidents()
hist_measure_df = load_hist_measurements()
hist_snow_df = load_hist_snow()

# K-NN Model to fill missing elevation data in accidents df based on 3 nearest neighbors
# (output: acc_complete_df):
def fill_missing_elevation_with_knn(acc_df, n_neighbors=3):
    acc_with_elevation = acc_df.dropna(subset=['start_zone_elevation'])
    acc_without_elevation = acc_df[acc_df['start_zone_elevation'].isna()]

    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(acc_with_elevation[['start_zone_coordinates_latitude', 'start_zone_coordinates_longitude']])

    distances, indices = knn.kneighbors(
        acc_without_elevation[['start_zone_coordinates_latitude', 'start_zone_coordinates_longitude']]
    )

    for i, idx in enumerate(acc_without_elevation.index):
        neighbor_indices = indices[i]
        mean_elevation = acc_with_elevation.iloc[neighbor_indices]['start_zone_elevation'].mean()
        acc_df.at[idx, 'start_zone_elevation'] = mean_elevation

    return acc_df
acc_complete_df = fill_missing_elevation_with_knn(acc_df)

# K-NN Model to map historical weather data to accidents based on 3 nearest neighbors
# (output: acc_mapped_df):
def find_closest_weather_stations(imis_df, acc_complete_df, hist_measure_df):
    weather_columns = {
        'mean_air_temp': 'air_temp_day_mean',
        'mean_wind_speed': 'wind_speed_day_mean',
        'max_wind_speed': 'wind_speed_day_max',
        'mean_snow_surf_temp': 'snow_surf_temp_day_mean',
        'mean_snow_ground_temp': 'snow_ground_temp_day_mean'
    }

    knn = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
    knn.fit(imis_df[['lat', 'lon', 'elevation']])

    accident_coords = acc_complete_df[['start_zone_coordinates_latitude',
                                       'start_zone_coordinates_longitude',
                                       'start_zone_elevation']].copy()
    accident_coords.columns = ['lat', 'lon', 'elevation']

    distances, indices = knn.kneighbors(accident_coords)

    for i in range(len(acc_complete_df)):
        nearest_station_indices = indices[i]
        nearest_station_codes = imis_df.iloc[nearest_station_indices]['code'].values
        accident_date = acc_complete_df.at[i, 'date']

        day_measurements = hist_measure_df[(hist_measure_df['station_code'].isin(nearest_station_codes)) &
                                           (hist_measure_df['measure_date'] == accident_date)]

        if not day_measurements.empty:
            for new_col, orig_col in weather_columns.items():
                acc_complete_df.at[i, new_col] = day_measurements[orig_col].mean()

    return acc_complete_df

# K-NN Model to map historical snow data to accidents based on 3 nearest neighbors
# (output: acc_mapped_df):
def find_closest_snow_stations(imis_df, acc_complete_df, hist_snow_df):
    snow_columns = {
        'mean_snow_height': 'snow_height_cm',
        'mean_new_snow': 'new_snow_cm'
    }

    knn = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
    knn.fit(imis_df[['lat', 'lon', 'elevation']])

    accident_coords = acc_complete_df[['start_zone_coordinates_latitude',
                                         'start_zone_coordinates_longitude',
                                         'start_zone_elevation']].copy()
    accident_coords.columns = ['lat', 'lon', 'elevation']

    distances, indices = knn.kneighbors(accident_coords)

    for i in range(len(acc_complete_df)):
        nearest_station_indices = indices[i]
        nearest_station_codes = imis_df.iloc[nearest_station_indices]['code'].values
        accident_date = acc_complete_df.at[i, 'date']

        day_snow_data = hist_snow_df[(hist_snow_df['station_code'].isin(nearest_station_codes)) &
                                     (hist_snow_df['measure_date'] == accident_date)]

        if not day_snow_data.empty:
            for new_col, orig_col in snow_columns.items():
                acc_complete_df.at[i, new_col] = day_snow_data[orig_col].mean()

    return acc_complete_df

# Aufruf der Funktionen
acc_mapped_df = find_closest_weather_stations(imis_df, acc_complete_df, hist_measure_df)
acc_mapped_df = find_closest_snow_stations(imis_df, acc_complete_df, hist_snow_df)

# Speichern des finalen DataFrames als CSV
acc_mapped_df.to_csv('assets/01_2_SLF_hist_mapped_avalanche_accidents.csv', index=False)
print(acc_mapped_df.head())
print('Data saved successfully.')