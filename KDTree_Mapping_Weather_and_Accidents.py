import numpy as np
from scipy.spatial import KDTree
from Data_Loader import load_imis_stations, load_accidents, load_hist_measurements, load_hist_snow, load_measurements, load_snow

# Load data:
imis_df = load_imis_stations()
acc_df = load_accidents()
hist_measure_df = load_hist_measurements()
hist_snow_df = load_hist_snow()
measure_df = load_measurements()
snow_df = load_snow()

def find_closest_stations(acc_df, imis_df):
    # Conversion of the coordinates to radians and adding the elevation to the coordinates:
    acc_coords = np.radians(acc_df[['start_zone_coordinates_latitude', 'start_zone_coordinates_longitude']].values)
    acc_elevations = acc_df['start_zone_elevation'].values
    imis_coords = np.radians(imis_df[['lat', 'lon']].values)
    imis_elevations = imis_df['elevation'].values  # Höhe der Wetterstationen

    # Combination of coordinates and elevation in a 3D array:
    acc_coords_3d = np.column_stack((acc_coords, acc_elevations))
    imis_coords_3d = np.column_stack((imis_coords, imis_elevations))

    # Create a KDTree for the weather stations:
    tree = KDTree(imis_coords_3d)

    # Find the closest weather stations for each accident:
    distances, indices = tree.query(acc_coords_3d, k=1)

    # Add the distances and the closest weather station to the accident dataframe:
    acc_df['distance_km'] = distances.flatten()
    acc_df['closest_station'] = imis_df.iloc[indices.flatten()]['code'].values

    return acc_df

acc_df = find_closest_stations(acc_df, imis_df)
print(acc_df.head())