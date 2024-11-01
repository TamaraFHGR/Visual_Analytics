from scipy.spatial import KDTree
import numpy as np
import pandas as pd

# 1.1) Load Geodata of 204 IMIS stations (some only collect wind and temperature data, others also snow data):
imis_df = pd.read_csv('assets/00_SLF_imis_stations.csv', sep=';', skiprows=0)

# 1.2) Load historical data of avalanche accidents (3'301 records since 1998):
acc_df = pd.read_csv('assets/01_SLF_hist_avalanche_accidents.csv', sep=';',skiprows=0)

# Konvertiere Breitengrad und Längengrad in Radians
acc_coords = np.radians(acc_df[['start_zone_coordinates_latitude', 'start_zone_coordinates_longitude']].values)
imis_coords = np.radians(imis_df[['lat', 'lon']].values)

# Erstelle einen KDTree
tree = KDTree(imis_coords)
# Finde die drei nächstgelegenen Wetterstationen für jeden Unfallpunkt
distances, indices = tree.query(acc_coords, k=3)

# Füge die Station_IDs und die Distanzen hinzu
acc_df['distance_km_1'] = distances[:, 0] * 6371  # Erdradius in km für die erste Station
acc_df['closest_station_1'] = imis_df.iloc[indices[:, 0]]['code'].values

acc_df['distance_km_2'] = distances[:, 1] * 6371  # Erdradius in km für die zweite Station
acc_df['closest_station_2'] = imis_df.iloc[indices[:, 1]]['code'].values

acc_df['distance_km_3'] = distances[:, 2] * 6371  # Erdradius in km für die dritte Station
acc_df['closest_station_3'] = imis_df.iloc[indices[:, 2]]['code'].values

print(acc_df)
#save the result to a new CSV file
acc_df.to_csv('assets/xx_SLF_accidents_with_closest_stations.csv', sep=';', index=False)