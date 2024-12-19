import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE  # Importiere t-SNE
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

acc_df = pd.read_csv('daily/07_K-Means_Trainings_Data.csv', sep=',', skiprows=0)

live_df = pd.read_csv('daily/06_SLF_daily_imis_all_live_data.csv', sep=';', skiprows=0)
live_df['measure_date'] = pd.to_datetime(live_df['measure_date'], format='%d.%m.%Y', errors='coerce')
live_df_filtered = live_df[live_df['measure_date'] > '2024-10-05'].copy()

# Definiere die Merkmale
features = [
    'air_temp_mean_stations', 'wind_speed_max_stations',
    'snow_height_mean_stations', 'new_snow_mean_stations',
]

acc_df_cleaned = acc_df.dropna(subset=features).copy()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(acc_df_cleaned[features])

# K-Means on the training data:
kmeans = KMeans(n_clusters=5, random_state=0)
cluster_labels = kmeans.fit_predict(data_scaled)
acc_df_cleaned['k_cluster'] = cluster_labels

# Prepare Live Data:
live_df_cleaned = live_df.dropna(subset=features).copy()
live_data_scaled = scaler.transform(live_df_cleaned[features])  # Scale with the same scaler as the training data

# Combine the training and live data for t-SNE:
combined_data_scaled = pd.concat(
    [pd.DataFrame(data_scaled), pd.DataFrame(live_data_scaled)],
    axis=0,
    ignore_index=True
)

# T-SNE Calculation on the combined data:
tsne = TSNE(n_components=2, random_state=0, max_iter=250, perplexity=30)
combined_tsne = tsne.fit_transform(combined_data_scaled)

train_tsne = combined_tsne[:len(data_scaled)]
live_tsne = combined_tsne[len(data_scaled):]

data_tsne_df = pd.DataFrame(train_tsne, columns=['t-SNE1', 't-SNE2'])
data_tsne_df['k_cluster'] = cluster_labels

live_data_tsne_df = pd.DataFrame(live_tsne, columns=['tSNE1', 'tSNE2'])
live_cluster_labels = kmeans.predict(live_data_scaled)  # K-Means auf Live-Daten anwenden
live_data_tsne_df['k_cluster'] = live_cluster_labels

# Additional columns for the training data:
additional_columns_train = [
    'municipality', 'canton',
    'date', 'alpine_region', 'elevation_group'
]
additional_data_train = acc_df_cleaned[additional_columns_train].reset_index(drop=True)
data_tsne_df = pd.concat([data_tsne_df, additional_data_train], axis=1)

# Additional columns for the live data:
additional_columns_live = [
    'station_code', 'station_name', 'canton_code',
    'measure_date', 'alpine_region', 'elevation_group'
]
additional_data_live = live_df_cleaned[additional_columns_live].reset_index(drop=True)
live_data_tsne_df = pd.concat([live_data_tsne_df, additional_data_live], axis=1)

# Save the data:
data_tsne_df.to_csv('daily/08_tSNE_Trainings_Data.csv', index=False)
live_data_tsne_df.to_csv('daily/09_tSNE_Live_Data.csv', index=False)