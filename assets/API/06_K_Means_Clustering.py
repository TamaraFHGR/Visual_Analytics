import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

acc_df = pd.read_csv('daily/07_K-Means_Trainings_Data.csv', sep=',', skiprows=0)
#extract only for latest measure_date (is max):
live_df = pd.read_csv('daily/06_SLF_daily_imis_all_live_data.csv', sep=';', skiprows=0)
live_df = live_df[live_df['measure_date'] == live_df['measure_date'].max()]

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

# PCA on the training data:
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)
data_pca_df = pd.DataFrame(data_pca, columns=['PCA1', 'PCA2'])
data_pca_df['k_cluster'] = cluster_labels

# K-Means on the live data:
live_df_cleaned = live_df.dropna(subset=features).copy()
live_data_scaled = scaler.transform(live_df_cleaned[features])
live_cluster_labels = kmeans.predict(live_data_scaled)

# PCA on the live data:
live_data_pca = pca.transform(live_data_scaled)
live_data_pca_df = pd.DataFrame(live_data_pca, columns=['PCA1', 'PCA2'])
live_data_pca_df['k_cluster'] = live_cluster_labels

# Save the data:
data_pca_df.to_csv('daily/08_PCA_Trainings_Data.csv', index=False)
live_data_pca_df.to_csv('daily/09_PCA_Live_Data.csv', index=False)