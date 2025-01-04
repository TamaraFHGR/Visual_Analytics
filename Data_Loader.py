import pandas as pd
import time

"""
------------------------------------------------------------------------------------------------------------------------
-> 1) Load historical (static) Data:
"""
def load_imis_stations():
    start_time = time.time()
    station_df = pd.read_csv('assets/API/daily/00_SLF_imis_stations.csv', sep=';', skiprows=0)
    end_time = time.time()
    print(f"function 'load_imis_stations()' runtime: {end_time - start_time:.4f} seconds") # 0.0040 seconds
    return station_df

stations_df = load_imis_stations()
#print(stations_df.head())

def load_hist_data():
    start_time = time.time()
    acc_df = pd.read_csv('assets/API/daily/01_SLF_hist_statistical_avalanche_data.csv', sep=',', skiprows=0)
    end_time = time.time()
    print(f"function 'load_hist_data()' runtime: {end_time - start_time:.4f} seconds")  # 0.0123 seconds
    return acc_df

acc_df = load_hist_data()
#print(acc_df.head())

"""
------------------------------------------------------------------------------------------------------------------------
Part 2 - Load real-time (daily, dynamic) Data:
"""
# 2.1) Load daily IMIS measurement data (updated every 30 minutes) and aggregate to daily data:
# New data collected every day at 6:00 UTC with '01_API_Load_IMIS_Daily_Data.py'
def load_measurements():
    measure_df = pd.read_csv('assets/API/daily/04_SLF_daily_imis_measurements_daily.csv', sep=';',skiprows=0)
    # Extract only relevant columns:
    measure_df = measure_df[[
        'station_code', 'date', 'TA_30MIN_MEAN', 'VW_30MIN_MEAN', 'VW_30MIN_MAX' , 'TSS_30MIN_MEAN', 'TS0_30MIN_MEAN',
    ]].rename(columns={
        'station_code': 'code',
        'date': 'date',
        'TA_30MIN_MEAN': 'air_temp_daily_mean',
        'VW_30MIN_MEAN': 'wind_speed_daily_mean',
        'VW_30MIN_MAX': 'wind_speed_daily_max',
        'TSS_30MIN_MEAN': 'snow_surf_temp_daily_mean',
        'TS0_30MIN_MEAN': 'snow_ground_temp_daily_mean'
    })

    # Filter on newest date only:
    measure_df['date'] = pd.to_datetime(measure_df['date'])
    latest_date = measure_df['date'].max()
    measure_df = measure_df[measure_df['date'] == latest_date]

    return measure_df

#-> replaced by 2.3 daily_df (combined df, weather and snow)
#measure_df = load_measurements()
#print(measure_df.head())
#print(len(measure_df)) # 202 rows

# 2.2) Load daily IMIS snow data (updated once a day):
# New data collected every day at 6:00 UTC with '02_API_Load_IMIS_Daily_Snow'
def load_snow():
    snow_df = pd.read_csv('assets/API/daily/05_SLF_daily_imis_snow.csv', sep=';',skiprows=0)
    # Extract only relevant columns:
    snow_df = snow_df[[
        'station_code', 'measure_date', 'HS', 'HN_1D'
    ]].rename(columns={
        'station_code': 'code',
        'measure_date': 'date',
        'HS': 'snow_height_daily_mean',
        'HN_1D': 'new_snow_daily_mean',
    })

    # Filter on newest date only:
    snow_df['date'] = pd.to_datetime(snow_df['date'])
    latest_date = snow_df['date'].max()
    snow_df = snow_df[snow_df['date'] == latest_date]
    return snow_df

#-> replaced by 2.3 daily_df (combined df, weather and snow)
#snow_df = load_snow()
#print(snow_df.head())
#print(len(snow_df)) # 132 rows

# 2.3) Load combined data (measurements and snow data):
def load_daily():
    start_time = time.time()
    daily_df = pd.read_csv('assets/API/daily/06_SLF_daily_imis_all_live_data.csv', sep=';',skiprows=0)
    end_time = time.time()
    print(f"function 'load_daily()' runtime: {end_time - start_time:.4f} seconds")  # 0.4492 seconds
    return daily_df

daily_df = load_daily()

"""
------------------------------------------------------------------------------------------------------------------------
Part 3 - Load Cluster K-Means Data:
"""

# 3.1) Load K-Means Clustered Trainings-Data:
def load_kmeans_training():
    start_time = time.time()
    kmeans_df = pd.read_csv('assets/K-Means_Clustering/01_hist_input_data_k-clustered.csv', sep=',', skiprows=0)
    end_time = time.time()
    print(f"function 'load_kmeans_training()' runtime: {end_time - start_time:.4f} seconds")  # 0.0089 seconds
    return kmeans_df

kmeans_df = load_kmeans_training()

# 3.2) Load PCA Trainings-Data:
def load_pca_training():
    start_time = time.time()
    pca_df = pd.read_csv('assets/K-Means_Clustering/02_hist_pca_data.csv', sep=',', skiprows=0)
    end_time = time.time()
    print(f"function 'load_pca_training()' runtime: {end_time - start_time:.4f} seconds")  # 0.0045 seconds
    return pca_df

pca_training_df = load_pca_training()

# 3.2) Load t-SNE Trainings-Data (for test only):
def load_tsne_training():
    tsne_training_df = pd.read_csv('assets/API/daily/08_tSNE_Trainings_Data.csv', sep=',', skiprows=0)
    return tsne_training_df

#tsne_training_df = load_tsne_training()

# 3.3) Load PCA Live-Data:
def load_pca_live():
    start_time = time.time()
    pca_live_df = pd.read_csv('assets/API/daily/09_PCA_Live_Data.csv', sep=',', skiprows=0, quotechar='"')
    pca_live_df['station_code'] = pca_live_df['station_code'].str.strip()
    end_time = time.time()
    print(f"function 'load_pca_live()' runtime: {end_time - start_time:.4f} seconds")  # 0.1853 seconds
    return pca_live_df

pca_live_df = load_pca_live()

# 3.3) Load t-SNE Live-Data (for test only):
def load_tsne_live():
    tsne_live_df = pd.read_csv('assets/API/daily/09_tSNE_Live_Data.csv', sep=',', skiprows=0, quotechar='"')
    tsne_live_df['station_code'] = tsne_live_df['station_code'].str.strip()
    return tsne_live_df

#tsne_live_df = load_tsne_live()

# 3.4) Load K-Means Cluster Centers:
def load_kmeans_centers_pca():
    start_time = time.time()
    kmeans_centers_pca_df = pd.read_csv('assets/K-Means_Clustering/03_hist_cluster_centers.csv', sep=',', skiprows=0)
    end_time = time.time()
    print(f"function 'load_kmeans_centers_pca()' runtime: {end_time - start_time:.4f} seconds")  # 0.0020 seconds
    return kmeans_centers_pca_df

kmeans_centers_df = load_kmeans_centers_pca()

# 3.4) Load K-Means Cluster Centers t-SNE (for test only):
def load_kmeans_centers_tsne():
    kmeans_centers_df = pd.read_csv('assets/K-Means_Clustering/03_hist_cluster_centers_tSNE.csv', sep=',', skiprows=0)
    return kmeans_centers_df

#kmeans_centers_df = load_kmeans_centers_tsne()

"""
total runtime = 0.0040 + 0.0123 + 0.4492 + 0.0089 + 0.0045 + 0.1853 + 0.0020 = 0.6662 seconds
"""