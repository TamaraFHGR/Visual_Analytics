*************
Data Description:
*************
- imis_df = load_imis_stations()
	- 	assets/API/daily/00_SLF_imis_stations.csv
 	- 	used in 1.1 imis_live_map
 
 - daily_snow_df = load_daily()
   	- 	assets/API/daily/06_SLF_daily_imis_all_live_data.csv
   	- 	used in 1.1 imis_live_map and 1.2 weather_trend

- k_centers_df = load_kmeans_centers()
	- 	assets/K-Means_Clustering/03_hist_cluster_centers.csv
 	- 	used in 2.1 k_means_cluster

- pca_training_df = load_pca_training()
	-	assets/K-Means_Clustering/02_hist_pca_data.csv
 	-	used in 2.1 k_means_cluster

- pca_live_df = load_pca_live()
	- 	assets/API/daily/09_PCA_Live_Data.csv
 	- 	used in 2.1 k_means_cluster and 2.2 cluster_matrix		

- acc_df = load_hist_data()
	- 	assets/API/daily/01_SLF_hist_statistical_avalanche_data.csv
	- 	used in 3.1 imis_acc_map
  
- kmeans_df = load_kmeans_training()
	- 	assets/K-Means_Clustering/01_hist_input_data_k-clustered.csv$
 	- 	used in 3.2 accident_stats

*************
Git Workflow:
*************
1) daily_imis_api_call.yml				--> # Daily at 06:00 UTC (07:00 local time)
	- 01_API_Load_IMIS_Daily_Data.py
	- 02_API_Load_IMIS_Daily_Snow.py

2) duplicate_deletion.yml				--> # Daily at 06:10 UTC (07:10 local time)
	- 03_Delete_Duplicates.py

3) aggregation_daily.yml				--> # Daily at 06:20 UTC (07:20 local time)
	- 04_Aggregate_to_daily_data.py

4) merge_all_daily_live_data.yml		--> # Daily at 06:30 UTC (07:30 local time)
	- 05_Prepare_live_data_for_PCA.py

5) k_means_clustering.yml				--> # Daily at 06:40 UTC (07:40 local time)
    - 06_K-Means_Clustering.py

	
