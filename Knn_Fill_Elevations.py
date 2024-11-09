from sklearn.neighbors import NearestNeighbors
from Data_Loader import load_accidents
from timeit import default_timer as timer

start = timer()
acc_df = load_accidents()

# K-NN Model to fill missing elevation data in acc_df based on 3 nearest neighbors:
def fill_missing_elevation_with_knn(acc_df, n_neighbors):
    # Split data into two sets:
    acc_with_elevation = acc_df.dropna(subset=['start_zone_elevation']) # Only rows with elevation data
    acc_without_elevation = acc_df[acc_df['start_zone_elevation'].isna()] # Only rows without elevation data

    # Fit K-NN model based on data with elevation dataset:
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(acc_with_elevation[['start_zone_coordinates_latitude', 'start_zone_coordinates_longitude']])

    # Find nearest neighbors for dataset without elevation:
    distances, indices = knn.kneighbors(
        acc_without_elevation[['start_zone_coordinates_latitude', 'start_zone_coordinates_longitude']]
    )

    # Fill missing elevation data with mean elevation values:
    for i, idx in enumerate(acc_without_elevation.index):
        neighbor_indices = indices[i]
        mean_elevation = round(acc_with_elevation.iloc[neighbor_indices]['start_zone_elevation'].mean(), 2)
        acc_df.at[idx, 'start_zone_elevation'] = mean_elevation # Append data to original DataFrame

    return acc_df

acc_df = fill_missing_elevation_with_knn(acc_df,3)
acc_df.to_csv('assets/01_2_SLF_hist_avalanche_accidents_mapped.csv', index=False)
print(f"Runtime: {timer()-start:.2f} s") # Runtime: 18 s
