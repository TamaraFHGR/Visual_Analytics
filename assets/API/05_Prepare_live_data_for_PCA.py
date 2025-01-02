import pandas as pd

live_measure_df = pd.read_csv('daily/04_SLF_daily_imis_measurements_daily.csv', sep=';', skiprows=0, dtype={'station_code': str})
live_snow_df = pd.read_csv('daily/05_SLF_daily_imis_snow.csv', sep=';', skiprows=0, dtype={'station_code': str})
imis_df = pd.read_csv('daily/00_SLF_imis_stations.csv', sep=';', skiprows=0, dtype={'code': str})

# Step 1: Reformat date and drop unnecessary columns:
live_measure_df['measure_date'] = pd.to_datetime(live_measure_df['measure_date']).dt.date
live_measure_df = live_measure_df[
    ['station_code',
     'measure_date',
     'TA_30MIN_MEAN',
     'VW_30MIN_MAX',
     'TSS_30MIN_MEAN',
    'TS0_30MIN_MEAN']
     ]
live_measure_df = live_measure_df.rename(columns={
    'TA_30MIN_MEAN': 'air_temp_mean_stations',
    'VW_30MIN_MEAN': 'wind_speed_mean_stations',
    'VW_30MIN_MAX': 'wind_speed_max_stations',
    'TSS_30MIN_MEAN': 'snow_surf_temp_mean_stations',
    'TS0_30MIN_MEAN': 'snow_ground_temp_mean_stations'
})
live_snow_df['measure_date'] = pd.to_datetime(live_snow_df['measure_date']).dt.date
live_snow_df = live_snow_df[
    ['station_code',
     'measure_date',
     'HS',
     'HN_1D']
]
live_snow_df = live_snow_df.rename(columns={
    'HS': 'snow_height_mean_stations',
    'HN_1D': 'new_snow_mean_stations'
})
# Step 2: Full outer join for all measures:
combined_measure_df = pd.merge(live_measure_df, live_snow_df, on=['station_code', 'measure_date'], how='outer')

# Step 3: Combine with IMIS data:
live_df = pd.merge(imis_df, combined_measure_df, left_on='code', right_on='station_code', how='outer')
print(live_df.head())

# Step 4: # Grouping of alpine regions based on cantons
def assign_alpine_region(canton):
    if canton in {"AI", "AR", "SG", "GR", "GL", "FL"}:
        return "Eastern Alps"
    elif canton in {"VD", "FR", "GE", "NE", "JU", "BE"}:
        return "Western Alps"
    elif canton in {"NW", "OW", "LU", "UR", "SZ", "SO"}:
        return "Central Alps"
    elif canton in {"TI", "VS"}:
        return "Southern Alps"
    else:
        return "NONE"

live_df['alpine_region'] = live_df['canton_code'].apply(assign_alpine_region)

# Step 5: add column with region_id:
def assign_region_id(region):
    if region == "Western Alps":
        return 1
    elif region == "Central Alps":
        return 2
    elif region == "Southern Alps":
        return 3
    elif region == "Eastern Alps":
        return 4
    else:
        return 0

live_df['region_id'] = live_df['alpine_region'].apply(assign_region_id)

# Step 6: add column with elevation groups:
live_df['elevation_group'] = pd.cut(live_df['elevation'], bins=[0, 2000, 3000, 4000, 5000], labels=['2000', '3000', '4000', '5000'])
#print(live_df[['canton_code', 'alpine_region', 'elevation', 'elevation_group']].head())

# Step 7: remove rows where alpine_region is NONE:
live_df = live_df[live_df['alpine_region'] != 'NONE']

# Step 8: Save the data:
live_df.to_csv('daily/06_SLF_daily_imis_all_live_data.csv', sep=';', index=False)

