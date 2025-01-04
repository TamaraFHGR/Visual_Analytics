import pandas as pd
import time

start_time = time.time()

measure_df = pd.read_csv('daily/04_SLF_daily_imis_measurements.csv', sep=';',skiprows=0)
snow_df = pd.read_csv('daily/05_SLF_daily_imis_snow.csv', sep=';',skiprows=0)

# Delete duplicates, ignore the index and sort by station_code and measure_date:
measure_df.drop_duplicates(subset=['station_code', 'measure_date'], inplace=True, ignore_index=True)
measure_df.sort_values(by=['station_code', 'measure_date'], inplace=True)
snow_df.drop_duplicates(subset=['station_code', 'measure_date'], inplace=True, ignore_index=True)
snow_df.sort_values(by=['station_code', 'measure_date'], inplace=True)

# Reset indices and start from 1:
measure_df.reset_index(drop=True, inplace=True)
snow_df.reset_index(drop=True, inplace=True)
measure_df.index = measure_df.index + 1
snow_df.index = snow_df.index + 1

# Save the cleaned data to a new CSV file:
measure_df.to_csv('daily/04_SLF_daily_imis_measurements.csv', sep=';', index=False)
snow_df.to_csv('daily/05_SLF_daily_imis_snow.csv', sep=';', index=False)

end_time = time.time()
time = end_time - start_time
print(f'Time: {time} seconds')     # Time: 4.3200010776519775 seconds
