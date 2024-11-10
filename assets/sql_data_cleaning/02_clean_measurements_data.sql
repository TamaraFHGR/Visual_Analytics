  drop table [FHGR_DV_HS24].[dbo].[02_SLF_hist_daily_measurements_clean];
  
  select count(*) from [dbo].[02_SLF_hist_daily_measurements];
-- 1'511'176 records

SELECT
	   md.[station_code]
	  ,im.[station_name]
	  ,im.[lon] as imis_longitude
	  ,im.[lat] as imis_latitude
	  ,im.[elevation] as imis_elevation
	  ,im.[canton_code] as canton
	  ,'WEATHER' as imis_type
	  --,im.[type] as imis_type
      ,md.[date] as measure_date
      --,[measure_date]
      ,md.[hyear]
      ,md.[TA_30MIN_MEAN] as air_temp_day_mean	--°C
	  ,md.[VW_30MIN_MEAN] as wind_speed_day_mean  --m/s
      ,md.[VW_30MIN_MAX] as wind_speed_day_max  --m/s
	  ,md.[TSS_30MIN_MEAN] as snow_surf_temp_day_mean --°C
      ,md.[TS0_30MIN_MEAN] as snow_ground_temp_day_mean --°C
      --,md.[DW_30MIN_MEAN]
      --,[DW_30MIN_SD]
	  --,[RH_30MIN_MEAN]
      --,[TS25_30MIN_MEAN]
      --,[TS50_30MIN_MEAN]
      --,[TS100_30MIN_MEAN]
      --,[RSWR_30MIN_MEAN]	  
  INTO [FHGR_DV_HS24].[dbo].[02_SLF_hist_daily_measurements_clean]
  FROM [FHGR_DV_HS24].[dbo].[02_SLF_hist_daily_measurements] md
  join [dbo].[00_SLF_imis_stations_clean] im
	on md.[station_code] = im.[code]
  where md.[date] > '1997-12-31';
-- 1'406'635 records remaining (73'943 records lost due to join ~5%)

select * from [FHGR_DV_HS24].[dbo].[02_SLF_hist_daily_measurements_clean];

