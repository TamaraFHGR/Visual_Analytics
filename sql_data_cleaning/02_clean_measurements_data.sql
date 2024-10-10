  select count(*) from [dbo].[02_SLF_hist_daily_measurements];
-- 1'511'176 records

SELECT
	   md.[station_code]
	  ,im.[station_name]
	  ,im.[lon] as imis_longitude
	  ,im.[lat] as imis_latitued
	  ,im.[elevation] as imis_elevation
	  ,im.[canton_code] as canton
	  ,im.[type] as imis_type
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
	on md.[station_code] = im.[code];


-- 1'437'233 records remaining (73'943 records lost due to join ~5%)

select * from [FHGR_DV_HS24].[dbo].[02_SLF_hist_daily_measurements_clean];


-- reduce double referencement again, to reduce data volume and limit to beginning of 01.01.1998:
SELECT
	   [station_code]
      --,[station_name]
      --,[imis_longitude]
      --,[imis_latitued]
      --,[imis_elevation]
      --,[canton]
      --,[imis_type]
      ,[measure_date]
      ,[hyear]
      ,[air_temp_day_mean]
      ,[wind_speed_day_mean]
      ,[wind_speed_day_max]
      ,[snow_surf_temp_day_mean]
      ,[snow_ground_temp_day_mean]
  FROM [FHGR_DV_HS24].[dbo].[02_SLF_hist_daily_measurements_clean]
  where [measure_date] > '1997-12-31'
  order by [measure_date] asc;
  -- 1'406'635 records