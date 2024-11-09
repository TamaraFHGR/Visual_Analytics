select count(*) from [dbo].[03_SLF_hist_daily_snow];
-- 1'118'694 records

SELECT
	   sd.[station_code]
	  ,im.[station_name]
	  ,im.[lon] as imis_longitude
	  ,im.[lat] as imis_latitued
	  ,im.[elevation] as imis_elevation
	  ,im.[canton_code] as canton
	  ,'SNOW' as imis_type
	  --,im.[type] as imis_type
      ,convert(date, left(sd.[measure_date],10), 120) as measure_date
      ,sd.[hyear]
      ,sd.[HS] as snow_height_cm
      ,sd.[HN_1D] as new_snow_cm
  INTO [FHGR_DV_HS24].[dbo].[03_SLF_hist_daily_snow_clean]
  FROM [FHGR_DV_HS24].[dbo].[03_SLF_hist_daily_snow] sd
  join [dbo].[00_SLF_imis_stations_clean] im
	on sd.[station_code] = im.[code]
  where [measure_date] > '1997-12-31';

-- 1'025'487 records remaining (67'679 records lost due to join ~6%)


select * from [FHGR_DV_HS24].[dbo].[03_SLF_hist_daily_snow_clean];
