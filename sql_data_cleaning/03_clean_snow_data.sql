select count(*) from [dbo].[03_SLF_hist_daily_snow];
-- 1'118'694 records

SELECT
	   sd.[station_code]
	  ,im.[station_name]
	  ,im.[lon] as imis_longitude
	  ,im.[lat] as imis_latitued
	  ,im.[elevation] as imis_elevation
	  ,im.[canton_code] as canton
	  ,im.[type] as imis_type
      ,convert(date, left(sd.[measure_date],10), 120) as measure_date
      ,sd.[hyear]
      ,sd.[HS] as snow_height_cm
      ,sd.[HN_1D] as new_snow_cm
  INTO [FHGR_DV_HS24].[dbo].[03_SLF_hist_daily_snow_clean]
  FROM [FHGR_DV_HS24].[dbo].[03_SLF_hist_daily_snow] sd
  join [dbo].[00_SLF_imis_stations_clean] im
	on sd.[station_code] = im.[code];

-- 1'051'015 records remaining (67'679 records lost due to join ~6%)

select * from [FHGR_DV_HS24].[dbo].[03_SLF_hist_daily_snow_clean];

-- reduce double referencement again, to reduce data volume and limit to start date 01.01.1998:
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
      ,[snow_height_cm]
      ,[new_snow_cm]
  FROM [FHGR_DV_HS24].[dbo].[03_SLF_hist_daily_snow_clean]
  where [measure_date] > '1997-12-31'
  order by [measure_date] asc;
  -- 1'025'442 records
