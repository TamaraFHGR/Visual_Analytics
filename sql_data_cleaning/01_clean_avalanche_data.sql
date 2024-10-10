--drop table [FHGR_DV_HS24].[dbo].[01_SLF_hist_avalanche_accidents_clean];

SELECT
		--[avalanche_id]
      [date]
      --,[date_quality]
      --,[hydrological_year]
      ,[canton]
      ,[municipality]
      ,[start_zone_coordinates_latitude]
      ,[start_zone_coordinates_longitude]
      --,[coordinates_quality]
      ,[start_zone_elevation]
      --,[start_zone_slope_aspect]
      --,[start_zone_inclination]
      --,[forecasted_dangerlevel_rating1]
      --,[forecasted_dangerlevel_rating2]
      --,[forecasted_most_dangerous_aspect_and_elevation]
      ,[number_dead]
      ,[number_caught]
      ,[number_fully_buried]
      --,[activity]
  INTO [FHGR_DV_HS24].[dbo].[01_SLF_hist_avalanche_accidents_clean]
  FROM [FHGR_DV_HS24].[dbo].[01_SLF_hist_avalanche_accidents]
  where [date] > '1991-12-31'
  order by [date] asc;
    -- 3'651 records

-- limit data to > '1998-01-01'
select
	*
from [FHGR_DV_HS24].[dbo].[01_SLF_hist_avalanche_accidents_clean]
where [date] > '1997-12-31'
order by [date] asc;
	-- 3'301 records
