-- measurements
SELECT
		'001' as nr
	  ,[station_code]
      ,LEFT([measure_date], 10) + 'T00:00:00Z' AS measure_date
      ,[HS]
      ,[TA_30MIN_MEAN]
      ,[RH_30MIN_MEAN]
      ,[TSS_30MIN_MEAN]
      ,[TS0_30MIN_MEAN]
      ,[TS25_30MIN_MEAN]
      ,[TS50_30MIN_MEAN]
      ,[TS100_30MIN_MEAN]
      ,[RSWR_30MIN_MEAN]
      ,[VW_30MIN_MEAN]
      ,[VW_30MIN_MAX]
      ,[DW_30MIN_MEAN]
      ,[DW_30MIN_SD]
  FROM [FHGR_DV_HS24].[dbo].[02_SLF_hist_daily_measurements]
  where date > '2022-12-31';

-- daily snow
  SELECT 
    '001' AS nr,
    [station_code],
    LEFT([measure_date], 10) + 'T00:00:00' AS measure_date, -- Nur die ersten 19 Zeichen behalten
    [HS],
    [HN_1D]
FROM [FHGR_DV_HS24].[dbo].[03_SLF_hist_daily_snow]
WHERE hyear > '2022';
