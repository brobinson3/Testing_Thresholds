# Testing_Thresholds

This code was used for the computational work in the following paper:

Robinson, B., & Herman, J. D. (2018). A framework for testing dynamic classification of vulnerable scenarios in ensemble water supply projections. Climatic Change, 1-18. (found at: https://link.springer.com/article/10.1007/s10584-018-2347-3)

This code is a statistical testing approach for threshold-based classifiers of vulnerable scenarios in long-term streamflow projections. For the paper, this code is run over many combinations of 'time' (window size) and 'agreement' parameters. A user would have to replicate all of those runs to replicate the work in that paper and to make all of the figures included.

The CMIP5 downscaled streamflow projections can be found at ftp://gdo-dcp.ucllnl.org/pub/dcp/archive/cmip5/hydro/routed_streamflow/. The projections used for this paper are in the folder "cmip5_ncar_mon", and the historical flows used are in the folder "historical_maurer02". The original list of the sites included and their five-letter codes is in the file marked NCAR_cmip5_streamflow_sites.txt, however a list of only the sites used for this study and their relevant data is in this repository (Sites_Master_List_Final.csv).

## Sites_Master_List_Final.csv

Only the first column (river name codes) of this file is used for Thresholds_Code_Final.py. Other information here is used for drawing the map and to keep track of additional data sources. The first set of longitude and latitude values is directly from the CMIP5 downscaled streamflow projections data (listed above), and the second set of longitude and latitude values is what is listed by each site's individual agencies (they are not always the same).

## To Run Thresholds_Code_Final.py

1. Download the files included in this repository and the data folder "cmip5_ncar_mon" from the site referenced above.
2. Make sure that you have the necessary libraries installed (numpy, matplotlib, and pandas).
3. Check that the "cmip_ncar_mon" folder is in the "Testing_Thresholds" folder you downloaded.
4. Run and wait. It should take about 25 minutes before the first print statement appears.

## Remaking Figures

All information necessary to remake the figures in this paper is either in the "Making_Figures" folder, in the "Multiple_Source_Historical_Data" folder, or is an output of the main script "Thresholds_Code_Final.py". The "Multiple_Source_Historical_Data" is data combined from the "historical_maurer02" data (from the CMIP5 data) and data sourced from local agencies (such as CDEC, NRCS, or USGS). This data is then used for showing current historical streamflows after the year 2000 in some figures (it is not, however, used in any calculations).
