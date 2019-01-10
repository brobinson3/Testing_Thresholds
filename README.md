# Testing_Thresholds

This code was used for the computational work in the following paper:

Robinson, B., & Herman, J. D. (2018). A framework for testing dynamic classification of vulnerable scenarios in ensemble water supply projections. Climatic Change, 1-18. (found at: https://link.springer.com/article/10.1007/s10584-018-2347-3)

The CMIP5 downscaled streamflow projections can be found at <ftp://gdo-dcp.ucllnl.org/pub/dcp/archive/cmip5/hydro/routed_streamflow/>. The projections used for this paper are in the folder "cmip5_ncar_mon", and the historical flows used are in the folder "historical_maurer02". The original list of the sites included and their five-letter codes is in the file marked NCAR_cmip5_streamflow_sites.txt, however a list of only the sites used for this study and their relevant data is in this repository (Sites_Master_List_Final.csv).

## Sites_Master_List_Final.csv

Only the first column (river name codes) of this file is used for Thresholds_Code_Final.py. Other information here is used for drawing the map and to keep track of additional data sources. The first set of longitude and latitude values is directly from the CMIP5 downscaled streamflow projections data (listed above), and the second set of longitude and latitude values is what is listed by each site's individual agencies (they are not always the same).

## To Run Thresholds_Code_Final.py

1. Download the files included in this repository and the data folder "cmip5_ncar_mon" from the site referenced above.
2. Make sure that you have the necessary libraries installed (numpy, matplotlib, and pandas).
3. Check that the location of Sites_Master_List_Final.csv in line 46 is correct.
4. Check that the location of the "cmip_ncar_mon" folder in lines 59 and 91 is correct.
5. Run and wait. It should take a while before the first print statement appears.

## Remaking Figures

All information necessary to remake the figures in this paper is either in the "Making_Figures" folder, in the "Multiple_Source_Historical_Data" folder, or is an output of the main script "Thresholds_Code_Final.py". If something is missing I would be happy to include it.
