import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas import *
import datetime

def taf_to_cfs(Q):
  return Q * 1000 / 86400 * 43560

def cfs_to_taf_d(Q):
  return Q * 2.29568411*10**-5 * 86400 / 1000

def cfs_to_taf_m(Q):
  return Q / 43559.9 / 1000 * 60 * 60 * 24 * 30.4375

def interpolate_consecutive(df, limit):  # Note: interpolates a ROW of data, not a column
  indices_to_drop = []
  current_consec = 0
  column_names = df.columns
  for i in range(0, df.shape[1]):
    if pd.isnull(df.iloc[0,i]):
      if current_consec >= limit:
        for j in range(0,current_consec+1):
          indices_to_drop.append(column_names[i-j])
      else:
        current_consec += 1
    else:
      current_consec = 0

  indices_to_drop2 = []
  for i in indices_to_drop:
    if i not in indices_to_drop2:
      indices_to_drop2.append(i)

  df = df.interpolate(axis=1)
  for year in range(0,df.shape[1]):
    for col in range(0,len(indices_to_drop2)):
      if df.columns[year] == indices_to_drop2[col]:
        df.iloc[:,year] = np.nan

  return df

### SITE NAMES ######################################################################################

df = pd.read_csv('Sites_Master_List_Final.csv', delimiter=',', header=None, skiprows=1) 	# Reads a list of five-digit site codes
sites = df[0].values 	# List of five-digit site codes

### PARAMETERS #####################################################################################

time = 50 				# Rolling window size
cutoff = 0.6 			# Tells the code when to cut off looking for a threshold value
agreement = 0.6			# Model agreement
danger_count = 10		# How many scenarios will be considered vulnerable (starting from the lowest at 2100)
interp_limit = 10		# Interpolation limit (not applicable)

### SETUP ONLY ######################################################################################

dff = pd.read_csv('cmip5_ncar_mon/streamflow_cmip5_ncar_month_FOL_I.csv', 		# Reads monthly ncar (historical) data for the Folsom site
                    index_col='datetime', 
                    parse_dates={'datetime': [0,1]},
                    date_parser=lambda x: pd.datetime.strptime(x, '%Y %m'))

projections_50yr_avg = np.mean(np.mean(dff.loc['1950-10':'2000-9'].pipe(cfs_to_taf_m).resample('AS-OCT').sum())) 	# Turns the ncar data into annual values resampled on the water year (not the calendar year), then takes the mean

cmip_annQ_spag = (dff.loc['1950-10':'2099-9'].pipe(cfs_to_taf_m)		# Reads the CMIP5 data and resamples it on the water year to get annual values, then divides by the ncar mean to normalize
                    .resample('AS-OCT').sum()
                    .rolling(window=time)
                    .mean())/(projections_50yr_avg)

cmip_annQ_spag = cmip_annQ_spag.dropna()	# Drops any N/A values

### SET UP SOLUTION ARRAYS ###########################################################################

saved_false_neg = pd.DataFrame(0, columns=sites, index=cmip_annQ_spag.index)
saved_false_pos = pd.DataFrame(0, columns=sites, index=cmip_annQ_spag.index)
saved_total_neg = pd.DataFrame(0, columns=sites, index=cmip_annQ_spag.index)
saved_total_pos = pd.DataFrame(0, columns=sites, index=cmip_annQ_spag.index)
saved_true_neg = pd.DataFrame(0, columns=sites, index=cmip_annQ_spag.index)
saved_true_pos = pd.DataFrame(0, columns=sites, index=cmip_annQ_spag.index)
saved_false_neg_percent = pd.DataFrame(0, columns=sites, index=cmip_annQ_spag.index)
saved_false_pos_percent = pd.DataFrame(0, columns=sites, index=cmip_annQ_spag.index)
error_matrix = pd.DataFrame(0, columns=sites, index=cmip_annQ_spag.index)

### INDIVIDUAL SITE CALCS ############################################################################

site_count = 0			# Tells it which site to start with (the first one on the list)
for site in sites:		# Calculates one site at a time

  ### Reads historical and CMIP5 data ###
  dff = pd.read_csv('cmip5_ncar_mon/streamflow_cmip5_ncar_month_'+site+'.csv', 		# Gets ncar (historical) data for this site
                    index_col='datetime', 
                    parse_dates={'datetime': [0,1]},
                    date_parser=lambda x: pd.datetime.strptime(x, '%Y %m'))

  projections_50yr_avg = np.mean(np.mean(dff.loc['1950-10':'2000-9'].pipe(cfs_to_taf_m).resample('AS-OCT').sum()))	# Turns the ncar data into annual values resampled on the water year (not the calendar year), then takes the mean

  cmip_annQ_spag = (dff.loc['1950-10':'2099-9'].pipe(cfs_to_taf_m)		# Reads the CMIP5 data and resamples it on the water year to get annual values, takes the mean using the rolling window size, then divides by the ncar mean to normalize
                    .resample('AS-OCT').sum()
                    .rolling(window=time)
                    .mean())/(projections_50yr_avg)

  cmip_annQ_spag = cmip_annQ_spag.dropna()	# Drops any N/A values


  ### Determines which of the scenarios are vulnerable ###
  ordered_scenarios_all = cmip_annQ_spag.iloc[cmip_annQ_spag.shape[0]-1].sort_values()		# Puts the scenarios into numerical order based on the last value
  lowest_all = ordered_scenarios_all.iloc[:danger_count]									# Takes X scenarios with the lowest end of century values 
  new_cutoff = lowest_all.max()																# In case we need it, here's the value that denotes a scenario vulnerable for this river

  ### Sets up the arrays to count true/false/total positives and negatives ###
  false_positives = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
  false_negatives = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
  false_positives_drought = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
  false_negatives_drought = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
  true_positives = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
  true_negatives = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
  total_positives = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
  total_negatives = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)

  ### Leave-one-out Methodology ###
  drought_error_count = 0 								# Start with 0 errors for this site
  for scenario in range(0, cmip_annQ_spag.shape[1]): 	# Go through code one scenario at a time (there are 97 scenarios)

    test_value_end = cmip_annQ_spag.iloc[cmip_annQ_spag.shape[0]-1,scenario]									# The last value (at 2100) of the removed scenario
    cmip_annQ_spag_lack = cmip_annQ_spag.drop(cmip_annQ_spag.columns[scenario], axis=1)							# All of the scenarios except the one removed
    CMIP5_filtered = pd.DataFrame(0, columns=cmip_annQ_spag_lack.columns, index=cmip_annQ_spag_lack.index)		# Make array
    just_drought_end = pd.DataFrame(0, columns=cmip_annQ_spag_lack.index, index=np.arange(1))					# Make array 
    threshold = pd.DataFrame(0, columns=cmip_annQ_spag_lack.index, index=np.arange(1))							# Make array

    ### 10 lowest end scenarios selected ###
    ordered_scenarios = cmip_annQ_spag_lack.iloc[cmip_annQ_spag_lack.shape[0]-1].sort_values()		# Puts remaining 96 scenarios into numerical order
    lowest = ordered_scenarios.iloc[:danger_count].index 											# Index names of the lowest 10 scenarios (the vulnerable ones)
    higher = ordered_scenarios.iloc[danger_count::].index 											# Index names of the non-vulnerable scenarios
    CMIP5_filtered_end = cmip_annQ_spag_lack.loc[:, lowest] 										# The CMIP5 data for the vulnerable scenarios
    CMIP5_non_filtered_end = cmip_annQ_spag_lack.loc[:, higher] 									# The CMIP5 data for the non-vulnerable scenarios

	### Finding the Threshold ### 
    for year in range(0,CMIP5_non_filtered_end.shape[0]):						# For all 100 years, one year at a time...
      just_drought_end.iloc[0,year] = CMIP5_filtered_end.iloc[year].mean()		# The mean of vulnerable scenarios for each year
      danger_sorted = CMIP5_filtered_end.iloc[year].sort_values()				# The sorted vulnerable scenario values for that year
      reg_sorted = CMIP5_non_filtered_end.iloc[year].sort_values()				# The sorted non-vulnerable scenario values for that year
      tests = np.arange(1.05,new_cutoff,-.001)									# To find threshold value: from 1.05 to the new vulnerability cutoff in increments of .001 (from high to low)
      for number in tests:														# To test threshold values
        danger_number = danger_sorted[danger_sorted < number].count()			# Of the vulnerable scenarios in that year, a count of the ones below the threshold value being tested
        reg_number = reg_sorted[reg_sorted < number].count()					# Of the non-vulnerable scenarios in that year, a count of the ones below the threshold value being tested
        if (danger_number >= ((danger_number+reg_number)*agreement) and 		# If the number of vulnerable scenarios is greater than or equal to the sum of vulnerable and non-vulnerable scenarios multiplied by the model agreement...
          danger_number != 0): 													# And if the number of scenarios is non-zero... 
            threshold.iloc[0,year] = number 									# ... Then this is our threshold value, so save it
            break 																# ... And end this loop so we don't overwrite our threshold value

    ### Interpolation ###
    threshold[threshold == 0] = np.nan 											# Make all of the zero values N/A
    threshold = threshold.interpolate(axis=1) 									# Interpolate
    # threshold = interpolate_consecutive(threshold, interp_limit)				# Alternate option: Interpolate with a limit (e.g. only over gaps less than 10 years, leaving anything larger as missing values)
    threshold.iloc[0,:] = threshold.iloc[0,:].fillna(value=0) 					# Change any remaining N/A values to zeros
    for year in range(0,CMIP5_non_filtered_end.shape[0]): 						# For all 100 years, one year at a time...
      if threshold.iloc[0,year] == 0: 											# If the threshold value is zero...
        drought_error_count += 1 												# ... Add one error to the total count
        error_matrix.iloc[year,site_count] += 1    								# ... And add one error to the error matrix at the correct year and site

    ### Error Calculations ###
    for year in range(0,threshold.shape[1]):									# For all 100 years, one year at a time...
      if (cmip_annQ_spag.iloc[year,scenario] > threshold.iloc[0,year] and 		# If the value from the scenario being tested is greater than the threshold value (indicating non-vulnerable)...
          threshold.iloc[0,year] != 0 and 										# And if the threshold value is non-zero (not a missing point)...
          test_value_end < new_cutoff): 										# And if the scenario being testd is actually vulnerable...
        false_negatives.iloc[year,scenario] += 1 								# ... Then count a false negative for this year and scenario

      if (cmip_annQ_spag.iloc[year,scenario] > threshold.iloc[0,year] and 		# If the value from the scenario being tested is greater than the threshold value (indicating non-vulnerable)...
          threshold.iloc[0,year] != 0 and 										# And if the threshold value is non-zero (not a missing point)...
          test_value_end > new_cutoff): 										# And if the scenario being tested is actually non-vulnerable...
        true_negatives.iloc[year,scenario] += 1 								# ... Then count a true negative for this year and scenario

      if (cmip_annQ_spag.iloc[year,scenario] < threshold.iloc[0,year] and 		# If the value from the scenario being tested is less than the threshold value (indicating vulnerable)...
          threshold.iloc[0,year] != 0 and 										# And if the threshold value is non-zero (not a missing point)...
          test_value_end > new_cutoff): 										# And if the scenario being tested is actually non-vulnerable...
        false_positives.iloc[year,scenario] += 1 								# ... Then count a false positive for this year and scenario

      if (cmip_annQ_spag.iloc[year,scenario] < threshold.iloc[0,year] and 		# If the value from the scenario being tested is less than the threshold value (indicating vulnerable)...
          threshold.iloc[0,year] != 0 and 										# And if the threshold value is non-zero (not a missing point)...
          test_value_end < new_cutoff): 										# And if the scenario being tested is actually vulnerable...
        true_positives.iloc[year,scenario] += 1 								# ... Then count a true positive for this year and scenario

      if (threshold.iloc[0,year] != 0 and 										# If the threshold value is non-zero (not a missing point)...
          test_value_end < new_cutoff): 										# And if the scenario being tested is actually vulnerable...
        total_positives.iloc[year,scenario] += 1 								# ... Then count a positive

      if (threshold.iloc[0,year] != 0 and 										# If the threshold value is non-zero (not a missing point)...
          test_value_end > new_cutoff): 										# And if the scenario being tested is actually non-vulnerable...
        total_negatives.iloc[year,scenario] += 1 								# ... Then count a negative
    
    threshold[threshold == 0] = np.nan 											# Change all threshold zero-values to N/A

  ### To Print Results for this Site ###
  mix_FP_poss = (false_positives.values.sum().sum()) / (total_negatives.sum().sum()) 					# The total percent of false positives
  mix_FN_poss = (false_negatives.values.sum().sum()) / (total_positives.sum().sum()) 					# The total percent of false negatives
  mix_TP_poss = (true_positives.values.sum().sum()) / (total_positives.sum().sum()) 					# The total percent of true positives
  mix_TN_poss = (true_negatives.values.sum().sum()) / (total_negatives.sum().sum()) 					# The total percent of true negatives

  print('-----------------------------------------------------')
  print('For {}, {} out of 91:'.format(site, (site_count+1))) 											# Print the site code and site number
  print('There were {} missing values out of a possible 9,700'.format(drought_error_count)) 			# Print the total number of missing points
  print('False positives ({}): {:.2f}%'.format(false_positives.values.sum().sum(),mix_FP_poss*100)) 	# Print false positives
  print('False negatives ({}): {:.2f}%'.format(false_negatives.values.sum().sum(),mix_FN_poss*100)) 	# Print false negatives
  print('True positives ({}): {:.2f}%'.format(true_positives.values.sum().sum(),mix_TP_poss*100)) 		# Print true positives
  print('True negatives ({}): {:.2f}%'.format(true_negatives.values.sum().sum(),mix_TN_poss*100)) 		# Print true negatives
  print('Check: {}'.format(mix_FP_poss+mix_FN_poss+mix_TP_poss+mix_TN_poss)) 							# Make sure that everything sums to 2 (because FP+TN=1 and FN+TP=1)

  ### Saving this site's results to main arrays ###
  saved_false_neg.iloc[:,site_count] = false_negatives.values.sum(axis=1)
  saved_false_pos.iloc[:,site_count] = false_positives.values.sum(axis=1)
  saved_total_neg.iloc[:,site_count] = total_negatives.values.sum(axis=1)
  saved_total_pos.iloc[:,site_count] = total_positives.values.sum(axis=1)
  saved_false_neg_percent.iloc[:,site_count] = false_negatives.values.sum(axis=1) / total_positives.values.sum(axis=1) *100
  saved_false_pos_percent.iloc[:,site_count] = false_positives.values.sum(axis=1) / total_negatives.values.sum(axis=1) *100
  saved_true_neg.iloc[:,site_count] = true_negatives.values.sum(axis=1)
  saved_true_pos.iloc[:,site_count] = true_positives.values.sum(axis=1)

  site_count += 1 	# Move on to the next site on the list

### SAVING FILES ############################################################################################################

saved_false_neg.to_csv('{}-yr_False_Negatives_with_{}-lowest_and_{}%_agreement_with_interpolation_all.csv'.format(time,danger_count,agreement*100))
saved_false_pos.to_csv('{}-yr_False_Positives_with_{}-lowest_and_{}%_agreement_with_interpolation_all.csv'.format(time,danger_count,agreement*100))
saved_total_neg.to_csv('{}-yr_Total_Negatives_with_{}-lowest_and_{}%_agreement_with_interpolation_all.csv'.format(time,danger_count,agreement*100))
saved_total_pos.to_csv('{}-yr_Total_Positives_with_{}-lowest_and_{}%_agreement_with_interpolation_all.csv'.format(time,danger_count,agreement*100))
saved_false_neg_percent.to_csv('{}-yr_Percent_False_Negatives_with_{}-lowest_and_{}%_agreement_with_interpolation_all.csv'.format(time,danger_count,agreement*100))
saved_false_pos_percent.to_csv('{}-yr_Percent_False_Positives_with_{}-lowest_and_{}%_agreement_with_interpolation_all.csv'.format(time,danger_count,agreement*100))
saved_true_neg.to_csv('{}-yr_True_Negatives_with_{}-lowest_and_{}%_agreement_with_interpolation_all.csv'.format(time,danger_count,agreement*100))
saved_true_pos.to_csv('{}-yr_True_Positives_with_{}-lowest_and_{}%_agreement_with_interpolation_all.csv'.format(time,danger_count,agreement*100))
error_matrix.to_csv('{}-yr_Error_Matrix_with_{}-lowest_and_{}%_agreement_with_interpolation_all.csv'.format(time,danger_count,agreement*100))

############################################################################################################################

print('Finished')
