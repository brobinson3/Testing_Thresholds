import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas import *
import seaborn as sns
import datetime
plt.show(block=False)

def init_plotting():
  sns.set_style('whitegrid')
  sns.set_style('whitegrid')
  plt.rcParams['figure.figsize'] = (9, 7)
  plt.rcParams['font.size'] = 10
  plt.rcParams['axes.labelsize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['axes.titlesize'] = 1.1*plt.rcParams['font.size']
  plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
  plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
  plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']

def taf_to_cfs(Q):
  return Q * 1000 / 86400 * 43560

def cfs_to_taf_d(Q):
  return Q * 2.29568411*10**-5 * 86400 / 1000

def cfs_to_taf_m(Q):
  return Q / 43559.9 / 1000 * 60 * 60 * 24 * 30.4375

def interpolate_consecutive(df, limit):
  # Note: interpolates a ROW of data, not a column
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

#########################################################################################

site = 'MERWI'
time = 50
cutoff = 0.6
agreement = 0.7
danger_count = 10

init_plotting()

dff = pd.read_csv('cmip5_ncar_mon/streamflow_cmip5_ncar_month_'+site+'.csv', 
                    index_col='datetime', 
                    parse_dates={'datetime': [0,1]},
                    date_parser=lambda x: pd.datetime.strptime(x, '%Y %m'))

projections_50yr_avg = np.mean(np.mean(dff.loc['1950-10':'2000-9'].pipe(cfs_to_taf_m).resample('AS-OCT').sum()))

annual_spag = (dff.loc['1950-10':'2099-9'].pipe(cfs_to_taf_m)
                  .resample('AS-OCT').sum())

cmip_annQ_spag2 = (dff.loc['1950-10':'2099-9'].pipe(cfs_to_taf_m)
                    .resample('AS-OCT').sum()
                    .rolling(window=time)
                    .mean())

cmip_annQ_spag = (dff.loc['1950-10':'2099-9'].pipe(cfs_to_taf_m)
                    .resample('AS-OCT').sum()
                    .rolling(window=time)
                    .mean())/(projections_50yr_avg)

cmip_annQ_spag = cmip_annQ_spag.dropna()

CMIP5_filtered = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
CMIP5_filtered_any = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
CMIP5_filtered_end = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)

drought_error_count = 0

scenarios_all = cmip_annQ_spag.tail(1)
ordered_scenarios_all = cmip_annQ_spag.iloc[99].sort_values()
lowest_all = ordered_scenarios_all.iloc[:danger_count]
new_cutoff = lowest_all.max()

false_positives = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
false_negatives = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
false_positives_drought = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
false_negatives_drought = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
true_positives = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)
true_negatives = pd.DataFrame(0, columns=cmip_annQ_spag.columns, index=cmip_annQ_spag.index)

ordered_scenarios = cmip_annQ_spag.iloc[99].sort_values()
lowest = ordered_scenarios.iloc[:danger_count].index
higher = ordered_scenarios.iloc[(danger_count)::].index
CMIP5_filtered_end = cmip_annQ_spag.loc[:, lowest]
CMIP5_filtered_end2 = cmip_annQ_spag2.loc[:, lowest]


# FILTER

CMIP5_non_filtered_end = cmip_annQ_spag.loc[:, higher]
CMIP5_filtered = CMIP5_filtered.loc[:, (CMIP5_filtered != 0).any(axis=0)]
just_drought_end = pd.DataFrame(0, columns=cmip_annQ_spag.index, index=np.arange(1))
half_half_end = pd.DataFrame(0, columns=cmip_annQ_spag.index, index=np.arange(1))

for year in range(0,CMIP5_filtered.shape[0]):
  just_drought_end.iloc[0,year] = CMIP5_filtered_end.iloc[year].mean()
  danger_sorted = CMIP5_filtered_end.iloc[year].sort_values()
  reg_sorted = CMIP5_non_filtered_end.iloc[year].sort_values()
  tests = np.arange(1.02,cutoff,-.001)
  for number in tests:
    danger_number = danger_sorted[danger_sorted < number].count()
    reg_number = reg_sorted[reg_sorted < number].count()
    if (danger_number >= ((danger_number+reg_number)*agreement) and
        danger_number != 0):
          half_half_end.iloc[0,year] = number
          break
  tests2 = np.arange(cutoff-.1,1.1,.001)
  for number in tests2:
    danger_number = danger_sorted[danger_sorted < number].count()
    if (danger_number >= (danger_sorted.shape[0])*agreement and
        danger_number != 0):
      just_drought_end.iloc[0,year] = number
      break

half_half_end[half_half_end == 0] = np.nan
half_half_end = interpolate_consecutive(half_half_end, 10)
half_half_end.iloc[0,:] = half_half_end.iloc[0,:].fillna(value=0)
for year in range(0,CMIP5_filtered.shape[0]):
  if half_half_end.iloc[0,year] == 0:
        drought_error_count += 1


#########################################################################################
### Leave-One-Out Testing ###

for scenario in range(1,2):
    test_value_end = cmip_annQ_spag.iloc[cmip_annQ_spag.shape[0]-1,scenario]
    cmip_annQ_spag_lack = cmip_annQ_spag.drop(cmip_annQ_spag.columns[scenario], axis=1)
    CMIP5_filtered_t = pd.DataFrame(0, columns=cmip_annQ_spag_lack.columns, index=cmip_annQ_spag_lack.index)
    CMIP5_filtered_any_t = pd.DataFrame(0, columns=cmip_annQ_spag_lack.columns, index=cmip_annQ_spag_lack.index)


    ### 10 lowest end scenarios selected ###############################################################
    ordered_scenarios_t = cmip_annQ_spag_lack.iloc[99].sort_values()
    lowest_t = ordered_scenarios_t.iloc[:danger_count].index
    higher_t = ordered_scenarios_t.iloc[danger_count::].index
    CMIP5_filtered_end_t = cmip_annQ_spag_lack.loc[:, lowest]
    CMIP5_non_filtered_end_t = cmip_annQ_spag_lack.loc[:, higher]

    CMIP5_filtered_t = CMIP5_filtered_t.loc[:, (CMIP5_filtered_t != 0).any(axis=0)]
    CMIP5_filtered_end_t = CMIP5_filtered_end_t.loc[:, (CMIP5_filtered_end_t != 0).any(axis=0)]
    half_half_end_t = pd.DataFrame(0, columns=cmip_annQ_spag_lack.index, index=np.arange(1))
 
    for year in range(0,CMIP5_filtered_t.shape[0]):
      danger_sorted_t = CMIP5_filtered_end_t.iloc[year].sort_values()
      reg_sorted_t = CMIP5_non_filtered_end_t.iloc[year].sort_values()
      tests_t = np.arange(1.05,new_cutoff,-.001)
      for number in tests_t:
        danger_number_t = danger_sorted_t[danger_sorted_t < number].count()
        reg_number_t = reg_sorted_t[reg_sorted_t < number].count()
        if (danger_number_t >= ((danger_number_t+reg_number_t)*agreement) and
          danger_number_t != 0):
            half_half_end_t.iloc[0,year] = number
            break
      tests2_t = np.arange(new_cutoff-.1,1.1,.001)
      for number in tests2_t:
        danger_number_t = danger_sorted_t[danger_sorted_t < number].count()
        if (danger_number_t >= (danger_sorted_t.shape[0])*agreement and
          danger_number_t != 0):
          break


    ### INTERPOLATION ####################################################################
    
    half_half_end_t[half_half_end_t == 0] = np.nan
    half_half_end_t = interpolate_consecutive(half_half_end_t, 10)


#########################################################################################

false_positives = pd.DataFrame(0, columns=cmip_annQ_spag.index, index=np.arange(1))
false_negatives = pd.DataFrame(0, columns=cmip_annQ_spag.index, index=np.arange(1))

false_positives_drought = pd.DataFrame(0, columns=cmip_annQ_spag.index, index=np.arange(1))
false_negatives_drought = pd.DataFrame(0, columns=cmip_annQ_spag.index, index=np.arange(1))

for year in range(0,half_half_end.shape[1]):
  for scenario in range(0,CMIP5_filtered_end.shape[1]):
    if (CMIP5_filtered_end.iloc[year,scenario] > half_half_end.iloc[0,year] and
        half_half_end.iloc[0,year] != 0):
      false_negatives.iloc[0,year] += 1

    if (CMIP5_filtered_end.iloc[year,scenario] > just_drought_end.iloc[0,year] and
        just_drought_end.iloc[0,year] != 0):
      false_negatives_drought.iloc[0,year] += 1

  for scenario in range(0,CMIP5_non_filtered_end.shape[1]): 
    if (CMIP5_non_filtered_end.iloc[year,scenario] < half_half_end.iloc[0,year] and
        half_half_end.iloc[0,year] != 0):
      false_positives.iloc[0,year] += 1

    if (CMIP5_non_filtered_end.iloc[year,scenario] < just_drought_end.iloc[0,year] and
        just_drought_end.iloc[0,year] != 0):
      false_positives_drought.iloc[0,year] += 1

half_half_end[half_half_end == 0] = np.nan


### Making Figure 3 ###############################################################################
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

cmip_annQ_spag2.iloc[:,1].plot(legend=None, color='cornflowerblue', ax=ax2)
cmip_annQ_spag2.iloc[:,1].plot(legend=None, color='gray', ax=ax2)

annual_spag.plot(legend=None, ax=ax1)
ax1.set_ylabel('Annual Flow (TAF)', fontsize=10, fontweight='semibold')
ax1.set_xlabel('', fontsize=10, fontweight='semibold')
ax1.set_title('(A) GCM-based Flow Data', fontsize=11, fontweight='bold')

cmip_annQ_spag2.plot(legend=None, color='gray', ax=ax2)
CMIP5_filtered_end2.plot(legend=None, color='cornflowerblue', ax=ax2)
ax2.set_ylabel('50-yr MA Annual Flow (TAF)', fontsize=10, fontweight='semibold')
ax2.set_xlabel('', fontsize=10, fontweight='semibold')
ax2.set_title('(B) 50-yr Moving Average', fontsize=11, fontweight='bold')
ax2.set_xlim('2000','2099')

cmip_annQ_spag.plot(legend=None, color='lightgray', ax=ax3)
CMIP5_filtered_end.plot(legend=None, color='cornflowerblue', ax=ax3)
half_half_end.iloc[0,:].plot(legend=None, color='indianred', ax=ax3)
half_half_end.iloc[0,0:51].plot(legend=None, color='r', ax=ax3)
ax3.set_ylabel('Normalized 50-yr MA Annual Flow', fontsize=10, fontweight='semibold')
ax3.set_xlabel('', fontsize=10, fontweight='semibold')
ax3.set_title('(C) Identifying the Threshold', fontsize=11, fontweight='bold')
ax3.set_xlim('2000','2099')

cmip_annQ_spag.plot(legend=None, color='lightgray', ax=ax4)
CMIP5_filtered_end.plot(legend=None, color='cornflowerblue', ax=ax4)
half_half_end_t.iloc[0,:].plot(legend=None, color='r', ax=ax4)
cmip_annQ_spag.iloc[:,1].plot(legend=None, color='limegreen', ax=ax4)
ax4.set_ylabel('Normalized 50-yr MA Annual Flow', fontsize=10, fontweight='semibold')
ax4.set_xlabel('', fontsize=10, fontweight='semibold')
ax4.set_title('(D) Leave-One-Out Testing', fontsize=11, fontweight='bold')
ax4.set_xlim('2000','2099')

ax1.set_xticks(['1950','1975','2000','2025','2050','2075','2100'])
ax1.set_xticklabels([1950,1975,2000,2025,2050,2075,2100], size=10, fontweight='semibold')
ax2.set_xticks(['2000','2025','2050','2075','2100'])
ax2.set_xticklabels([2000,2025,2050,2075,2100], size=10, fontweight='semibold')
ax3.set_xticks(['2000','2025','2050','2075','2100'])
ax3.set_xticklabels([2000,2025,2050,2075,2100], size=10, fontweight='semibold')
ax4.set_xticks(['2000','2025','2050','2075','2100'])
ax4.set_xticklabels([2000,2025,2050,2075,2100], size=10, fontweight='semibold')

ax1.set_yticks([1000,2000,3000,4000,5000,6000])
ax1.set_yticklabels([1000,2000,3000,4000,5000,6000], size=10, fontweight='semibold')
ax2.set_yticks([2500,2750,3000,3250,3500,3750,4000])
ax2.set_yticklabels([2500,2750,3000,3250,3500,3750,4000], size=10, fontweight='semibold')
ax3.set_yticks([0.8,0.9,1.0,1.1,1.2])
ax3.set_yticklabels([0.8,0.9,1.0,1.1,1.2], size=10, fontweight='semibold')
ax4.set_yticks([0.8,0.9,1.0,1.1,1.2])
ax4.set_yticklabels([0.8,0.9,1.0,1.1,1.2], size=10, fontweight='semibold')

ax3.axvline(pd.to_datetime('2049'), color='k', linestyle='--')
ax2.legend(['Vulnerable','Not Vulnerable'],frameon=True,fontsize=8,edgecolor='k')
ax3.text(pd.to_datetime('2050'), .877, ']', fontweight='black', fontsize=9)
ax3.annotate('', xy=(pd.to_datetime('2052'),0.885), xytext=(pd.to_datetime('2062'),0.82), fontweight='bold', va="center", ha="center", arrowprops=dict(facecolor='k', width=2, headwidth=9, headlength=10, shrink=0.03),)
ax3.text(pd.to_datetime('2065'), 0.78, '60% Model\nAgreement', fontweight='bold', va='center', ha='center')
ax3.text(pd.to_datetime('2051'), 1.19, 't', fontweight='bold', fontstyle='italic')
thresh_line, = plt.plot([],[],color='r')
ax3.legend([thresh_line],['Threshold'],frameon=True,fontsize=8,edgecolor='k', loc='lower left')

ax4.annotate('True Positive', xy=(pd.to_datetime('2075'),0.862), xytext=(pd.to_datetime('2075'),0.76), fontweight='bold', va="center", ha="center", arrowprops=dict(facecolor='k', width=2, headwidth=9, headlength=10, shrink=0.05),)
ax4.annotate('False\nNegative', xy=(pd.to_datetime('2059'),0.897), xytext=(pd.to_datetime('2062'),1.02), fontweight='bold', va="center", ha="center", arrowprops=dict(facecolor='k', width=2, headwidth=9, headlength=10, shrink=0.05),)
rem, = plt.plot([],[],color='limegreen')
ax4.legend([thresh_line,rem],['Threshold','One Scenario Removed\nfrom Consideration'],frameon=True,fontsize=8,edgecolor='k', loc='lower left')

f.subplots_adjust(hspace=.3, wspace=.3)

plt.savefig('Figure 3.svg',bbox_inches="tight")

plt.show()



