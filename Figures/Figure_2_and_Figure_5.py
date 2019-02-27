import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas import *
import seaborn as sns
import datetime

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

sites = ['PRIES','FOL_I','MRFPD','CAMEO']
time = 50
cutoff = 0.6
agreement = 0.6
danger_count = 10

init_plotting()

for site in sites:
  dfh = pd.read_csv('historical_maurer02/streamflow_vic4.1.2_hist_ncar_month.csv', 
                index_col='datetime', 
                parse_dates={'datetime': [0,1]},
                date_parser=lambda x: pd.datetime.strptime(x, '%Y %m'))

  dfh_new = pd.read_csv('Multiple Source Historical Data/'+site+'_Historical_new.csv',
                names=['Datetime','Q'],
                index_col='Datetime',
                parse_dates=True)

  dff = pd.read_csv('cmip5_ncar_mon/streamflow_cmip5_ncar_month_'+site+'.csv', 
                    index_col='datetime', 
                    parse_dates={'datetime': [0,1]},
                    date_parser=lambda x: pd.datetime.strptime(x, '%Y %m'))

  hist_avg = dfh_new.loc['1950-10':'2099-9'].resample('AS-OCT').sum().mean().values
  
  if (site == 'CAMEO' or site == 'PRIES' or site == 'FOL_I'):
    hist = (dfh_new.loc['1950-10':'2099-9'].resample('AS-OCT').sum().rolling(window=time).mean())/hist_avg
  if site == 'MRFPD':
    hist = (dfh_new.loc['1934-10':'2099-9'].resample('AS-OCT').sum().rolling(window=time).mean())/hist_avg

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

  historical = hist.dropna()

  hist_end_value = historical.tail(1)
  hist_end_year = pd.to_datetime(hist_end_value.index).year

  year = hist_end_year
  change = hist_end_value

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

  half_half_end[half_half_end == 0] = np.nan
  if site == 'PRIES':
    cmip_annQ_spag_PRIES = cmip_annQ_spag
    CMIP5_filtered_end_PRIES = CMIP5_filtered_end
    historical_PRIES = historical
    half_half_end_PRIES = half_half_end
  if site == 'FOL_I':
    cmip_annQ_spag_FOL_I = cmip_annQ_spag
    CMIP5_filtered_end_FOL_I = CMIP5_filtered_end
    historical_FOL_I = historical
    half_half_end_FOL_I = half_half_end
  if site == 'MRFPD':
    cmip_annQ_spag_MRFPD = cmip_annQ_spag
    CMIP5_filtered_end_MRFPD = CMIP5_filtered_end
    historical_MRFPD = historical
    half_half_end_MRFPD = half_half_end
  if site == 'CAMEO':
    cmip_annQ_spag_CAMEO = cmip_annQ_spag
    CMIP5_filtered_end_CAMEO = CMIP5_filtered_end
    historical_CAMEO = historical
    half_half_end_CAMEO = half_half_end


### Figure 2 ###################################################################################################################

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
t1 = '(A) Colorado River (Colorado Basin)'
t2 = '(B) Missouri River (Missouri Basin)'
t3 = '(C) Columbia River (Pacific Northwest Basin)'
t4 = '(D) American River (California Basin)'

rcpcolors = ['#ffb3b3', '#ff3333', '#b30000', '#330000'] #['#b3d1ff', '#3385ff', '#0047b3', '#001433'] #Blue color scheme
rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']

historical_MRFPD.plot(legend=None, color='#ffb3b3', ax=ax2, linewidth=0.5)
historical_MRFPD.plot(legend=None, color='#ff3333', ax=ax2, linewidth=0.5)
historical_MRFPD.plot(legend=None, color='#b30000', ax=ax2, linewidth=0.5)
historical_MRFPD.plot(legend=None, color='#330000', ax=ax2, linewidth=0.5)
historical_CAMEO.plot(legend=None, color='k', ax=ax1, linewidth=2)
historical_MRFPD.plot(legend=None, color='k', ax=ax2, linewidth=2)
historical_PRIES.plot(legend=None, color='k', ax=ax3, linewidth=2)
historical_FOL_I.plot(legend=None, color='k', ax=ax4, linewidth=2)

for r,c in zip(rcps, rcpcolors):
  (cmip_annQ_spag_CAMEO.loc['2016'::].filter(regex=r)
               .plot(legend=None, color=c, ax=ax1, linewidth=0.5))
for r,c in zip(rcps, rcpcolors):
  (cmip_annQ_spag_MRFPD.loc['2015'::].filter(regex=r)
               .plot(legend=None, color=c, ax=ax2, linewidth=0.5))
for r,c in zip(rcps, rcpcolors):
  (cmip_annQ_spag_PRIES.loc['2016'::].filter(regex=r)
               .plot(legend=None, color=c, ax=ax3, linewidth=0.5))
for r,c in zip(rcps, rcpcolors):
  (cmip_annQ_spag_FOL_I.loc['2016'::].filter(regex=r)
               .plot(legend=None, color=c, ax=ax4, linewidth=0.5))

ax1.axvline(pd.to_datetime('2016'), color='k', linestyle='--')
ax2.axvline(pd.to_datetime('2015'), color='k', linestyle='--')
ax3.axvline(pd.to_datetime('2016'), color='k', linestyle='--')
ax4.axvline(pd.to_datetime('2016'), color='k', linestyle='--')

ax1.set_xlabel('', fontsize=8, fontweight='semibold')
ax2.set_xlabel('', fontsize=8, fontweight='semibold')
ax3.set_xlabel('', fontsize=8, fontweight='semibold')
ax4.set_xlabel('', fontsize=8, fontweight='semibold')

ax1.set_title(t1, fontsize=11, fontweight='bold')
ax2.set_title(t2, fontsize=11, fontweight='bold')
ax3.set_title(t3, fontsize=11, fontweight='bold')
ax4.set_title(t4, fontsize=11, fontweight='bold')

ax1.set_yticks([0.8,1.0,1.2,1.4])
ax1.set_yticklabels([0.8,1.0,1.2,1.4], size=10, fontweight='semibold')
ax2.set_yticks([0.8,1.0,1.2,1.4])
ax2.set_yticklabels([0.8,1.0,1.2,1.4], size=10, fontweight='semibold')
ax3.set_yticks([0.8,1.0,1.2,1.4])
ax3.set_yticklabels([0.8,1.0,1.2,1.4], size=10, fontweight='semibold')
ax4.set_yticks([0.4,0.6,0.8,1.0,1.2,1.4,1.6])
ax4.set_yticklabels([0.4,0.6,0.8,1.0,1.2,1.4,1.6], size=10, fontweight='semibold')

ax1.set_ylabel('Normalized 50-yr MA Annual Flow', fontsize=9, fontweight='semibold')
ax3.set_ylabel('Normalized 50-yr MA Annual Flow', fontsize=9, fontweight='semibold')

ax1.set_xticks(['2000','2025','2050','2075','2100'])
ax1.set_xticklabels([2000,2025,2050,2075,2100], size=10, fontweight='semibold')
ax2.set_xticks(['1975','2000','2025','2050','2075','2100'])
ax2.set_xticklabels([1975,2000,2025,2050,2075,2100], size=10, fontweight='semibold')
ax3.set_xticks(['2000','2025','2050','2075','2100'])
ax3.set_xticklabels([2000,2025,2050,2075,2100], size=10, fontweight='semibold')
ax4.set_xticks(['2000','2025','2050','2075','2100'])
ax4.set_xticklabels([2000,2025,2050,2075,2100], size=10, fontweight='semibold')


ax2.legend(['RCP 2.6','RCP 4.5','RCP 6.0','RCP 8.5','Historical'],frameon=True,fontsize=8,edgecolor='k')

f.subplots_adjust(hspace=.3)
# plt.savefig('Figure_2.svg', format='svg', bbox_inches='tight')
plt.show()



### Figure 5 ###################################################################################################

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
t1 = '(A) Colorado River (Colorado Basin)'
t2 = '(B) Missouri River (Missouri Basin)'
t3 = '(C) Columbia River (Pacific Northwest Basin)'
t4 = '(D) American River (California Basin)'

historical_CAMEO.plot(color='cornflowerblue',ax=ax1) #Vulnerable
historical_CAMEO.plot(color='lightgray',ax=ax1) #Not Vulnerable
historical_CAMEO.plot(color='r',ax=ax1) #Threshold
historical_CAMEO.plot(color='k',ax=ax1) #Historical Flow

cmip_annQ_spag_CAMEO.plot(legend=None, color='lightgray', ax=ax1)
CMIP5_filtered_end_CAMEO.plot(legend=None, color='cornflowerblue', ax=ax1)
historical_CAMEO.plot(legend=None, color='k', ax=ax1)
half_half_end_CAMEO.iloc[0,:].plot(legend=None, color='r', ax=ax1)
ax1.set_title(t1, fontsize=11, fontweight='bold')
ax1.set_xlabel('', fontsize=8, fontweight='semibold')
ax1.set_ylabel('Normalized 50-yr MA Annual Flow', fontsize=9, fontweight='semibold')
ax1.set_xlim('2000','2099')
ax1.set_xticks(['2000','2025','2050','2075','2100'])
ax1.set_xticklabels([2000,2025,2050,2075,2100], size=10, fontweight='semibold')
ax1.set_yticks([0.8,1.0,1.2,1.4])
ax1.set_yticklabels([0.8,1.0,1.2,1.4], size=10, fontweight='semibold')

cmip_annQ_spag_MRFPD.plot(legend=None, color='lightgray', ax=ax2)
CMIP5_filtered_end_MRFPD.plot(legend=None, color='cornflowerblue', ax=ax2)
historical_MRFPD.plot(legend=None, color='k', ax=ax2)
half_half_end_MRFPD.iloc[0,:].plot(legend=None, color='r', ax=ax2)
ax2.set_title(t2, fontsize=11, fontweight='bold')
ax2.set_xlabel('', fontsize=8, fontweight='semibold')
ax2.set_ylabel('', fontsize=8, fontweight='semibold')
ax2.set_xlim('2000','2099')
ax2.set_xticks(['2000','2025','2050','2075','2100'])
ax2.set_xticklabels([2000,2025,2050,2075,2100], size=10, fontweight='semibold')
ax2.set_yticks([0.8,1.0,1.2,1.4])
ax2.set_yticklabels([0.8,1.0,1.2,1.4], size=10, fontweight='semibold')

cmip_annQ_spag_PRIES.plot(legend=None, color='lightgray', ax=ax3)
CMIP5_filtered_end_PRIES.plot(legend=None, color='cornflowerblue', ax=ax3)
historical_PRIES.plot(legend=None, color='k', ax=ax3)
half_half_end_PRIES.iloc[0,:].plot(legend=None, color='r', ax=ax3)
ax3.set_title(t3, fontsize=11, fontweight='bold')
ax3.set_xlabel('', fontsize=8, fontweight='semibold')
ax3.set_ylabel('Normalized 50-yr MA Annual Flow', fontsize=9, fontweight='semibold')
ax3.set_xlim('2000','2099')
ax3.set_xticks(['2000','2025','2050','2075','2100'])
ax3.set_xticklabels([2000,2025,2050,2075,2100], size=10, fontweight='semibold')
ax3.set_yticks([0.8,1.0,1.2,1.4])
ax3.set_yticklabels([0.8,1.0,1.2,1.4], size=10, fontweight='semibold')

cmip_annQ_spag_FOL_I.plot(legend=None, color='lightgray', ax=ax4)
CMIP5_filtered_end_FOL_I.plot(legend=None, color='cornflowerblue', ax=ax4)
historical_FOL_I.plot(legend=None, color='k', ax=ax4)
half_half_end_FOL_I.iloc[0,:].plot(legend=None, color='r', ax=ax4)
ax4.set_title(t4, fontsize=11, fontweight='bold')
ax4.set_xlabel('', fontsize=8, fontweight='semibold')
ax4.set_ylabel('', fontsize=8, fontweight='semibold')
ax4.set_xlim('2000','2099')
ax4.set_xticks(['2000','2025','2050','2075','2100'])
ax4.set_xticklabels([2000,2025,2050,2075,2100], size=10, fontweight='semibold')
ax4.set_yticks([0.4,0.6,0.8,1.0,1.2,1.4,1.6])
ax4.set_yticklabels([0.4,0.6,0.8,1.0,1.2,1.4,1.6], size=10, fontweight='semibold')


ax1.legend(['Vulnerable','Not Vulnerable','Threshold','Historical Flow'],frameon=True,fontsize=8,edgecolor='k')

f.subplots_adjust(hspace=.3)

# plt.savefig('Figure_5.svg', format='svg')
plt.show()

###############################

print('Finished')
