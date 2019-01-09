from __future__ import division
import numpy as np 
import copy
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas import *
import seaborn as sns
from scipy.stats import skew, norm
import datetime
from matplotlib.collections import LineCollection
plt.show(block=False)

def init_plotting():
  sns.set_style('whitegrid')
  sns.set_style('whitegrid')
  plt.rcParams['figure.figsize'] = (9, 7)
  plt.rcParams['font.size'] = 16
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

############################################################################################################
init_plotting()


FNP_70_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Percent_False_Negatives_with_10-lowest_and_70.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
FPP_70_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Percent_False_Positives_with_10-lowest_and_70.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')

FNP_80_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Percent_False_Negatives_with_10-lowest_and_80.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
FPP_80_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Percent_False_Positives_with_10-lowest_and_80.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')

EM_70_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Error_Matrix_with_10-lowest_and_70.0%_agreement_with-interpolating_all.csv',
                              index_col='datetime')
EM_80_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Error_Matrix_with_10-lowest_and_80.0%_agreement_with-interpolating_all.csv',
                              index_col='datetime')

EM_60_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Error_Matrix_with_10-lowest_and_60.0%_agreement_with-interpolating_all.csv',
                              index_col='datetime')
EM_50_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Error_Matrix_with_10-lowest_and_50.0%_agreement_with-interpolating_all.csv',
                              index_col='datetime')
EM_40_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Error_Matrix_with_10-lowest_and_40.0%_agreement_with-interpolating_all.csv',
                              index_col='datetime')
y20EM_60_inter_10 = pd.read_csv('Data/More_Rivers2_20-yr_Error_Matrix_with_10-lowest_and_60.0%_agreement_with-interpolating_all.csv',
                              index_col='datetime')
y30EM_60_inter_10 = pd.read_csv('Data/More_Rivers2_30-yr_Error_Matrix_with_10-lowest_and_60.0%_agreement_with-interpolating_all.csv',
                              index_col='datetime')
y40EM_60_inter_10 = pd.read_csv('Data/More_Rivers2_40-yr_Error_Matrix_with_10-lowest_and_60.0%_agreement_with-interpolating_all.csv',
                              index_col='datetime')
y60EM_60_inter_10 = pd.read_csv('Data/More_Rivers2_60-yr_Error_Matrix_with_10-lowest_and_60.0%_agreement_with-interpolating_all.csv',
                              index_col='datetime')

FN_80_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_False_Negatives_with_10-lowest_and_80.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
FP_80_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_False_Positives_with_10-lowest_and_80.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TN_80_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_True_Negatives2_with_10-lowest_and_80.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TP_80_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_True_Positives2_with_10-lowest_and_80.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')

FN_70_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_False_Negatives_with_10-lowest_and_70.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
FP_70_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_False_Positives_with_10-lowest_and_70.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TN_70_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_True_Negatives2_with_10-lowest_and_70.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TP_70_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_True_Positives2_with_10-lowest_and_70.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')

FN_60_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_False_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
FP_60_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_False_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TN_60_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_True_Negatives2_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TP_60_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_True_Positives2_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
FNP_60_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Percent_False_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
FPP_60_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Percent_False_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')

FN_50_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_False_Negatives_with_10-lowest_and_50.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
FP_50_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_False_Positives_with_10-lowest_and_50.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TN_50_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_True_Negatives2_with_10-lowest_and_50.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TP_50_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_True_Positives2_with_10-lowest_and_50.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
FNP_50_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Percent_False_Negatives_with_10-lowest_and_50.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
FPP_50_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Percent_False_Positives_with_10-lowest_and_50.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')

FN_40_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_False_Negatives_with_10-lowest_and_40.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
FP_40_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_False_Positives_with_10-lowest_and_40.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TN_40_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_True_Negatives2_with_10-lowest_and_40.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TP_40_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_True_Positives2_with_10-lowest_and_40.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
FNP_40_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Percent_False_Negatives_with_10-lowest_and_40.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
FPP_40_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Percent_False_Positives_with_10-lowest_and_40.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')

y20FN_60_inter_10 = pd.read_csv('Data/More_Rivers2_20-yr_False_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y20FP_60_inter_10 = pd.read_csv('Data/More_Rivers2_20-yr_False_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y20TN_60_inter_10 = pd.read_csv('Data/More_Rivers2_20-yr_True_Negatives2_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y20TP_60_inter_10 = pd.read_csv('Data/More_Rivers2_20-yr_True_Positives2_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y20FNP_60_inter_10 = pd.read_csv('Data/More_Rivers2_20-yr_Percent_False_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y20FPP_60_inter_10 = pd.read_csv('Data/More_Rivers2_20-yr_Percent_False_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')

y30FN_60_inter_10 = pd.read_csv('Data/More_Rivers2_30-yr_False_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y30FP_60_inter_10 = pd.read_csv('Data/More_Rivers2_30-yr_False_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y30TN_60_inter_10 = pd.read_csv('Data/More_Rivers2_30-yr_True_Negatives2_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y30TP_60_inter_10 = pd.read_csv('Data/More_Rivers2_30-yr_True_Positives2_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y30FNP_60_inter_10 = pd.read_csv('Data/More_Rivers2_30-yr_Percent_False_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y30FPP_60_inter_10 = pd.read_csv('Data/More_Rivers2_30-yr_Percent_False_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')

y40FN_60_inter_10 = pd.read_csv('Data/More_Rivers2_40-yr_False_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y40FP_60_inter_10 = pd.read_csv('Data/More_Rivers2_40-yr_False_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y40TN_60_inter_10 = pd.read_csv('Data/More_Rivers2_40-yr_True_Negatives2_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y40TP_60_inter_10 = pd.read_csv('Data/More_Rivers2_40-yr_True_Positives2_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y40FNP_60_inter_10 = pd.read_csv('Data/More_Rivers2_40-yr_Percent_False_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y40FPP_60_inter_10 = pd.read_csv('Data/More_Rivers2_40-yr_Percent_False_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')

y60FN_60_inter_10 = pd.read_csv('Data/More_Rivers2_60-yr_False_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y60FP_60_inter_10 = pd.read_csv('Data/More_Rivers2_60-yr_False_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y60TN_60_inter_10 = pd.read_csv('Data/More_Rivers2_60-yr_True_Negatives2_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y60TP_60_inter_10 = pd.read_csv('Data/More_Rivers2_60-yr_True_Positives2_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y60FNP_60_inter_10 = pd.read_csv('Data/More_Rivers2_60-yr_Percent_False_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y60FPP_60_inter_10 = pd.read_csv('Data/More_Rivers2_60-yr_Percent_False_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')

TotN_80_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Total_Negatives_with_10-lowest_and_80.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TotP_80_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Total_Positives_with_10-lowest_and_80.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TotN_70_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Total_Negatives_with_10-lowest_and_70.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TotP_70_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Total_Positives_with_10-lowest_and_70.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TotN_60_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Total_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TotP_60_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Total_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TotN_50_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Total_Negatives_with_10-lowest_and_50.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TotP_50_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Total_Positives_with_10-lowest_and_50.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TotN_40_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Total_Negatives_with_10-lowest_and_40.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
TotP_40_inter_10 = pd.read_csv('Data/More_Rivers2_50-yr_Total_Positives_with_10-lowest_and_40.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y20TotN_60_inter_10 = pd.read_csv('Data/More_Rivers2_20-yr_Total_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y20TotP_60_inter_10 = pd.read_csv('Data/More_Rivers2_20-yr_Total_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y30TotN_60_inter_10 = pd.read_csv('Data/More_Rivers2_30-yr_Total_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y30TotP_60_inter_10 = pd.read_csv('Data/More_Rivers2_30-yr_Total_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y40TotN_60_inter_10 = pd.read_csv('Data/More_Rivers2_40-yr_Total_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y40TotP_60_inter_10 = pd.read_csv('Data/More_Rivers2_40-yr_Total_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y60TotN_60_inter_10 = pd.read_csv('Data/More_Rivers2_60-yr_Total_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')
y60TotP_60_inter_10 = pd.read_csv('Data/More_Rivers2_60-yr_Total_Positives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv',
                              index_col='datetime')

############################################################################################################

sites = EM_70_inter_10.columns.values
datetime = EM_70_inter_10.index

for site in sites:
  for date in datetime:
    if EM_70_inter_10.loc[date,site] > 85:
      FN_70_inter_10.loc[date,site] = np.nan
      FP_70_inter_10.loc[date,site] = np.nan
      TN_70_inter_10.loc[date,site] = np.nan
      TP_70_inter_10.loc[date,site] = np.nan
    if EM_80_inter_10.loc[date,site] > 85:
      FN_80_inter_10.loc[date,site] = np.nan
      FP_80_inter_10.loc[date,site] = np.nan
      TN_80_inter_10.loc[date,site] = np.nan
      TP_80_inter_10.loc[date,site] = np.nan    
    if EM_60_inter_10.loc[date,site] > 85:
      FN_60_inter_10.loc[date,site] = np.nan
      FP_60_inter_10.loc[date,site] = np.nan
      TN_60_inter_10.loc[date,site] = np.nan
      TP_60_inter_10.loc[date,site] = np.nan
    if EM_50_inter_10.loc[date,site] > 85:
      FN_50_inter_10.loc[date,site] = np.nan
      FP_50_inter_10.loc[date,site] = np.nan
      TN_50_inter_10.loc[date,site] = np.nan
      TP_50_inter_10.loc[date,site] = np.nan
    if EM_40_inter_10.loc[date,site] > 85:
      FN_40_inter_10.loc[date,site] = np.nan
      FP_40_inter_10.loc[date,site] = np.nan
      TN_40_inter_10.loc[date,site] = np.nan
      TP_40_inter_10.loc[date,site] = np.nan


datetime = y20EM_60_inter_10.index
for site in sites:
  for date in datetime:
    if y20EM_60_inter_10.loc[date,site] > 85:
      y20FN_60_inter_10.loc[date,site] = np.nan
      y20FP_60_inter_10.loc[date,site] = np.nan
      y20TN_60_inter_10.loc[date,site] = np.nan
      y20TP_60_inter_10.loc[date,site] = np.nan
 

datetime = y30EM_60_inter_10.index
for site in sites:
  for date in datetime:
    if y30EM_60_inter_10.loc[date,site] > 85:
      y30FN_60_inter_10.loc[date,site] = np.nan
      y30FP_60_inter_10.loc[date,site] = np.nan
      y30TN_60_inter_10.loc[date,site] = np.nan
      y30TP_60_inter_10.loc[date,site] = np.nan


datetime = y40EM_60_inter_10.index
for site in sites:
  for date in datetime:
    if y40EM_60_inter_10.loc[date,site] > 85:
      y40FN_60_inter_10.loc[date,site] = np.nan
      y40FP_60_inter_10.loc[date,site] = np.nan
      y40TN_60_inter_10.loc[date,site] = np.nan
      y40TP_60_inter_10.loc[date,site] = np.nan
  

datetime = y60EM_60_inter_10.index
for site in sites:
  for date in datetime:
    if y60EM_60_inter_10.loc[date,site] > 85:
      y60FN_60_inter_10.loc[date,site] = np.nan
      y60FP_60_inter_10.loc[date,site] = np.nan
      y60TN_60_inter_10.loc[date,site] = np.nan
      y60TP_60_inter_10.loc[date,site] = np.nan


FPP2_80_inter_10 = FP_80_inter_10 / TotN_80_inter_10 * 100
FNP2_80_inter_10 = FN_80_inter_10 / TotP_80_inter_10 * 100
TPP2_80_inter_10 = TP_80_inter_10 / TotP_80_inter_10 * 100
TNP2_80_inter_10 = TN_80_inter_10 / TotN_80_inter_10 * 100

FPP2_70_inter_10 = FP_70_inter_10 / TotN_70_inter_10 * 100
FNP2_70_inter_10 = FN_70_inter_10 / TotP_70_inter_10 * 100
TPP2_70_inter_10 = TP_70_inter_10 / TotP_70_inter_10 * 100
TNP2_70_inter_10 = TN_70_inter_10 / TotN_70_inter_10 * 100

FPP2_60_inter_10 = FP_60_inter_10 / TotN_60_inter_10 * 100
FNP2_60_inter_10 = FN_60_inter_10 / TotP_60_inter_10 * 100
TPP2_60_inter_10 = TP_60_inter_10 / TotP_60_inter_10 * 100
TNP2_60_inter_10 = TN_60_inter_10 / TotN_60_inter_10 * 100

FPP2_50_inter_10 = FP_50_inter_10 / TotN_50_inter_10 * 100
FNP2_50_inter_10 = FN_50_inter_10 / TotP_50_inter_10 * 100
TPP2_50_inter_10 = TP_50_inter_10 / TotP_50_inter_10 * 100
TNP2_50_inter_10 = TN_50_inter_10 / TotN_50_inter_10 * 100

FPP2_40_inter_10 = FP_40_inter_10 / TotN_40_inter_10 * 100
FNP2_40_inter_10 = FN_40_inter_10 / TotP_40_inter_10 * 100
TPP2_40_inter_10 = TP_40_inter_10 / TotP_40_inter_10 * 100
TNP2_40_inter_10 = TN_40_inter_10 / TotN_40_inter_10 * 100

y20FPP2_60_inter_10 = y20FP_60_inter_10 / y20TotN_60_inter_10 * 100
y20FNP2_60_inter_10 = y20FN_60_inter_10 / y20TotP_60_inter_10 * 100
y20TPP2_60_inter_10 = y20TP_60_inter_10 / y20TotP_60_inter_10 * 100
y20TNP2_60_inter_10 = y20TN_60_inter_10 / y20TotN_60_inter_10 * 100

y30FPP2_60_inter_10 = y30FP_60_inter_10 / y30TotN_60_inter_10 * 100
y30FNP2_60_inter_10 = y30FN_60_inter_10 / y30TotP_60_inter_10 * 100
y30TPP2_60_inter_10 = y30TP_60_inter_10 / y30TotP_60_inter_10 * 100
y30TNP2_60_inter_10 = y30TN_60_inter_10 / y30TotN_60_inter_10 * 100

y40FPP2_60_inter_10 = y40FP_60_inter_10 / y40TotN_60_inter_10 * 100
y40FNP2_60_inter_10 = y40FN_60_inter_10 / y40TotP_60_inter_10 * 100
y40TPP2_60_inter_10 = y40TP_60_inter_10 / y40TotP_60_inter_10 * 100
y40TNP2_60_inter_10 = y40TN_60_inter_10 / y40TotN_60_inter_10 * 100

y60FPP2_60_inter_10 = y60FP_60_inter_10 / y60TotN_60_inter_10 * 100
y60FNP2_60_inter_10 = y60FN_60_inter_10 / y60TotP_60_inter_10 * 100
y60TPP2_60_inter_10 = y60TP_60_inter_10 / y60TotP_60_inter_10 * 100
y60TNP2_60_inter_10 = y60TN_60_inter_10 / y60TotN_60_inter_10 * 100
###########################################################################################################

FNP_70_inter_10_avg = FNP_70_inter_10.mean(axis=1)
FPP_70_inter_10_avg = FPP_70_inter_10.mean(axis=1)

FNP_80_inter_10_avg = FNP_80_inter_10.mean(axis=1)
FPP_80_inter_10_avg = FPP_80_inter_10.mean(axis=1)

FNP_60_inter_10_avg = FNP_60_inter_10.mean(axis=1)
FPP_60_inter_10_avg = FPP_60_inter_10.mean(axis=1)

FNP_50_inter_10_avg = FNP_50_inter_10.mean(axis=1)
FPP_50_inter_10_avg = FPP_50_inter_10.mean(axis=1)

FNP_40_inter_10_avg = FNP_40_inter_10.mean(axis=1)
FPP_40_inter_10_avg = FPP_40_inter_10.mean(axis=1)

y20FNP_60_inter_10_avg = y20FNP_60_inter_10.mean(axis=1)
y20FPP_60_inter_10_avg = y20FPP_60_inter_10.mean(axis=1)

y30FNP_60_inter_10_avg = y30FNP_60_inter_10.mean(axis=1)
y30FPP_60_inter_10_avg = y30FPP_60_inter_10.mean(axis=1)

y40FNP_60_inter_10_avg = y40FNP_60_inter_10.mean(axis=1)
y40FPP_60_inter_10_avg = y40FPP_60_inter_10.mean(axis=1)

y60FNP_60_inter_10_avg = y60FNP_60_inter_10.mean(axis=1)
y60FPP_60_inter_10_avg = y60FPP_60_inter_10.mean(axis=1)
############################################################################################################

FPP2_60_inter_10.index = pd.to_datetime(FPP2_60_inter_10.index).year
FNP2_60_inter_10.index = pd.to_datetime(FNP2_60_inter_10.index).year
TPP2_60_inter_10.index = pd.to_datetime(TPP2_60_inter_10.index).year
TNP2_60_inter_10.index = pd.to_datetime(TNP2_60_inter_10.index).year

y20FPP2_60_inter_10.index = pd.to_datetime(y20FPP2_60_inter_10.index).year
y20FNP2_60_inter_10.index = pd.to_datetime(y20FNP2_60_inter_10.index).year
y20TPP2_60_inter_10.index = pd.to_datetime(y20TPP2_60_inter_10.index).year
y20TNP2_60_inter_10.index = pd.to_datetime(y20TNP2_60_inter_10.index).year

y30FPP2_60_inter_10.index = pd.to_datetime(y30FPP2_60_inter_10.index).year
y30FNP2_60_inter_10.index = pd.to_datetime(y30FNP2_60_inter_10.index).year
y30TPP2_60_inter_10.index = pd.to_datetime(y30TPP2_60_inter_10.index).year
y30TNP2_60_inter_10.index = pd.to_datetime(y30TNP2_60_inter_10.index).year

y40FPP2_60_inter_10.index = pd.to_datetime(y40FPP2_60_inter_10.index).year
y40FNP2_60_inter_10.index = pd.to_datetime(y40FNP2_60_inter_10.index).year
y40TPP2_60_inter_10.index = pd.to_datetime(y40TPP2_60_inter_10.index).year
y40TNP2_60_inter_10.index = pd.to_datetime(y40TNP2_60_inter_10.index).year

y60FPP2_60_inter_10.index = pd.to_datetime(y60FPP2_60_inter_10.index).year
y60FNP2_60_inter_10.index = pd.to_datetime(y60FNP2_60_inter_10.index).year
y60TPP2_60_inter_10.index = pd.to_datetime(y60TPP2_60_inter_10.index).year
y60TNP2_60_inter_10.index = pd.to_datetime(y60TNP2_60_inter_10.index).year

### Plotting Agreement Variation #############################################################

ax1 = plt.axes([.1,.1,.4,.5])
ax2 = plt.axes([.56,.1,.4,.5])

ax1.plot(FNP2_60_inter_10.index, FNP2_40_inter_10.mean(axis=1), color='#ff9999')
ax1.plot(FNP2_60_inter_10.index, FNP2_50_inter_10.mean(axis=1), color='#ff4d4d')
ax1.plot(FNP2_60_inter_10.index, FNP2_60_inter_10.mean(axis=1), color='#ff0000')
ax1.plot(FNP2_60_inter_10.index, FNP2_70_inter_10.mean(axis=1), color='#b30000')
ax1.plot(FNP2_60_inter_10.index, FNP2_80_inter_10.mean(axis=1), color='#660000')

ax1.plot(TPP2_60_inter_10.index, TPP2_40_inter_10.mean(axis=1), color='#adebad')
ax1.plot(TPP2_60_inter_10.index, TPP2_50_inter_10.mean(axis=1), color='#70db70')
ax1.plot(TPP2_60_inter_10.index, TPP2_60_inter_10.mean(axis=1), color='#33cc33')
ax1.plot(TPP2_60_inter_10.index, TPP2_70_inter_10.mean(axis=1), color='#248f24')
ax1.plot(TPP2_60_inter_10.index, TPP2_80_inter_10.mean(axis=1), color='#145214')

ax1.set_ylabel('Percent (%)', fontsize=14, fontweight='semibold')
ax1.set_yticklabels([0,20,40,60,80,100], size=13, fontweight='semibold')
ax1.set_ylim(0,100)
ax1.set_xlim(2000, 2100)
ax1.set_xticks([2000,2025,2050,2075,2100])
ax1.set_xticklabels([2000,2025,2050,2075,2100], size=13, fontweight='semibold')

a, = ax2.plot(FPP2_60_inter_10.index, FPP2_40_inter_10.mean(axis=1), color='#ff9999')
b, = ax2.plot(FPP2_60_inter_10.index, FPP2_50_inter_10.mean(axis=1), color='#ff4d4d')
c, = ax2.plot(FPP2_60_inter_10.index, FPP2_60_inter_10.mean(axis=1), color='#ff0000')
d, = ax2.plot(FPP2_60_inter_10.index, FPP2_70_inter_10.mean(axis=1), color='#b30000')
e, = ax2.plot(FPP2_60_inter_10.index, FPP2_80_inter_10.mean(axis=1), color='#660000')

f, = ax2.plot(TNP2_60_inter_10.index, TNP2_40_inter_10.mean(axis=1), color='#adebad')
g, = ax2.plot(TNP2_60_inter_10.index, TNP2_50_inter_10.mean(axis=1), color='#70db70')
h, = ax2.plot(TNP2_60_inter_10.index, TNP2_60_inter_10.mean(axis=1), color='#33cc33')
i, = ax2.plot(TNP2_60_inter_10.index, TNP2_70_inter_10.mean(axis=1), color='#248f24')
j, = ax2.plot(TNP2_60_inter_10.index, TNP2_80_inter_10.mean(axis=1), color='#145214')

ax2.set_xlabel('', fontsize=1)

ax2.set_yticklabels([])
ax2.set_ylim(0,100)
ax2.set_xlim(2000, 2100)
ax2.set_xticks([2000,2025,2050,2075,2100])
ax2.set_xticklabels([2000,2025,2050,2075,2100], size=13, fontweight='semibold')

ax1.text(2001,92, '(A)', fontweight='bold', fontsize=15)
ax2.text(2001,85, '(B)', fontweight='bold', fontsize=15)


ax1.text(2067,89, 'True Positives', fontweight='bold', fontsize=12, va='center', ha='right', color='#33cc33')
ax1.text(2068,7, 'False Negatives', fontweight='bold', fontsize=12, va='center', ha='right', color='#ff0000')
ax2.text(2052,82, 'True Negatives', fontweight='bold', fontsize=12, va='center', ha='right', color='#33cc33')
ax2.text(2050,17, 'False Positives', fontweight='bold', fontsize=12, va='center', ha='right', color='#ff0000')


true_legend = plt.legend([f,g,h,i,j],['40% Agreement','50% Agreement', '60% Agreement', '70% Agreement', '80% Agreement'], title='Correct', frameon=True, fontsize=10, loc=6)
false_legend = plt.legend([a,b,c,d,e],['40% Agreement','50% Agreement', '60% Agreement', '70% Agreement', '80% Agreement'], title='Incorrect', frameon=True, fontsize=10, loc=7)

true_legend.get_frame().set_edgecolor('k')
false_legend.get_frame().set_edgecolor('k')

ax2.add_artist(true_legend)
ax2.add_artist(false_legend)

plt.setp(true_legend.get_title(),fontsize=12,fontweight='semibold')
plt.setp(false_legend.get_title(),fontsize=12,fontweight='semibold')

plt.savefig('Figure_6.svg', format='svg', bbox_inches='tight')
plt.show()


### Plotting Window Size Variation ################################################################

ax1 = plt.axes([.1,.1,.4,.5])
ax2 = plt.axes([.56,.1,.4,.5])

ax1.plot(y20FNP2_60_inter_10.index, y20FNP2_60_inter_10.mean(axis=1), color='#ff9999')
ax1.plot(y30FNP2_60_inter_10.index, y30FNP2_60_inter_10.mean(axis=1), color='#ff4d4d')
ax1.plot(y40FNP2_60_inter_10.index, y40FNP2_60_inter_10.mean(axis=1), color='#ff0000')
ax1.plot(FNP2_60_inter_10.index, FNP2_60_inter_10.mean(axis=1), color='#b30000')
ax1.plot(y60FNP2_60_inter_10.index, y60FNP2_60_inter_10.mean(axis=1), color='#660000')

ax1.plot(y20TPP2_60_inter_10.index, y20TPP2_60_inter_10.mean(axis=1), color='#adebad')
ax1.plot(y30TPP2_60_inter_10.index, y30TPP2_60_inter_10.mean(axis=1), color='#70db70')
ax1.plot(y40TPP2_60_inter_10.index, y40TPP2_60_inter_10.mean(axis=1), color='#33cc33')
ax1.plot(TPP2_60_inter_10.index, TPP2_60_inter_10.mean(axis=1), color='#248f24')
ax1.plot(y60TPP2_60_inter_10.index, y60TPP2_60_inter_10.mean(axis=1), color='#145214')

ax1.set_ylabel('Percent (%)', fontsize=14, fontweight='semibold')
ax1.set_yticklabels([0,20,40,60,80,100], size=13, fontweight='semibold')
ax1.set_ylim(0,100)
ax1.set_xlim(2000, 2100)
ax1.set_xticks([2000,2025,2050,2075,2100])
ax1.set_xticklabels([2000,2025,2050,2075,2100], size=13, fontweight='semibold')

a, = ax2.plot(y20FPP2_60_inter_10.index, y20FPP2_60_inter_10.mean(axis=1), color='#ff9999')
b, = ax2.plot(y30FPP2_60_inter_10.index, y30FPP2_60_inter_10.mean(axis=1), color='#ff4d4d')
c, = ax2.plot(y40FPP2_60_inter_10.index, y40FPP2_60_inter_10.mean(axis=1), color='#ff0000')
d, = ax2.plot(FPP2_60_inter_10.index, FPP2_60_inter_10.mean(axis=1), color='#b30000')
e, = ax2.plot(y60FPP2_60_inter_10.index, y60FPP2_60_inter_10.mean(axis=1), color='#660000')

f, = ax2.plot(y20TNP2_60_inter_10.index, y20TNP2_60_inter_10.mean(axis=1), color='#adebad')
g, = ax2.plot(y30TNP2_60_inter_10.index, y30TNP2_60_inter_10.mean(axis=1), color='#70db70')
h, = ax2.plot(y40TNP2_60_inter_10.index, y40TNP2_60_inter_10.mean(axis=1), color='#33cc33')
i, = ax2.plot(TNP2_60_inter_10.index, TNP2_60_inter_10.mean(axis=1), color='#248f24')
j, = ax2.plot(y60TNP2_60_inter_10.index, y60TNP2_60_inter_10.mean(axis=1), color='#145214')

ax2.set_xlabel('', fontsize=1)

ax2.set_yticklabels([])
ax2.set_ylim(0,100)
ax2.set_xlim(2000, 2100)
ax2.set_xticks([2000,2025,2050,2075,2100])
ax2.set_xticklabels([2000,2025,2050,2075,2100], size=13, fontweight='semibold')

ax1.text(2001,95, '(A)', fontweight='bold', fontsize=15)
ax2.text(2001,85, '(B)', fontweight='bold', fontsize=15)

ax1.text(2074,88, 'True Positives', fontweight='bold', fontsize=12, va='center', ha='right', color='#33cc33')
ax1.text(2076,7, 'False Negatives', fontweight='bold', fontsize=12, va='center', ha='right', color='#ff0000')
ax2.text(2052,82, 'True Negatives', fontweight='bold', fontsize=12, va='center', ha='right', color='#33cc33')
ax2.text(2050,17, 'False Positives', fontweight='bold', fontsize=12, va='center', ha='right', color='#ff0000')

true_legend = plt.legend([f,g,h,i,j],['20-yr','30-yr', '40-yr', '50-yr', '60-yr'], title='Correct', frameon=True, fontsize=10, loc=6)
false_legend = plt.legend([a,b,c,d,e],['20-yr','30-yr', '40-yr', '50-yr', '60-yr'], title='Incorrect', frameon=True, fontsize=10, loc=7)

true_legend.get_frame().set_edgecolor('k')
false_legend.get_frame().set_edgecolor('k')

ax2.add_artist(true_legend)
ax2.add_artist(false_legend)

plt.setp(true_legend.get_title(),fontsize=12,fontweight='semibold')
plt.setp(false_legend.get_title(),fontsize=12,fontweight='semibold')

plt.savefig('Figure_7.svg', format='svg', bbox_inches='tight')
plt.show()
