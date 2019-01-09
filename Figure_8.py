import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mpl_toolkits
import math
from mpl_toolkits.basemap import Basemap
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, (ax1,ax2) = plt.subplots(2,1)

### Create map background ############################################################
m1 = Basemap(llcrnrlon=-126, llcrnrlat=28, urcrnrlon=-88, urcrnrlat=53, projection='cyl', resolution='f', area_thresh=1, ax=ax1) # whole US is 24-53 and -125.6--60
m1.drawmapboundary(fill_color='steelblue', zorder=-99)
m1.arcgisimage(service='World_Physical_Map', xpixels=1000, dpi=1000, verbose=True) # World_Shaded_Relief (original), World_Terrain_Base (not much color), World_Physical_Map (I like this one best so far),
m1.drawstates(color='gray', linewidth=0.4)
m1.drawcoastlines(color='gray', linewidth=0.5)
m1.drawcountries(color='k')
m1.drawrivers(linewidth=0.4, linestyle='solid', color='b')

m2 = Basemap(llcrnrlon=-126, llcrnrlat=28, urcrnrlon=-88, urcrnrlat=53, projection='cyl', resolution='f', area_thresh=1, ax=ax2) # whole US is 24-53 and -125.6--60
m2.drawmapboundary(fill_color='steelblue', zorder=-99)
m2.arcgisimage(service='World_Physical_Map', xpixels=1000, dpi=1000, verbose=True) # World_Shaded_Relief (original), World_Terrain_Base (not much color), World_Physical_Map (I like this one best so far),
m2.drawstates(color='gray', linewidth=0.4)
m2.drawcoastlines(color='gray', linewidth=0.5)
m2.drawcountries(color='k')
m2.drawrivers(linewidth=0.4, linestyle='solid', color='b')

### load reservoir data and scatterplot (lat,lon,elev) #################################
df = pd.read_csv('Sites_Master_List_Final.csv')
lons = df.iloc[:,13].values
lats = df.iloc[:,12].values
dams = df.iloc[:,17]

results1 = pd.read_csv('Data/More_Rivers2_50-yr_Percent_False_Negatives_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv')
results2 = pd.read_csv('Data/More_Rivers2_50-yr_Danger_Cross_Dates_with_10-lowest_and_60.0%_agreement_with-interpolation_all.csv')

results1.iloc[:,0] = pd.to_datetime(results1.iloc[:,0].values).year
results2 = results2.mean(0)
results2 = results2.iloc[1::]

target_FN = 50

results_year1 = pd.DataFrame(index=results1.columns, columns=['Year'])

for site in range(0, results1.shape[1]):
	for n in range(results1.shape[0]-1,-1,-1):
		if results1.iloc[n,site] >= target_FN:
			results_year1.iloc[site,0] = n+2000
			break

for site in range(0, results1.shape[1]):
	if (math.isnan(results_year1.iloc[site,0]) == True):
		for n in range(results1.shape[0]-1,-1,-1):
			if (math.isnan(results1.iloc[n,site]) == True):
				results_year1.iloc[site,0] = n+2000
				break

results_year1 = results_year1.iloc[1::,:].values
results_year1 = results_year1.astype('float64')
results_year1 = results_year1.flatten()


### Plotting Results ##############################################################
x,y = m1(lons,lats)
map1 = m1.scatter(x,y,s=23,c=results_year1, marker='o', edgecolor='k', linewidth=.5, zorder=6, cmap='rainbow', vmin=2000, vmax=2100)
x,y = m2(lons,lats)
map2 = m2.scatter(x,y,s=24,c=results2, marker='o', edgecolor='k', linewidth=.5, zorder=6, cmap='rainbow', vmin=2000, vmax=2100)

divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("bottom", size="5%", pad=0.15)
cbar = fig.colorbar(map1, cax=cax1, orientation='horizontal')
cbar.ax.set_xticklabels(['2000','2020','2040','2060','2080','2100'], weight='semibold')

divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("bottom", size="5%", pad=0.15)
fig.colorbar(map2, cax=cax2, orientation='horizontal')

bbox_props = dict(boxstyle='round', fc='w', alpha=0.9, pad=.3)
ax1.text(-125,29,'(A) Year When False Negative Rate Falls Below 50%', fontsize=8, fontweight='bold', bbox=bbox_props)
ax2.text(-125,29,'(B) Mean Year Vulnerable Scenarios First Cross Threshold', fontsize=8, fontweight='bold', bbox=bbox_props)

fig.subplots_adjust(hspace=.1)
plt.savefig('Figure_8.svg', format='svg', bbox_inches="tight", dpi=500, xpixels=500)
plt.show()
