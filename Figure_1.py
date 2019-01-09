import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import mpl_toolkits
from mpl_toolkits.basemap import Basemap

# create map background  
m = Basemap(llcrnrlon=-126, llcrnrlat=24, urcrnrlon=-66, urcrnrlat=53, projection='cyl', resolution='f', area_thresh=1) # whole US is 24-53 and -125.6--60
m.drawmapboundary(fill_color='steelblue', zorder=-99)
m.arcgisimage(service='World_Physical_Map', xpixels=10000, dpi=10000, verbose= True) # World_Shaded_Relief (original), World_Terrain_Base (not much color), World_Physical_Map (I like this one best so far),
m.drawstates(color='gray')
m.drawcountries(color='k')
m.drawcoastlines(color='gray')
m.drawrivers(linewidth=0.5, linestyle='solid', color='b')


# load reservoir data and scatterplot (lat,lon,dams)
df = pd.read_csv('Sites_Master_List_Final.csv')
lons = df.iloc[:,13].values
lats = df.iloc[:,12].values
dams = df.iloc[:,17]


# Plotting CMIP5 sites
x,y = m(lons,lats)
m.scatter(x,y,s=25,c='k', marker='o', edgecolor='None', zorder=6)

for row in range(0, lons.shape[0]):
	if dams.iloc[row] == 1:
		m.scatter(lons[row], lats[row], s=25, c='r', marker='o', edgecolor='None', zorder=7)


text_color = 'k'


row = 0 # American R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 1 # Chippewa R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 2 # Feather R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 3 # Skagit R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 4 # Sacramento R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 5 # Colorado R
plt.text(x[row]-2, y[row]+.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 6 # Columbia R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 7 # Missouri R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 8 # Missouri R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 9 # Missouri R
plt.text(x[row]-3.5, y[row]-.25, df.iloc[row,4], color=text_color, weight='semibold')

row = 10 # Red R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 11 # Missouri R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 12 # Arkansas R
plt.text(x[row]-3.8, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 13 # Colorado R.
plt.text(x[row]+.15, y[row], df.iloc[row,4], color=text_color, weight='semibold')

row = 14 # Arkansas R
plt.text(x[row]-3.8, y[row]-.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 15 # Colorado R.
plt.text(x[row]+.2, y[row]-.15, df.iloc[row,4], color=text_color, weight='semibold')

row = 16 # Green R.
plt.text(x[row]-2.4, y[row]+.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 17 # Osage R. at Bagnell D.
plt.text(x[row]+.2, y[row]-.25, df.iloc[row,4], color=text_color, weight='semibold')

row = 18 # Sacramento R.
plt.text(x[row]+.15, y[row]-.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 19 # Illinois R.
plt.text(x[row]+.2, y[row]-.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 20 # Des Moines R.
plt.text(x[row]+.15, y[row]-.12, df.iloc[row,4], color=text_color, weight='semibold')

row = 21 # Upper Mississippi R. at Keokuk D.
plt.text(x[row]+.15, y[row]-.3, df.iloc[row,4], color=text_color, weight='semibold')

row = 22 # Iowa R.
plt.text(x[row]+.2, y[row]-.12, df.iloc[row,4], color=text_color, weight='semibold')

row = 23 # Rock R.
plt.text(x[row]+.1, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 24 # Mississippi R.
plt.text(x[row]+.15, y[row]-.15, df.iloc[row,4], color=text_color, weight='semibold')

row = 25 # Snake R. at Hells Canyon D.
plt.text(x[row]-1.5, y[row]-.6, df.iloc[row,4], color=text_color, weight='semibold')

row = 26 # Salmon R.
plt.text(x[row]-3.1, y[row], df.iloc[row,4], color=text_color, weight='semibold')

row = 27 # Snake R. at Palisades D.
plt.text(x[row]-1.1, y[row]+.3, df.iloc[row,4], color=text_color, weight='semibold')

row = 28 # Snake R. at American Falls D.
plt.text(x[row]+.15, y[row]-.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 29 # Snake R. at CJ Strike D.
plt.text(x[row]-1, y[row]-.6, df.iloc[row,4], color=text_color, weight='semibold')

row = 30 # Missouri R.
plt.text(x[row]-.1, y[row]+.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 31 # Yellowstone R.
plt.text(x[row]-2.2, y[row]-.5, df.iloc[row,4], color=text_color, weight='semibold')

row = 32 # Missouri R.
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 33 # Yellowstone R.
plt.text(x[row]-4.6, y[row]-.35, df.iloc[row,4], color=text_color, weight='semibold')

row = 34 # Yellowstone R.
plt.text(x[row]+.15, y[row], df.iloc[row,4], color=text_color, weight='semibold')

row = 35 # Missouri R. at Fort Peck D.
plt.text(x[row]-1.7, y[row]+.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 36 # Missouri R.
plt.text(x[row]-.2, y[row]-.55, df.iloc[row,4], color=text_color, weight='semibold')

row = 37 # Flathead R. at Kerr D.
plt.text(x[row]+.15, y[row]-.5, df.iloc[row,4], color=text_color, weight='semibold')

row = 38 # Pend d'Oreille R. at Albeni Falls D.
plt.text(x[row]+.15, y[row]-.3, df.iloc[row,4], color=text_color, weight='semibold')

row = 39 # Spokane R. at Long Lake D.
plt.text(x[row]-2, y[row]-.5, df.iloc[row,4], color=text_color, weight='semibold')

row = 40 # Columbia R. at Grand Coulee D.
plt.text(x[row]-3.8, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 41 # Columbia R. at John Day D.
plt.text(x[row]+.15, y[row]-.4, df.iloc[row,4], color=text_color, weight='semibold')

row = 42 # Lewis R. at Merwin D.
plt.text(x[row]-2.55, y[row]-.4, df.iloc[row,4], color=text_color, weight='semibold')

row = 43 # Cowlitz R. at Mossyrock D.
plt.text(x[row]-3.1, y[row]-.3, df.iloc[row,4], color=text_color, weight='semibold')

row = 44 # Columbia R. at Priest Rapids D.
plt.text(x[row]-3.6, y[row]+.15, df.iloc[row,4], color=text_color, weight='semibold')

row = 45 # Duncan R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 46 # Columbia R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 47 # Colorado R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 48 # Colorado R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 49 # Grande R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 50 # Grande R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 51 # Green R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 52 # Green R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 53 # Neches R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 54 # North Platte R
plt.text(x[row]+.15, y[row], df.iloc[row,4], color=text_color, weight='semibold')

row = 55 # Cross Bayou R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 56 # Missouri R
plt.text(x[row]+.15, y[row]-.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 57 # Colorado R
plt.text(x[row]+.15, y[row]-.4, df.iloc[row,4], color=text_color, weight='semibold')

row = 58 # San Juan R
plt.text(x[row]-.2, y[row]-.6, df.iloc[row,4], color=text_color, weight='semibold')

row = 59 # Mississippi R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 60 # Mississippi R
plt.text(x[row]-4.3, y[row], df.iloc[row,4], color=text_color, weight='semibold')

row = 61 # Bighorn R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 62 # Boise R
plt.text(x[row]+.15, y[row]-.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 63 # Arkansas R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 64 # Verdigris R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 65 # Tuolumne R
plt.text(x[row]+.15, y[row]+.05, df.iloc[row,4], color=text_color, weight='semibold')

row = 66 # Clearwater R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 67 # Missouri R
plt.text(x[row]-.3, y[row]+.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 68 # Missouri R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 69 # Jefferson R
plt.text(x[row]-3.6, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 70 # Snake R
plt.text(x[row]-1.1, y[row]+.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 71 # Kootenai R
plt.text(x[row]-.1, y[row]+.26, df.iloc[row,4], color=text_color, weight='semibold')

row = 72 # Arkansas R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 73 # Merced R
plt.text(x[row]+.15, y[row]-.3, df.iloc[row,4], color=text_color, weight='semibold')

row = 74 # Madison R
plt.text(x[row]-3.5, y[row]-.25, df.iloc[row,4], color=text_color, weight='semibold')

row = 75 # San Joaquin R
plt.text(x[row]+.15, y[row]-.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 76 # Minnesota R
plt.text(x[row]-4.2, y[row]-.15, df.iloc[row,4], color=text_color, weight='semibold')

row = 77 # Gunnison R
plt.text(x[row]+.15, y[row]-.3, df.iloc[row,4], color=text_color, weight='semibold')

row = 78 # Stanislaus R
plt.text(x[row]-.2, y[row]+.2, df.iloc[row,4], color=text_color, weight='semibold')

row = 79 # Missouri R
plt.text(x[row]+.15, y[row]-.35, df.iloc[row,4], color=text_color, weight='semibold')

row = 80 # Owyhee R
plt.text(x[row]-3.35, y[row]-.3, df.iloc[row,4], color=text_color, weight='semibold')

row = 81 # Payette R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 82 # Deschutes R
plt.text(x[row]-2.6, y[row]-.6, df.iloc[row,4], color=text_color, weight='semibold')

row = 83 # Kansas R
plt.text(x[row]-3.1, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 84 # Sabine R
plt.text(x[row]-3, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 85 # Missouri R
plt.text(x[row]+.15, y[row]-.4, df.iloc[row,4], color=text_color, weight='semibold')

row = 86 # North Platte R
plt.text(x[row]+.15, y[row]-.3, df.iloc[row,4], color=text_color, weight='semibold')

row = 87 # Shoshone R
plt.text(x[row]+.15, y[row]-.4, df.iloc[row,4], color=text_color, weight='semibold')

row = 88 # St. Croix R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 89 # Wind R
plt.text(x[row]+.15, y[row]-.1, df.iloc[row,4], color=text_color, weight='semibold')

row = 90 # Yellowstone R
plt.text(x[row]-.1, y[row]+.2, df.iloc[row,4], color=text_color, weight='semibold')


bbox_props = dict(boxstyle='round', fc='w', alpha=0.9, pad=1)

a = m.scatter(-75.6, 30.6, s=25, c='k', marker='o', edgecolor='None', zorder=7)
b = m.scatter(-75.6, 29.3, s=25, c='r', marker='o', edgecolor='None', zorder=7)
plt.text(-76,30,'     River Site\n\n     River Site with Dam', weight='semibold', va='center',ha='left', bbox=bbox_props)

plt.show() # Once the viewer window opens, maximize it, and then the map should be legible.
