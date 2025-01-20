import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Read the OpenET data
df = pd.read_csv('/home/george/Documents/Work/portales_analysis/portales_openET_2020.csv')
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['P_minus_ET'] = df['Precip (gridMET)'] - df['Ensemble ET']

# Read the VWC data, skipping the header rows
vwc = pd.read_csv('/home/george/Scripts/gnssIR/data/refl_code/Files/p038/p038_vwc.txt',
                 skiprows=3, delim_whitespace=True,
                 names=['FracYr', 'Year', 'DOY', 'VWC', 'Month', 'Day'])

# Convert DOY to datetime
vwc['DateTime'] = pd.to_datetime(vwc['Year'].astype(str) + ' ' + vwc['DOY'].astype(str),
                               format='%Y %j')

# Calculate rate of change
vwc['dS_dt'] = (vwc['VWC'] - vwc['VWC'].shift(1))  # simple difference for now

# Create figure with four subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# Panel 1: Precipitation and ET
ax1.plot(df['DateTime'], df['Precip (gridMET)'], label='Precipitation', color='blue')
ax1.plot(df['DateTime'], df['Ensemble ET'], label='ET', color='red')
ax1.set_ylabel('mm/day')
ax1.legend()
ax1.grid(True)
ax1.set_title('Precipitation and ET')

# Panel 2: P-ET
ax2.plot(df['DateTime'], df['P_minus_ET'], label='P-ET', color='green')
ax2.set_ylabel('mm/day')
ax2.legend()
ax2.grid(True)
ax2.set_title('P-ET (Net Water Input)')

# Panel 3: VWC
ax3.plot(vwc['DateTime'], vwc['VWC'], label='VWC', color='purple')
ax3.set_ylabel('VWC (m³/m³)')
ax3.legend()
ax3.grid(True)
ax3.set_title('Soil Moisture (VWC)')

# Panel 4: Compare P-ET and dS/dt
ax4.plot(df['DateTime'], df['P_minus_ET'], label='P-ET', color='green', alpha=0.7)
ax4_twin = ax4.twinx()
ax4_twin.plot(vwc['DateTime'], vwc['dS_dt'], label='dS/dt', color='red')
ax4.set_ylabel('P-ET (mm/day)')
ax4_twin.set_ylabel('dS/dt (VWC change per day)')

# Combine legends for last panel
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax4.grid(True)
ax4.set_title('P-ET vs Soil Moisture Change')
plt.xlabel('Date')
plt.tight_layout()
plt.show()

# Print basic statistics
print("\nBasic statistics for P-ET (mm/day):")
print(df['P_minus_ET'].describe())
print("\nBasic statistics for dS/dt (VWC change per day):")
print(vwc['dS_dt'].describe())