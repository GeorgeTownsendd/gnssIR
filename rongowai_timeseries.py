import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import re

# File path
file_path = '/home/george/Documents/Work/rongowai_timeseries/taupo_timeseries.gpkg'
layer_name = 'spatial'

# Read the geopackage file
print(f"Reading geopackage file: {file_path}")
data = gpd.read_file(file_path, layer=layer_name)
print(f"Loaded {len(data)} records")


# Extract date from filename
def extract_date_from_filename(filename):
    match = re.match(r'(\d{8})-\d{6}_.*', filename)
    if match:
        date_str = match.group(1)
        return datetime.strptime(date_str, '%Y%m%d').date()
    return None


# Add date column
data['date'] = data['filename'].apply(extract_date_from_filename)
data = data.dropna(subset=['date'])  # Remove records with missing dates

# Create land cover type mapping for reference
landcover_mapping = {
    -1: 'Ocean',
    1: 'Artificial',
    2: 'Barely vegetated',
    3: 'Inland water',
    4: 'Crop',
    5: 'Grass',
    6: 'Shrub',
    7: 'Forest'
}

# Filter data by polarization type
lhcp_data = data[data['ddm_ant'] == 2].copy()
rhcp_data = data[data['ddm_ant'] == 3].copy()

# Also filter for grass land only (sp_surface_type = 5)
lhcp_grass = lhcp_data[lhcp_data['sp_surface_type'] == 5].copy()
rhcp_grass = rhcp_data[rhcp_data['sp_surface_type'] == 5].copy()

print(f"LHCP measurements (all land types): {len(lhcp_data)}")
print(f"RHCP measurements (all land types): {len(rhcp_data)}")
print(f"LHCP measurements (grass only): {len(lhcp_grass)}")
print(f"RHCP measurements (grass only): {len(rhcp_grass)}")

# Calculate daily statistics for all land types
lhcp_daily = lhcp_data.groupby('date')['surface_reflectivity_peak'].agg(['mean', 'std', 'count']).reset_index()
rhcp_daily = rhcp_data.groupby('date')['surface_reflectivity_peak'].agg(['mean', 'std', 'count']).reset_index()

# Calculate daily statistics for grass land only
lhcp_grass_daily = lhcp_grass.groupby('date')['surface_reflectivity_peak'].agg(['mean', 'std', 'count']).reset_index()
rhcp_grass_daily = rhcp_grass.groupby('date')['surface_reflectivity_peak'].agg(['mean', 'std', 'count']).reset_index()

# Prepare data for CSV export
csv_data = lhcp_daily[['date', 'mean', 'count']].rename(
    columns={'mean': 'lhcp_mean_all', 'count': 'daily_observations'})
csv_data['rhcp_mean_all'] = rhcp_daily['mean']
csv_data['lhcp_mean_grass'] = pd.Series(dtype='float64')
csv_data['rhcp_mean_grass'] = pd.Series(dtype='float64')
csv_data['grass_observations'] = pd.Series(dtype='int')

# Add grass data to the CSV dataframe
for date in lhcp_grass_daily['date']:
    grass_row = lhcp_grass_daily[lhcp_grass_daily['date'] == date].iloc[0]
    csv_data.loc[csv_data['date'] == date, 'lhcp_mean_grass'] = grass_row['mean']
    csv_data.loc[csv_data['date'] == date, 'grass_observations'] = grass_row['count']

for date in rhcp_grass_daily['date']:
    grass_row = rhcp_grass_daily[rhcp_grass_daily['date'] == date].iloc[0]
    csv_data.loc[csv_data['date'] == date, 'rhcp_mean_grass'] = grass_row['mean']

# Save to CSV
csv_filename = 'taupo_reflectivity_analysis.csv'
csv_data.to_csv(csv_filename, index=False)
print(f"Data exported to {csv_filename}")

# Create the plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

# Top panel: Daily average reflectivity - all land types
ax1.scatter(lhcp_daily['date'], lhcp_daily['mean'], color='blue', label='LHCP (All Types)', s=40, alpha=0.6, marker='o')
ax1.scatter(rhcp_daily['date'], rhcp_daily['mean'], color='red', label='RHCP (All Types)', s=40, alpha=0.6, marker='o')

# Top panel: Daily average reflectivity - grass land only (different colors/symbols)
ax1.scatter(lhcp_grass_daily['date'], lhcp_grass_daily['mean'], color='green', label='LHCP (Grass Only)', s=60,
            alpha=0.8, marker='*')
ax1.scatter(rhcp_grass_daily['date'], rhcp_grass_daily['mean'], color='purple', label='RHCP (Grass Only)', s=60,
            alpha=0.8, marker='*')

# Bottom panel: Observation counts
ax2.bar(lhcp_daily['date'], lhcp_daily['count'], color='gray', alpha=0.7, label='All Land Types')
ax2.bar(lhcp_grass_daily['date'], lhcp_grass_daily['count'], color='green', alpha=0.7, label='Grass Only')

# Format date axis
date_format = mdates.DateFormatter('%Y-%m-%d')
ax2.xaxis.set_major_formatter(date_format)
ax2.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Set consistent x-axis limits
min_date = min(data['date']) - pd.Timedelta(days=5)
max_date = max(data['date']) + pd.Timedelta(days=5)
ax1.set_xlim([min_date, max_date])

# Add labels and styling
ax1.set_ylabel('Average Surface Reflectivity')
ax1.set_title('Daily Average Surface Reflectivity by Polarization and Land Type (Taupo)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Date')
ax2.set_ylabel('Observation Count')
ax2.set_title('Number of Observations per Day')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_filename = 'taupo_reflectivity_grass_comparison.png'
plt.savefig(output_filename, dpi=300)
plt.show()
