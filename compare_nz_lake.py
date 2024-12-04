import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Read the lake data CSV (semicolon separated)
lake_df = pd.read_csv('nz_lake_data.csv', sep=';', parse_dates=['datetime'])

# Read the GNSS reflector height data (space separated with comments)
gnss_df = pd.read_csv('/data/refl_code/Files/tgho/tgho_dailyRH.txt',
                      comment='%',
                      delim_whitespace=True,
                      names=['year', 'doy', 'RH', 'numval', 'month', 'day',
                             'RH_sigma', 'RH_amp', 'Hortho_RH'])

# Convert year and day of year to datetime for GNSS data
gnss_df['datetime'] = pd.to_datetime(gnss_df['year'].astype(str) + ' ' +
                                     gnss_df['doy'].astype(str),
                                     format='%Y %j')

# Find overlapping period
start_date = max(lake_df['datetime'].min(), gnss_df['datetime'].min())
end_date = min(lake_df['datetime'].max(), gnss_df['datetime'].max())

# Filter GNSS data for overlap period
gnss_overlap = gnss_df[(gnss_df['datetime'] >= start_date) &
                       (gnss_df['datetime'] <= end_date)].copy()

# For lake data, include one point before and one point after GNSS data
lake_extended = lake_df[
    (lake_df['datetime'] >= lake_df[lake_df['datetime'] < start_date]['datetime'].iloc[-1]) &
    (lake_df['datetime'] <= lake_df[lake_df['datetime'] > end_date]['datetime'].iloc[0])
].copy()

# Create the figure
plt.figure(figsize=(12, 6))

# Plot both datasets on the same axis
plt.plot(lake_extended['datetime'], lake_extended['water_level'],
         'o-', color='blue', label='Satellite Altimetry Level (DAHITI)', markersize=6)
plt.plot(gnss_overlap['datetime'], gnss_overlap['Hortho_RH'],
         's-', color='red', label='GNSS-IR Level', markersize=6)

# Add error bars if data is not too dense
if len(lake_extended) < 50:  # Only add error bars if we have relatively few points
    plt.errorbar(lake_extended['datetime'], lake_extended['water_level'],
                 yerr=lake_extended['error'],
                 fmt='none', color='blue', alpha=0.3)
    plt.errorbar(gnss_overlap['datetime'], gnss_overlap['Hortho_RH'],
                 yerr=gnss_overlap['RH_sigma'],
                 fmt='none', color='red', alpha=0.3)

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Water Level (m)')
plt.title('Lake Taupo Water Level: GNSS-IR vs Satellite Altimetry')

# Add grid
plt.grid(True, alpha=0.3)

# Rotate x-axis labels
plt.gcf().autofmt_xdate()

# Add legend in top right
plt.legend(loc='upper right')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('refl/Files/tgho/water_level_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Print the number of points
print(f"Number of lake measurements (including extra points): {len(lake_extended)}")
print(f"Number of GNSS-IR measurements: {len(gnss_overlap)}")
print(f"Period: {lake_extended['datetime'].min().date()} to {lake_extended['datetime'].max().date()}")