import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy import interpolate

# Read the GNSS-IR spline data (skip the 8 header lines)
gnss_df = pd.read_csv('data/refl_code/Files/titg/titg_spline_out.txt',
                      skiprows=8,
                      delim_whitespace=True,
                      names=['MJD', 'RH', 'year', 'month', 'day', 'hour', 'minute', 'second', 'sea_level'])

# Create datetime column for GNSS data
gnss_df['datetime'] = pd.to_datetime(gnss_df[['year', 'month', 'day', 'hour', 'minute', 'second']])

# Read the tide gauge data with special handling for leading spaces
tide_df = pd.read_csv('thursday_island_2023.csv',
                      skipinitialspace=True,
                      parse_dates=[0],
                      index_col=False)

# Rename the datetime column for consistency
tide_df = tide_df.rename(columns={tide_df.columns[0]: 'datetime'})

# Remove any rows where sea level is missing or invalid
tide_df = tide_df[tide_df['Sea Level'].notna()]

# Find overlapping period
start_date = max(tide_df['datetime'].min(), gnss_df['datetime'].min())
end_date = min(tide_df['datetime'].max(), gnss_df['datetime'].max())

# Filter data for overlap period
gnss_overlap = gnss_df[(gnss_df['datetime'] >= start_date) &
                       (gnss_df['datetime'] <= end_date)].copy()

tide_overlap = tide_df[(tide_df['datetime'] >= start_date) &
                      (tide_df['datetime'] <= end_date)].copy()

# Create interpolation function for tide gauge data
tide_interp = interpolate.interp1d(tide_overlap['datetime'].astype(np.int64),
                                  tide_overlap['Sea Level'],
                                  kind='linear',
                                  bounds_error=False)

# Interpolate tide gauge data to GNSS timestamps
gnss_times = gnss_overlap['datetime'].astype(np.int64)
tide_at_gnss = tide_interp(gnss_times)

# Remove any NaN values that might have been introduced by interpolation
valid_mask = ~np.isnan(tide_at_gnss)
gnss_valid = gnss_overlap[valid_mask]
tide_valid = tide_at_gnss[valid_mask]

# Calculate the mean offset
offset = np.mean(tide_valid - gnss_valid['sea_level'])
print(f"Mean offset: {offset:.3f} meters")

# Calculate RMSE after offset correction
differences = tide_valid - (gnss_valid['sea_level'] + offset)
rmse = np.sqrt(np.mean(differences**2))

# Calculate percentiles for y-axis limits
p02, p98 = np.percentile(differences, [2, 98])
y_margin = 0.1 * (p98 - p02)  # Add 10% margin
ylim_min, ylim_max = p02 - y_margin, p98 + y_margin

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1], sharex=True)

# Top subplot
# First plot uncorrected GNSS data (semi-transparent)
ax1.plot(gnss_overlap['datetime'], gnss_overlap['sea_level'],
         's-', color='gray', label='GNSS-IR (original)', markersize=4, alpha=0.3)

# Then plot tide gauge and corrected GNSS data
ax1.plot(tide_overlap['datetime'], tide_overlap['Sea Level'],
         'o-', color='blue', label='Tide Gauge', markersize=4, alpha=0.7)
ax1.plot(gnss_overlap['datetime'], gnss_overlap['sea_level'] + offset,
         's-', color='red', label=f'GNSS-IR (offset corrected: +{offset:.3f} m)', markersize=4, alpha=0.7)

ax1.set_ylabel('Sea Level (m)')
ax1.set_title('Thursday Island Sea Level: GNSS-IR vs Tide Gauge')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# Bottom subplot - Differences after offset correction
ax2.plot(gnss_valid['datetime'], differences,
         'k.', label=f'RMSE: {rmse:.3f} m', markersize=2)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax2.fill_between(gnss_valid['datetime'], -rmse, rmse,
                 color='gray', alpha=0.2, label='±RMSE')

ax2.set_ylim(ylim_min, ylim_max)
ax2.set_ylabel('Difference (m)')
ax2.set_xlabel('Date (UTC)')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# Rotate x-axis labels
plt.gcf().autofmt_xdate()

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('sea_level_comparison_with_rmse.png', dpi=300, bbox_inches='tight')
plt.close()

# Print statistics
print(f"RMSE: {rmse:.3f} meters")
print(f"2nd percentile: {p02:.3f} meters")
print(f"98th percentile: {p98:.3f} meters")
print(f"Number of valid comparison points: {len(differences)}")