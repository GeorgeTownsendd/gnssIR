import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.polynomial import polynomial as P

# File path
file_path = "/home/george/Scripts/gnssIR/data/refl_code/2025/snr/mchl/mchl0010.25.snr88"

# Define column names
column_names = [
    "sat_num", "elevation", "azimuth", "time_sec", "elevation_rate",
    "S6", "S1", "S2", "S5", "S7", "S8"
]

# Read the data - fixed to address warning
data = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)

# Convert time strings to seconds
def time_to_seconds(time_str):
    hours, minutes, seconds = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60 + seconds

# Convert seconds to time
def seconds_to_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}"  # Removed seconds for cleaner labels

# Set time window for full figure
full_start_time_sec = time_to_seconds("15:45:00")
full_end_time_sec = time_to_seconds("21:28:00")

# Set time window for zoomed figure
zoom_start_time_sec = time_to_seconds("16:45:00")
zoom_end_time_sec = time_to_seconds("20:15:00")

# Set time window for descending arc
descending_start_time_sec = time_to_seconds("20:30:00")
descending_end_time_sec = full_end_time_sec

# Filter data for satellite 6
sat_data = data[data['sat_num'] == 6].sort_values('time_sec')

# Filter data for full time range
full_data = sat_data[(sat_data['time_sec'] >= full_start_time_sec) &
                      (sat_data['time_sec'] <= full_end_time_sec)]

# Filter data for zoomed time range
zoom_data = sat_data[(sat_data['time_sec'] >= zoom_start_time_sec) &
                      (sat_data['time_sec'] <= zoom_end_time_sec)]

# Filter data for descending arc
descending_data = sat_data[(sat_data['time_sec'] >= descending_start_time_sec) &
                           (sat_data['time_sec'] <= descending_end_time_sec)]

# Detrend the descending arc data using a second order polynomial
descending_times = descending_data['time_sec'].values
descending_snr = descending_data['S2'].values

# Normalize times to avoid numerical issues
norm_times = (descending_times - descending_times[0]) / (descending_times[-1] - descending_times[0])

# Fit polynomial (second order)
coeffs = P.polyfit(norm_times, descending_snr, 2)
trend = P.polyval(norm_times, coeffs)

# Create detrended signal
detrended_snr = descending_snr - trend

# Create 1-hour increment ticks for plots
time_positions = []
time_labels = []

# Find the next on-the-hour time after start_time_sec
start_hour = (full_start_time_sec // 3600) + 1  # Round up to the next hour
current_time = start_hour * 3600  # Convert to seconds

# Create ticks for each hour
while current_time <= full_end_time_sec:
    time_positions.append(current_time)
    time_labels.append(seconds_to_time(current_time))
    current_time += 3600  # Add 1 hour (3600 seconds)

# Set figure size and style
plt.rcParams.update({'font.size': 14})

# Common title for plots
common_title = 'GPS PRN 6 - L2C Signal @ MCHL'

# Find global y-axis limits for SNR data
y_min_snr = full_data['S2'].min() - 2
y_max_snr = full_data['S2'].max() + 2

# Find global y-axis limits for elevation data
y_min_elev = full_data['elevation'].min() - 2
y_max_elev = full_data['elevation'].max() + 2

# Create figure 1 - Full data WITH inset
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Plot SNR data
ax1.plot(full_data['time_sec'], full_data['S2'], 'b-', linewidth=2.5, label='SNR')

# Set up ax1 (SNR)
ax1.set_xlim(full_start_time_sec, full_end_time_sec)
ax1.set_ylim(y_min_snr, y_max_snr)
ax1.set_xticks(time_positions)
ax1.set_xticklabels(time_labels, fontsize=16, rotation=0)
ax1.set_xlabel('Time (hh:mm)', fontsize=18, fontweight='bold')
ax1.set_ylabel('SNR (dB-Hz)', fontsize=18, fontweight='bold', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_title(common_title, fontsize=20, fontweight='bold')
ax1.set_facecolor('#f8f8f8')

# Create a second y-axis for elevation
ax1_twin = ax1.twinx()
ax1_twin.plot(full_data['time_sec'], full_data['elevation'], 'r-',
              linewidth=1.5, alpha=0.5, label='Elevation')
ax1_twin.set_ylabel('Elevation (degrees)', fontsize=16, color='red')
ax1_twin.tick_params(axis='y', labelcolor='red')
ax1_twin.set_ylim(y_min_elev, y_max_elev)  # Set consistent y-axis limits for elevation

# Create a combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Create inset axes for the detrended descending arc - fixed parameter specification
axins = inset_axes(ax1, width="40%", height="40%",
                  bbox_to_anchor=(-0.3, -0.475, 1, 1),  # x, y, width, height as 4-tuple
                  bbox_transform=ax1.transAxes)

## Plot the detrended SNR data for the descending arc
axins.plot(descending_times, detrended_snr, 'b-', linewidth=1.5)

# Set limits for inset axes
axins.set_xlim(descending_start_time_sec, descending_end_time_sec)
# Calculate appropriate y-limits for detrended data
detrend_min = min(detrended_snr) - 0.5
detrend_max = max(detrended_snr) + 0.5
axins.set_ylim(detrend_min, detrend_max)

# Remove all axis elements: ticks, tick labels, and axes lines
axins.tick_params(axis='both', which='both', bottom=False, top=False,
                 left=False, right=False, labelbottom=False, labelleft=False)
axins.grid(False)

# Create custom x-ticks for the inset (keep ticks but no labels)
inset_time_positions = []

# Start at 20:00 and create 30-min ticks
current_time = descending_start_time_sec
while current_time <= descending_end_time_sec:
    inset_time_positions.append(current_time)
    current_time += 1800  # Add 30 minutes

axins.set_xticks(inset_time_positions)
axins.set_xticklabels([])  # Empty labels
axins.set_yticklabels([])  # Empty labels

# No y-axis label for inset

plt.tight_layout()
plt.savefig('snr_full_range_with_inset.png', dpi=300)  # Save figure 1 with inset

# Create figure 2 - Zoomed data
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plot SNR data (zoomed)
ax2.plot(zoom_data['time_sec'], zoom_data['S2'], 'b-', linewidth=2.5, label='SNR')

# Set up ax2 (SNR)
ax2.set_xlim(full_start_time_sec, full_end_time_sec)
ax2.set_ylim(y_min_snr, y_max_snr)
ax2.set_xticks(time_positions)
ax2.set_xticklabels(time_labels, fontsize=16, rotation=0)
ax2.set_xlabel('Time (hh:mm)', fontsize=18, fontweight='bold')
ax2.set_ylabel('SNR (dB-Hz)', fontsize=18, fontweight='bold', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_title(common_title, fontsize=20, fontweight='bold')
ax2.set_facecolor('#f8f8f8')

# Create a second y-axis for elevation - using FULL data to match first plot
ax2_twin = ax2.twinx()
# Plot the full elevation data to keep the plots identical
ax2_twin.plot(full_data['time_sec'], full_data['elevation'], 'r-',
              linewidth=1.5, alpha=0.5, label='Elevation')
ax2_twin.set_ylabel('Elevation (degrees)', fontsize=16, color='red')
ax2_twin.tick_params(axis='y', labelcolor='red')
ax2_twin.set_ylim(y_min_elev, y_max_elev)  # Set consistent y-axis limits for elevation

# Create a combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('snr_zoomed_data.png', dpi=300)  # Save figure 2

plt.show()