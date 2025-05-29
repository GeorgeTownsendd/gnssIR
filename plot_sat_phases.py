import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter

# Define paths
base_path = "/home/george/Scripts/gnssIR/data/refl_code/2025/phase/gns1/"
file_range = range(118, 137)  # 106 to 117 inclusive
rainfall_path = "/home/george/Documents/Work/wark_rainfall_comparison/wark_rainfall.csv"

# Column names from the header
column_names = [
    "Year", "DOY", "Hour", "Phase", "Nv", "Azimuth", "Sat", "Ampl",
    "emin", "emax", "DelT", "aprioriRH", "freq", "estRH", "pk2noise", "LSPAmp"
]

# Initialize an empty list to store dataframes
dfs = []

# Load each file
for day in file_range:
    file_path = os.path.join(base_path, f"{day}.txt")
    try:
        df = pd.read_csv(file_path, sep=r'\s+', skiprows=2, names=column_names)
        df['Source_File'] = day
        dfs.append(df)
        print(f"Loaded {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Load rainfall data
try:
    rainfall_df = pd.read_csv(rainfall_path, skiprows=4)
    print(f"Loaded rainfall data: {len(rainfall_df)} rows")
except Exception as e:
    print(f"Error loading rainfall data: {e}")
    rainfall_df = None

# Combine all dataframes
if dfs:
    combined_df = pd.concat(dfs)

    # Wrap phase values to be between 0 and 360 degrees
    combined_df['Phase'] = combined_df['Phase'] % 360


    # Define function to bin azimuth into ranges - using 30-degree bins
    def assign_azimuth_bin(azimuth):
        azimuth = azimuth % 360
        bin_number = int(azimuth / 30)
        return bin_number * 30


    # Apply binning to all observations
    combined_df['Azimuth_Bin'] = combined_df['Azimuth'].apply(assign_azimuth_bin)

    # Determine if the satellite is rising or setting
    # Note: This is a simplified approach. A more accurate approach would
    # require analyzing the elevation angle's rate of change over time.
    # For now, we'll assume satellites in eastern half are rising, western half are setting
    combined_df['Arc_Type'] = 'Unknown'
    combined_df.loc[(combined_df['Azimuth_Bin'] >= 0) & (combined_df['Azimuth_Bin'] < 180), 'Arc_Type'] = 'Rising'
    combined_df.loc[(combined_df['Azimuth_Bin'] >= 180) & (combined_df['Azimuth_Bin'] < 360), 'Arc_Type'] = 'Setting'

    # Create a unique identifier for each satellite arc using Satellite, Azimuth bin, and Arc type
    combined_df['Sat_Arc'] = combined_df['Sat'].astype(str) + '_Az' + combined_df['Azimuth_Bin'].astype(str) + '_' + \
                             combined_df['Arc_Type']

    # Convert DOY and Hour to proper datetime
    combined_df['datetime'] = combined_df.apply(
        lambda row: datetime(int(row['Year']), 1, 1) + timedelta(days=int(row['DOY']) - 1) + timedelta(
            hours=row['Hour']),
        axis=1
    )

    # Count occurrences of each track across different days
    track_days = combined_df.groupby('Sat_Arc')['DOY'].nunique()

    # Filter tracks that appear in at least 3 different days
    eligible_tracks = track_days[track_days >= 3].index.tolist()

    # Get the list of all unique tracks
    all_tracks = combined_df['Sat_Arc'].unique().tolist()

    # Check if rainfall_df loaded successfully before proceeding
    if (eligible_tracks or all_tracks) and rainfall_df is not None:
        print(f"\nTracks present in at least 3 days: {eligible_tracks}")

        if eligible_tracks:
            # Check if the required columns exist after loading
            if 'Timestamp (UTC)' not in rainfall_df.columns or 'Value (mm)' not in rainfall_df.columns:
                print(f"Error: Required columns 'Timestamp (UTC)' or 'Value (mm)' not found in rainfall data.")
                print(f"Actual columns found: {rainfall_df.columns.tolist()}")
            else:
                # Process rainfall data
                # Convert timestamp to datetime
                rainfall_df['Timestamp (UTC)'] = pd.to_datetime(rainfall_df['Timestamp (UTC)'])

                # Aggregate rainfall by hour
                rainfall_hourly = rainfall_df.copy()
                rainfall_hourly['Hour'] = rainfall_hourly['Timestamp (UTC)'].dt.floor('h')
                rainfall_agg = rainfall_hourly.groupby('Hour')['Value (mm)'].sum().reset_index()
                rainfall_agg.rename(columns={'Hour': 'datetime'}, inplace=True)


                # Function to calculate circular mean of angles in degrees
                def circular_mean(angles):
                    angles_rad = np.deg2rad(angles)
                    mean_sin = np.mean(np.sin(angles_rad))
                    mean_cos = np.mean(np.cos(angles_rad))
                    mean_rad = np.arctan2(mean_sin, mean_cos)
                    return np.rad2deg(mean_rad) % 360


                # Function to calculate circular standard deviation
                def circular_std(angles):
                    angles_rad = np.deg2rad(angles)
                    mean_sin = np.mean(np.sin(angles_rad))
                    mean_cos = np.mean(np.cos(angles_rad))
                    r = np.sqrt(mean_sin ** 2 + mean_cos ** 2)  # Mean resultant length
                    return np.rad2deg(np.sqrt(-2 * np.log(r)))  # Circular standard deviation


                # Create normalized phase data for each valid track
                all_normalized_data = []

                # Process all tracks
                for track_id in all_tracks:
                    track_data = combined_df[combined_df['Sat_Arc'] == track_id].copy()
                    track_data = track_data.sort_values(by='datetime')

                    # Calculate the circular mean phase for this track
                    mean_phase = circular_mean(track_data['Phase'])

                    # Create a normalized phase that accounts for circular values
                    normalized_phase = []
                    for phase in track_data['Phase']:
                        # Calculate the shortest angular distance
                        diff = (phase - mean_phase) % 360
                        if diff > 180:
                            diff -= 360
                        normalized_phase.append(diff)

                    track_data['Phase_Normalized'] = normalized_phase
                    all_normalized_data.append(track_data)

                # Combine all normalized data
                normalized_df = pd.concat(all_normalized_data)

                # Create figure with two subplots sharing x-axis
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                               gridspec_kw={'height_ratios': [2, 1]})

                # Dynamic Y-axis calculation for ax1
                if not normalized_df.empty:
                    min_phase_dev = normalized_df['Phase_Normalized'].min()
                    max_phase_dev = normalized_df['Phase_Normalized'].max()
                    data_range = max_phase_dev - min_phase_dev
                    # Add 10% padding, or a minimum padding (e.g., 5 degrees) if range is very small
                    padding = max(data_range * 0.1, 5)
                    y_min_ax1 = min_phase_dev - padding
                    y_max_ax1 = max_phase_dev + padding
                else:
                    # Fallback if no data
                    y_min_ax1 = -180
                    y_max_ax1 = 180

                # Sort tracks by satellite number and then by azimuth bin
                sorted_tracks = []
                for track_id in all_tracks:
                    # Extract info from track_id (format: "Sat_Az{azimuth}_ArcType")
                    parts = track_id.split('_')
                    sat_num = int(parts[0])
                    azimuth = int(parts[1][2:])
                    arc_type = parts[2]

                    sorted_tracks.append((sat_num, azimuth, track_id, arc_type))

                # Sort by satellite number first, then by azimuth
                sorted_tracks.sort()

                # Plot normalized data for each track in the top subplot
                for sat_num, azimuth, track_id, arc_type in sorted_tracks:
                    track_data = normalized_df[normalized_df['Sat_Arc'] == track_id]

                    # Format the label with satellite number, azimuth bin and arc type
                    label = f'Sat {sat_num} (Az ~{azimuth}°, {arc_type})'

                    # Plot with a solid line and markers
                    ax1.plot(track_data['datetime'], track_data['Phase_Normalized'], 'o-',
                             linewidth=1, markersize=3, label=label)

                # Set up the top subplot
                ax1.set_ylabel('Phase Deviation from Mean (degrees)', fontsize=11)
                ax1.set_title('Satellite Phase Changes Over Time (Track-Based Normalization)', fontsize=13)
                ax1.grid(True, which='major', alpha=0.5, linewidth=0.8)
                ax1.grid(True, which='minor', alpha=0.2, linewidth=0.4)
                ax1.set_ylim(y_min_ax1, y_max_ax1)  # Use dynamic limits
                ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')

                # Sort all data by datetime for calculations
                normalized_df = normalized_df.sort_values(by='datetime')

                # Define 6-hour windows
                min_time = normalized_df['datetime'].min()
                max_time = normalized_df['datetime'].max()

                # Create 6-hour time bins from min to max time
                time_bins = []
                current_time = min_time.floor('H')  # Start bins on the hour
                while current_time <= max_time:
                    time_bins.append(current_time)
                    current_time += timedelta(hours=6)
                if time_bins[-1] <= max_time:  # Ensure last bin covers max_time if needed
                    time_bins.append(current_time)

                # Calculate statistics for each bin
                bin_times = []
                bin_means = []
                bin_errors = []

                for i in range(len(time_bins) - 1):
                    start_time = time_bins[i]
                    end_time = time_bins[i + 1]

                    # Get data in this time window
                    window_data = normalized_df[(normalized_df['datetime'] >= start_time) &
                                                (normalized_df['datetime'] < end_time)]

                    if not window_data.empty:
                        bin_times.append(start_time + (end_time - start_time) / 2)  # Midpoint of the bin
                        bin_means.append(np.mean(window_data['Phase_Normalized']))  # Mean of normalized values
                        # For error bars, use standard error of the mean
                        if len(window_data) > 1:
                            bin_errors.append(np.std(window_data['Phase_Normalized']) / np.sqrt(len(window_data)))
                        else:
                            bin_errors.append(0)  # No error if only one point

                # Create a secondary y-axis for the bottom subplot
                ax3 = ax2.twinx()

                # Plot 6-hour rolling average with error bars in the bottom subplot (left y-axis)
                ax2.errorbar(bin_times, bin_means, yerr=bin_errors, fmt='o-', linewidth=2, color='blue',
                             ecolor='red',  # Changed error bar color
                             capsize=5, markersize=5, label='6hr Avg Phase Dev')

                # Plot rainfall data as bar chart on the right y-axis
                # Filter rainfall to the same date range as our phase data
                filtered_rainfall = rainfall_agg[
                    (rainfall_agg['datetime'] >= min_time.floor('H')) &  # Align with phase data range start
                    (rainfall_agg['datetime'] <= max_time.ceil('H'))  # Align with phase data range end
                    ]

                # Plot rainfall data as blue bars
                if not filtered_rainfall.empty:
                    bars = ax3.bar(filtered_rainfall['datetime'], filtered_rainfall['Value (mm)'],
                                   width=timedelta(hours=1), color='skyblue', alpha=0.7, align='edge',
                                   label='Hourly Rainfall')
                else:
                    print("No rainfall data within the phase data time range.")

                # Set up the bottom subplot axes
                ax2.set_xlabel('Time', fontsize=12)
                ax2.set_ylabel('6-Hour Avg Phase Deviation', fontsize=11, color='blue')
                ax2.tick_params(axis='y', labelcolor='blue', labelsize=10)

                ax3.set_ylabel('Hourly Rainfall (mm)', fontsize=11, color='skyblue')
                ax3.tick_params(axis='y', labelcolor='skyblue', labelsize=10)
                ax3.set_ylim(bottom=0)

                # Add a title to the bottom subplot that includes both metrics
                ax2.set_title('6-Hour Phase Deviation (mean ± SE) and Hourly Rainfall', fontsize=12)

                # Add legends to the bottom subplot for both metrics
                lines1, labels1 = ax2.get_legend_handles_labels()
                lines2, labels2 = ax3.get_legend_handles_labels()
                if lines1 or lines2:
                    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize='small')

                # Format x-axis dates
                date_format = DateFormatter('%m-%d %H:%M')
                ax2.xaxis.set_major_formatter(date_format)
                fig.autofmt_xdate()  # Rotate date labels

                # Grid settings
                ax2.grid(True, which='major', axis='y', alpha=0.5, linewidth=0.8)
                ax3.grid(False)

                # Adjust the layout
                plt.subplots_adjust(right=0.88, hspace=0.15)
                plt.tight_layout()
                # Show the plot
                plt.show()

    # Handle cases where rainfall data failed or no eligible tracks
    elif rainfall_df is None:
        print("Could not proceed because rainfall data failed to load.")
    elif not eligible_tracks:
        print("No tracks appear in at least 3 different days.")
    else:
        # This case should ideally not be reached if the above checks work
        print("Could not proceed. Check eligible tracks and rainfall data loading.")

else:
    print("No phase data was loaded.")