import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Define path pattern
base_path = "/home/george/Scripts/gnssIR/data/refl_code/2025/phase/gns1/"
file_range = range(104, 109)  # 104 to 108 inclusive

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
        # Skip the first two lines (header rows)
        df = pd.read_csv(file_path, delim_whitespace=True, skiprows=2, names=column_names)
        # Add the file's day to track the source
        df['Source_File'] = day
        dfs.append(df)
        print(f"Loaded {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Combine all dataframes
if dfs:
    combined_df = pd.concat(dfs)

    # Wrap phase values to be between 0 and 360 degrees
    combined_df['Phase'] = combined_df['Phase'] % 360

    # Convert DOY and Hour to proper datetime
    combined_df['datetime'] = combined_df.apply(
        lambda row: datetime(int(row['Year']), 1, 1) + timedelta(days=int(row['DOY']) - 1) + timedelta(
            hours=row['Hour']),
        axis=1
    )

    # Count occurrences of each satellite across different days
    sat_days = combined_df.groupby('Sat')['DOY'].nunique()

    # Filter satellites that appear in at least 3 different days
    eligible_sats = sat_days[sat_days >= 3].index.tolist()

    if eligible_sats:
        print(f"\nSatellites present in at least 3 days: {eligible_sats}")

        # Create a plot for phase changes with extra space on the right for the legend
        plt.figure(figsize=(12, 8))

        # Dictionary to store legend information
        legend_info = []

        # Plot each eligible satellite
        for sat in eligible_sats:
            sat_data = combined_df[combined_df['Sat'] == sat]

            # Sort by datetime to ensure chronological order
            sat_data = sat_data.sort_values(by='datetime')

            # Plot this satellite's phase data
            line, = plt.plot(sat_data['datetime'], sat_data['Phase'], 'o-', linewidth=2, markersize=5)

            # Add to legend
            legend_info.append(f'Satellite {sat}')

        # Set up the plot
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Phase (degrees)', fontsize=12)
        plt.title('Satellite Phase Changes Over Time (0-360 degrees)', fontsize=14)

        # Set up grid with more prominent major grid lines
        plt.grid(True, which='major', alpha=0.5, linewidth=1.0)
        plt.grid(True, which='minor', alpha=0.2, linewidth=0.5)

        # Place the legend outside the plot
        plt.legend(legend_info, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.ylim(0, 360)  # Set y-axis limits to 0-360

        # Let matplotlib handle the date formatting
        plt.gcf().autofmt_xdate()

        # Adjust the layout to make room for the legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)

        # Show the plot
        plt.show()

    else:
        print("No satellites appear in at least 3 different days")
else:
    print("No data was loaded")