#!/usr/bin/env python3
"""
Simple script to plot two GNSS-IR arcs stacked vertically.
All parameters are specified directly in the script.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def read_snr_file(file_path, sat_num):
    """Read SNR data for a specific satellite from an SNR file.

    Args:
        file_path (str): Path to the SNR file
        sat_num (int): Satellite number to filter for

    Returns:
        tuple: (elevation, snr, azimuth, time) arrays
    """
    try:
        # Load the data
        data = np.loadtxt(file_path)

        # Filter for the specified satellite
        # Assuming the satellite number is in column 4 (0-based index)
        # Adjust this if your file format is different
        sat_mask = data[:, 4] == sat_num
        filtered_data = data[sat_mask]

        if filtered_data.size == 0:
            print(f"Warning: No data found for satellite {sat_num} in {file_path}")
            return None, None, None, None

        # Extract elevation (column 1), SNR (column 3), azimuth (column 2), time (column 0)
        # Adjust these indices if your file format is different
        elevation = filtered_data[:, 1]
        snr = filtered_data[:, 3]
        azimuth = filtered_data[:, 2]
        time = filtered_data[:, 0]

        return elevation, snr, azimuth, time

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None, None, None


def plot_arcs(file1, sat1, file2, sat2, output_file=None):
    """Plot two arcs stacked vertically.

    Args:
        file1 (str): Path to first SNR file
        sat1 (int): Satellite number for first file
        file2 (str): Path to second SNR file
        sat2 (int): Satellite number for second file
        output_file (str, optional): Path to save the output figure
    """
    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Process first file
    elev1, snr1, az1, time1 = read_snr_file(file1, sat1)
    if elev1 is not None:
        # Convert elevation to sin(elevation)
        x1 = np.sin(np.radians(elev1))

        # Calculate mean azimuth
        mean_az1 = np.mean(az1)

        # Plot data
        ax1.plot(x1, snr1, 'r.', alpha=0.7)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Set title and labels
        filename1 = Path(file1).name
        ax1.set_title(f"Satellite {sat1} - {filename1}\nMean Azimuth: {mean_az1:.1f}°", fontsize=10)
        ax1.set_ylabel("SNR (dB-Hz)")
    else:
        ax1.text(0.5, 0.5, f"No data found for satellite {sat1}",
                 ha='center', va='center', transform=ax1.transAxes)

    # Process second file
    elev2, snr2, az2, time2 = read_snr_file(file2, sat2)
    if elev2 is not None:
        # Convert elevation to sin(elevation)
        x2 = np.sin(np.radians(elev2))

        # Calculate mean azimuth
        mean_az2 = np.mean(az2)

        # Plot data
        ax2.plot(x2, snr2, 'g.', alpha=0.7)
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Set title and labels
        filename2 = Path(file2).name
        ax2.set_title(f"Satellite {sat2} - {filename2}\nMean Azimuth: {mean_az2:.1f}°", fontsize=10)
        ax2.set_ylabel("SNR (dB-Hz)")
        ax2.set_xlabel("sin(Elevation Angle)")
    else:
        ax2.text(0.5, 0.5, f"No data found for satellite {sat2}",
                 ha='center', va='center', transform=ax2.transAxes)

    # Adjust layout
    plt.tight_layout()

    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    # Show the plot
    plt.show()


def main():
    """Hardcoded parameters instead of command line arguments."""
    # Define input parameters directly in the script
    # In filename format gns11050.25.snr88:
    # - "105" is the day of year
    # - "25" is the year (2025)
    # - Satellite number needs to be specified separately

    file1 = "/home/george/Scripts/gnssIR/data/refl_code/2025/snr/gns1/gns11160.25.snr88"
    sat1 = 1  # Specify the actual satellite number here

    file2 = "/home/george/Scripts/gnssIR/data/refl_code/2025/snr/gns1/gns11170.25.snr88"
    sat2 = 1  # Specify the actual satellite number here

    # Create output filename based on input files and satellites
    output_file = f"arc_comparison_sat{sat1}_vs_sat{sat2}_doy105v104.png"

    print(f"Comparing satellite {sat1} (day 105, 2025) from {Path(file1).name}")
    print(f"with satellite {sat2} (day 104, 2025) from {Path(file2).name}")

    plot_arcs(file1, sat1, file2, sat2, output_file)


if __name__ == "__main__":
    main()