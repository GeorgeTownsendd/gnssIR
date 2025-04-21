#!/usr/bin/env python3
"""
Compare GNSS SNR data for a specific satellite (PRN) across multiple days.
This script processes SNR files and plots the L2C GPS signal data for ascending arcs.

Example usage:
    python compare_phase_by_day.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
import os

# List of SNR files to compare (EDIT THESE)
SNR_FILES = [
    "/home/george/Scripts/gnssIR/data/refl_code/2025/snr/gns1/gns11050.25.snr88",
    "/home/george/Scripts/gnssIR/data/refl_code/2025/snr/gns1/gns11060.25.snr88",
    "/home/george/Scripts/gnssIR/data/refl_code/2025/snr/gns1/gns11070.25.snr88",
    # Add more files as needed
]

# Define reflector height (in meters) - EDIT THIS
RH = 3.0

# Specify which GPS satellite PRN to analyze - EDIT THIS
TARGET_PRN = 121

# Data filtering and processing options - EDIT THESE
DECIMATE = 30  # Decimate data to this interval in seconds (e.g., 30s) - set to 1 for no decimation
MIN_ELEV = 5  # Minimum elevation angle (degrees)
MAX_ELEV = 25  # Maximum elevation angle (degrees)


def read_snr_file(file_path, target_prn):
    """
    Read and filter SNR data from file for a specific PRN

    Args:
        file_path (str): Path to the SNR file
        target_prn (int): Target PRN/satellite number to extract

    Returns:
        dict: Dictionary with satellite data (or None if not found)
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None

    try:
        # Read the SNR file
        data = np.loadtxt(file_path)

        # Filter to only include rows for the target PRN
        prn_mask = data[:, 0] == target_prn
        if not np.any(prn_mask):
            print(f"Warning: PRN {target_prn} not found in {file_path}")
            return None

        prn_data = data[prn_mask]

        # Sort by time to ensure chronological order
        prn_data = prn_data[np.argsort(prn_data[:, 3])]

        # Filter for rising arc (positive elevation rate) between MIN_ELEV and MAX_ELEV degrees
        arc_data = []
        in_arc = False
        last_time = None

        for row in prn_data:
            elev_angle = row[1]
            azimuth = row[2]
            time_secs = row[3]
            elev_rate = row[4]  # Elevation angle rate of change
            snr = row[7]  # L2 SNR data (index 7)

            # Check if we're in a rising arc within elevation limits
            if MIN_ELEV <= elev_angle <= MAX_ELEV and elev_rate > 0:
                # Apply decimation if needed
                if DECIMATE > 1:
                    # For the first point or if enough time has passed since the last point
                    if last_time is None or (time_secs - last_time) >= DECIMATE:
                        in_arc = True
                        arc_data.append({
                            'elev_angle': elev_angle,
                            'azimuth': azimuth,
                            'time_secs': time_secs,
                            'snr': snr
                        })
                        last_time = time_secs
                else:
                    # No decimation, include all points
                    in_arc = True
                    arc_data.append({
                        'elev_angle': elev_angle,
                        'azimuth': azimuth,
                        'time_secs': time_secs,
                        'snr': snr
                    })
            elif in_arc and (elev_angle > MAX_ELEV or elev_rate <= 0):
                # We've reached the end of the rising arc
                break

        if not arc_data:
            print(f"Warning: No valid ascending arc data found for PRN {target_prn} in {file_path}")
            return None

        return arc_data

    except Exception as e:
        print(f"Error reading SNR file: {e}")
        return None


def fit_sinusoid(elev_angles, snr_data, rh):
    """
    Fit sinusoid to SNR data using least squares

    Args:
        elev_angles (np.array): Elevation angles in degrees
        snr_data (np.array): SNR data values
        rh (float): Reflector height in meters

    Returns:
        tuple: (x, fitted_curve, amplitude, phase)
    """
    # L2 wavelength in meters
    L2_wavelength = 0.24421

    # Convert reflector height from meters to wavelengths
    h2wL2 = 4 * np.pi / L2_wavelength

    # Sine of elevation angle
    x = np.sin(elev_angles * np.pi / 180)

    # Angular frequency based on reflector height
    omega = rh * h2wL2

    # Set up design matrix for least squares
    A = np.zeros((len(x), 2))
    A[:, 0] = np.sin(omega * x)
    A[:, 1] = np.cos(omega * x)

    # Solve least squares
    C = A.T @ A
    cvec = A.T @ snr_data

    try:
        xvec = inv(C) @ cvec
        As, Ac = xvec

        # Calculate amplitude and phase
        amplitude = np.sqrt(As ** 2 + Ac ** 2)
        phase = np.arctan2(Ac, As)

        # Generate fitted curve
        fitted_curve = amplitude * np.sin(omega * x + phase)

        return x, fitted_curve, amplitude, phase

    except np.linalg.LinAlgError:
        print("Error: Could not fit sinusoid - singular matrix encountered.")
        return x, np.zeros_like(x), 0, 0


def format_time_range(seconds_array):
    """Convert seconds of day to formatted time range string"""
    start_sec = int(seconds_array[0])
    end_sec = int(seconds_array[-1])

    start_h, remainder = divmod(start_sec, 3600)
    start_m, start_s = divmod(remainder, 60)

    end_h, remainder = divmod(end_sec, 3600)
    end_m, end_s = divmod(remainder, 60)

    return f"{start_h:02d}:{start_m:02d}:{start_s:02d} - {end_h:02d}:{end_m:02d}:{end_s:02d}"


def extract_day_from_filename(filename):
    """Extract the day of year from an SNR filename"""
    # Parse from format like "gns11050" where 105 is the day of year
    basename = os.path.basename(filename).split('.')[0]
    # Day number is characters 5-7 (0-indexed)
    try:
        # Remove the last digit (which is always 0 as per your comment)
        day_str = basename[5:8]
        # Convert to integer (dropping the trailing 0)
        day = int(day_str[:-1])
        return day
    except (IndexError, ValueError):
        return None


def compare_satellite_data(data_list, day_labels, rh):
    """
    Compare SNR data for a specific satellite across multiple days

    Args:
        data_list (list): List of data for each day
        day_labels (list): Labels for each day
        rh (float): Reflector height in meters
    """
    # Count valid datasets
    valid_data = [data for data in data_list if data is not None]
    if not valid_data:
        print(f"No valid data for PRN {TARGET_PRN}")
        return

    n_plots = len(valid_data)

    # Create a list to store all x and y values for consistent scaling
    all_x_values = []
    all_snr_values = []
    all_fit_values = []

    # Process each dataset
    processed_data = []
    for i, data in enumerate(data_list):
        if data is None:
            continue

        # Extract arrays from data dictionaries
        elev = np.array([d['elev_angle'] for d in data])
        snr = np.array([d['snr'] for d in data])
        time_secs = np.array([d['time_secs'] for d in data])
        az = np.array([d['azimuth'] for d in data])

        # Fit sinusoid
        x, yc, amplitude, phase = fit_sinusoid(elev, snr, rh)

        # Calculate mean azimuth
        mean_az = np.mean(az)

        # Format time range
        time_range = format_time_range(time_secs)

        # Store processed data
        processed_data.append({
            'day': day_labels[i],
            'x': x,
            'snr': snr,
            'yc': yc,
            'time_secs': time_secs,
            'amplitude': amplitude,
            'phase': phase,
            'mean_az': mean_az,
            'time_range': time_range
        })

        # Collect values for scale normalization
        all_x_values.extend(x)
        all_snr_values.extend(snr)
        all_fit_values.extend(yc)

    # Calculate common plot limits
    x_min, x_max = min(all_x_values), max(all_x_values)
    all_y_values = all_snr_values + all_fit_values
    y_min, y_max = min(all_y_values), max(all_y_values)

    # Add some padding
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    # Create figure with n_plots + 1 subplots (individual days + superimposed plot)
    fig = plt.figure(figsize=(12, 3 * (n_plots + 1)), dpi=100)
    gs = fig.add_gridspec(n_plots + 1, 1)

    # Create superimposed plot at the top
    superimposed_ax = fig.add_subplot(gs[0, 0])

    # Color cycle for different days
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    # Plot superimposed data
    legend_entries = []
    for i, data in enumerate(processed_data):
        color = colors[i % len(colors)]

        # Plot both the data points and model fit
        # Make data points more transparent to highlight the fitted wave
        superimposed_ax.plot(data['x'], data['snr'], '.', color=color, alpha=0.2,
                             markersize=4, label=f'Day {data["day"]} Data')
        superimposed_ax.plot(data['x'], data['yc'], '-', color=color, linewidth=2.0,
                             alpha=1.0, label=f'Day {data["day"]} Fit')

        # Add phase and amplitude text
        legend_entries.append(f'Day {data["day"]}: Amp={data["amplitude"]:.2f}, Phase={np.degrees(data["phase"]):.1f}°')

    superimposed_ax.grid(True, linestyle='--', alpha=0.7)
    superimposed_ax.set_xlabel('sin(elevation angle)', fontsize=10)
    superimposed_ax.set_ylabel('SNR (dB-Hz)', fontsize=10)
    superimposed_ax.set_title(f'Superimposed SNR Data - GPS PRN {TARGET_PRN}', fontsize=14)
    superimposed_ax.set_xlim(x_min - x_padding, x_max + x_padding)
    superimposed_ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Add legend with phase and amplitude info
    superimposed_ax.legend(legend_entries, loc='upper right', fontsize=8)

    # Create individual day subplots
    axes = [fig.add_subplot(gs[i + 1, 0]) for i in range(n_plots)]

    # Plot each day's data
    for i, (data, ax) in enumerate(zip(processed_data, axes)):
        color = colors[i % len(colors)]

        # Plot SNR data and model fit
        # Make data points more transparent to highlight the fitted wave
        ax.plot(data['x'], data['snr'], '.', color=color, alpha=0.3, label='SNR Data')
        ax.plot(data['x'], data['yc'], 'k-', linewidth=2.0, label='Model Fit')
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set labels and title
        ax.set_ylabel('SNR (dB-Hz)', fontsize=10)
        ax.set_title(f'GPS PRN {TARGET_PRN} - Day {data["day"]}', fontsize=12)

        # Add legend to the first plot only to avoid redundancy
        if i == 0:
            ax.legend(loc='upper right')

        # Set consistent axis limits
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Add x-label to the bottom plot only
        if i == n_plots - 1:
            ax.set_xlabel('sin(elevation angle)', fontsize=10)

        # Add info text box
        info_text = (f'Satellite: GPS PRN {TARGET_PRN}\n'
                     f'Mean Az: {data["mean_az"]:.1f}°\n'
                     f'RH: {rh:.3f} m\n'
                     f'Time: {data["time_range"]}\n'
                     f'Amplitude: {data["amplitude"]:.2f}\n'
                     f'Phase: {np.degrees(data["phase"]):.2f}°')

        ax.text(0.98, 0.03, info_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(
                    facecolor='white',
                    alpha=0.9,
                    edgecolor='gray',
                    boxstyle='round,pad=0.4',
                    linewidth=1,
                ),
                zorder=5
                )

    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the comparison"""
    print(f"Reading SNR files to compare PRN {TARGET_PRN} across {len(SNR_FILES)} days...")
    print(f"Using elevation range: {MIN_ELEV}° to {MAX_ELEV}°")
    if DECIMATE > 1:
        print(f"Decimating data to {DECIMATE}-second intervals")

    # Extract day identifiers from filenames
    day_labels = []
    for file_path in SNR_FILES:
        day = extract_day_from_filename(file_path)
        if day:
            day_labels.append(str(day))
        else:
            # Use fallback naming if day extraction fails
            day_labels.append(os.path.basename(file_path).split('.')[0])

    # Read specific satellite data from all SNR files
    data_list = []
    for i, file_path in enumerate(SNR_FILES):
        data = read_snr_file(file_path, TARGET_PRN)
        data_list.append(data)

        if data is None:
            print(f"Error: PRN {TARGET_PRN} not found or no valid arc in file: {file_path}")
            print("Available PRNs in this file:")
            try:
                all_data = np.loadtxt(file_path)
                available_prns = np.unique(all_data[:, 0]).astype(int)
                print(available_prns)
            except Exception as e:
                print(f"Could not read PRNs from file: {e}")
        else:
            print(f"Found {len(data)} data points for PRN {TARGET_PRN} in day {day_labels[i]}")

    # Check if we have any valid data
    if all(data is None for data in data_list):
        print(f"Error: No valid data found for PRN {TARGET_PRN} in any files.")
        return

    # Compare data across all days
    print(f"Comparing data for PRN {TARGET_PRN} across {len(SNR_FILES)} days...")
    compare_satellite_data(data_list, day_labels, RH)

    print("Comparison complete!")


if __name__ == "__main__":
    main()