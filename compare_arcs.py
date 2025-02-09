import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import sys

# Set quadrant selection (1-4); set to None to use all subplots
selected_quadrant = 1 # Change to 1, 2, 3, or 4 to focus on one quadrant

def decimate_data(data, sample_rate):
    """
    Decimate data based on sample rate in seconds.
    Assumes data is collected at 1Hz by default.
    """
    if sample_rate <= 1:
        return data
    return data[::sample_rate]

def find_closest_file(directory, target_filename, tolerance=25):
    """
    Finds the closest matching file in a directory based on azimuth tolerance.
    """
    target_az = int(re.search(r'az(\d+)', target_filename).group(1))
    files = list(directory.glob('*.txt'))

    closest_file = None
    min_diff = tolerance + 1  # Start with a value greater than tolerance

    for file in files:
        match = re.search(r'az(\d+)', file.name)
        if match:
            az = int(match.group(1))
            diff = abs(az - target_az)
            if diff <= tolerance and diff < min_diff:
                closest_file = file
                min_diff = diff

    return closest_file

# List of files to process
files = ['sat229_L1_E_az349.txt']

# Base data path
base_path = Path("/home/george/Scripts/gnssIR/data/refl_code/2025/arcs")

# Define stations and their labels
stations = ['g1s1', 'g2s1', 'g3s1', 'g4s1']
labels = ['Horizontal Patch', 'Vertical Patch', 'Horizontal GPS500', 'Vertical GPS500']

# Set sample rate (in seconds)
sample_rate = 15  # Change this value to decimate data (e.g., 5 for 5s samples, 30 for 30s samples)

# Create subplots
if selected_quadrant is None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SNR Comparisons for GNSA, GNSB, GNSC, and GNSD', fontsize=16)
    axes = axes.flatten()
else:
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f'SNR Comparison - Quadrant {selected_quadrant}', fontsize=16)
    axes = [ax]

# Process selected quadrant or all quadrants
quadrant_map = {1: 0, 2: 1, 3: 2, 4: 3}
if selected_quadrant is not None:
    axes = [axes[quadrant_map[selected_quadrant]]]

for ax, filename in zip(axes, files):
    for i, station in enumerate(stations):
        # Construct potential directories
        failqc_dir = base_path / station / "037" / "failQC"
        default_dir = base_path / station / "037"

        # Find closest matching file
        filepath = find_closest_file(failqc_dir, filename) or find_closest_file(default_dir, filename)

        if filepath:
            try:
                data = np.loadtxt(filepath, skiprows=1)
                # Decimate the data
                data = decimate_data(data, sample_rate)
                # Convert elevation to sin(elevation) and get dSNR
                sin_elev = np.sin(np.radians(data[:, 0]))
                dsnr = data[:, 1]
                # Plot with increased opacity
                ax.plot(sin_elev, dsnr, label=labels[i], alpha=1.0)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        else:
            print(f"No matching file found within tolerance for {filename} in {station}")

    # Customize the plot
    ax.set_xlabel('sin(elevation angle)')
    ax.set_ylabel('dSNR (volts/volts)')
    ax.set_title(f'Arc: {filename.replace(".txt", "")}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Adjust layout and show plot
plt.tight_layout()
plt.show()
print(f"Plot generated with {sample_rate}s sample rate")
