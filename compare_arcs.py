import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def decimate_data(data, sample_rate):
    """
    Decimate data based on sample rate in seconds.
    Assumes data is collected at 1Hz by default.
    """
    if sample_rate <= 1:
        return data
    return data[::sample_rate]


# Setup the subplot figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('SNR Comparisons for GNSA, GNSB, GNSC, and GNSD', fontsize=16)

# List of files to process
files = [
    'sat032_L2_G_az344.txt'
]

# List of axes for easy iteration
axes = [ax1, ax2, ax3, ax4]

base_path = "/home/george/Scripts/gnssIR/data/refl_code/2025/arcs"

# Define stations and their labels
stations = ['gnsa', 'gnsb', 'gnsc', 'gnsd']
labels = ['Vertical patch antenna', 'Horizontal patch antenna', 'Vertical LEIAR20', 'Horizontal LEIAR20']

# Set sample rate (in seconds)
sample_rate = 15  # Change this value to decimate data (e.g., 5 for 5s samples, 30 for 30s samples)

# Process each file
for ax, filename in zip(axes, files):
    for i, station in enumerate(stations):
        # Read the file
        filepath = f"{base_path}/{station}/037/failQC/{filename}"
        try:
            data = np.loadtxt(filepath, skiprows=1)
            # Decimate the data
            data = decimate_data(data, sample_rate)
            # Convert elevation to sin(elevation) and get dSNR
            sin_elev = np.sin(np.radians(data[:, 0]))
            dsnr = data[:, 1]
            # Plot with increased opacity (alpha=1.0 is fully opaque)
            ax.plot(sin_elev, dsnr, label=labels[i], alpha=1.0)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # Customize each subplot
    ax.set_xlabel('sin(elevation angle)')
    ax.set_ylabel('dSNR (volts/volts)')
    ax.set_title(f'Arc: {filename.replace(".txt", "")}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
print(f"Plot generated with {sample_rate}s sample rate")