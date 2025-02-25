import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the base directory
base_dir = '/home/george/Documents/Work/azimuth_mask_testing'


def read_vwc_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_lines = [line for line in lines if not line.strip().startswith('%')]
    data = []

    for line in data_lines:
        values = line.strip().split()
        data.append({
            'FracYr': float(values[0]),
            'Year': int(values[1]),
            'DOY': int(values[2]),
            'VWC': float(values[3]),
            'Month': int(values[4]),
            'Day': int(values[5])
        })

    return pd.DataFrame(data)


# Read the VWC data
full_vwc = read_vwc_file(f'{base_dir}/full_sky_vwc.txt')
half_vwc = read_vwc_file(f'{base_dir}/half_sky_vwc.txt')

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Full Sky vs Half Sky VWC Comparison', fontsize=14)

# Plot 1: VWC over time
ax1.plot(full_vwc['FracYr'], full_vwc['VWC'], 'b-', label='Full Sky', linewidth=2)
ax1.plot(half_vwc['FracYr'], half_vwc['VWC'], 'r--', label='Half Sky', linewidth=2)
ax1.set_xlabel('Fractional Year')
ax1.set_ylabel('VWC')
ax1.set_title('VWC Time Series')
ax1.legend()
ax1.grid(True)

# Plot 2: Histogram of VWC differences
merged_vwc = pd.merge(full_vwc, half_vwc, on='FracYr', suffixes=('_full', '_half'))
vwc_diff = merged_vwc['VWC_half'] - merged_vwc['VWC_full']  # half - full

# Add mean and std to histogram
mean_diff = vwc_diff.mean()
std_diff = vwc_diff.std()

ax2.hist(vwc_diff, bins=20, color='skyblue', edgecolor='black')
ax2.set_xlabel('VWC Differences (Half Sky - Full Sky)')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of VWC Differences')
ax2.grid(True)

# Print basic statistics
print("\nVWC Differences Statistics:")
print(f"Mean: {mean_diff:.3f}")
print(f"Std Dev: {std_diff:.3f}")
print(f"Min: {vwc_diff.min():.3f}")
print(f"Max: {vwc_diff.max():.3f}")

# Adjust layout
plt.tight_layout()
plt.show()