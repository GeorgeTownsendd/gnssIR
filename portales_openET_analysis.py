import numpy as np
import matplotlib.pyplot as plt

def read_gnss_file(filepath):
    # Skip the header row starting with %
    data = np.loadtxt(filepath, skiprows=1)
    elev_angle = data[:, 0]
    # Convert elevation angle to sin(elevation)
    sin_elev = np.sin(np.deg2rad(elev_angle))
    dsnr = data[:, 1]
    time = data[:, 2]
    return sin_elev, dsnr, time

# Define the base path
base_path = "/home/george/Documents/Work/arcs_comparison"

# Example files - you can change these to specific files you want to compare
kita_success = f"{base_path}/kita/sat001_L2_G_az215.txt"
kita_fail = f"{base_path}/kita/failQC/sat001_L2_G_az034.txt"
p038_success = f"{base_path}/p038/sat001_L2_G_az036.txt"
p038_fail = f"{base_path}/p038/failQC/sat017_L2_G_az100.txt"

# Create figure and subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('GNSS-IR Arc Comparison: KITA vs P038')

# Plot KITA successful arc
sin_elev, dsnr, time = read_gnss_file(kita_success)
ax1.plot(sin_elev, dsnr, 'b.-')
ax1.set_title('KITA - Successful Arc')
ax1.set_xlabel('sin(Elevation Angle)')
ax1.set_ylabel('dSNR (V/V)')
ax1.grid(True)

# Plot P038 successful arc
sin_elev, dsnr, time = read_gnss_file(p038_success)
ax2.plot(sin_elev, dsnr, 'g.-')
ax2.set_title('P038 - Successful Arc')
ax2.set_xlabel('sin(Elevation Angle)')
ax2.set_ylabel('dSNR (V/V)')
ax2.grid(True)

# Plot KITA failed arc
sin_elev, dsnr, time = read_gnss_file(kita_fail)
ax3.plot(sin_elev, dsnr, 'r.-')
ax3.set_title('KITA - Failed Arc')
ax3.set_xlabel('sin(Elevation Angle)')
ax3.set_ylabel('dSNR (V/V)')
ax3.grid(True)

# Plot P038 failed arc
sin_elev, dsnr, time = read_gnss_file(p038_fail)
ax4.plot(sin_elev, dsnr, 'm.-')
ax4.set_title('P038 - Failed Arc')
ax4.set_xlabel('sin(Elevation Angle)')
ax4.set_ylabel('dSNR (V/V)')
ax4.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()