import os
import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from gnssrefl import rinex2snr_cl, gnssir_cl, vwc_cl
from arc_management import copy_arcs


def load_vwc_data(station, year, doy):
    """Load VWC data for a specific station and date"""
    vwc_file = f'data/refl_code/Files/{station.lower()}/{station.lower()}_vwc.txt'

    try:
        # Skip the header lines (first 4 lines)
        data = np.loadtxt(vwc_file, skiprows=4)

        # Find matching row for year and doy
        matches = (data[:, 1] == year) & (data[:, 2] == doy)

        if np.any(matches):
            return float(data[matches][0, 3])  # VWC is in column 3
        else:
            print(f"Warning: No VWC data found for station {station}, year {year}, doy {doy}")
            return None

    except Exception as e:
        print(f"Error loading VWC data: {e}")
        return None


def load_full_vwc_data(station, year):
    """Load all VWC data for a station and year

    Args:
        station (str): Station identifier
        year (int): Year

    Returns:
        tuple: (doys, vwc_values) arrays for the full year
    """
    vwc_file = f'data/refl_code/Files/{station.lower()}/{station.lower()}_vwc.txt'

    try:
        # Skip the header lines (first 4 lines)
        data = np.loadtxt(vwc_file, skiprows=4)

        # Filter for the specified year
        year_mask = data[:, 1] == year
        year_data = data[year_mask]

        return year_data[:, 2], year_data[:, 3]  # DOY, VWC

    except Exception as e:
        print(f"Error loading VWC data: {e}")
        return None, None


def load_rh_data(station, year, track):
    """Load RH data for a specific station and track"""
    rh_file = f'data/refl_code/input/{station.lower()}_phaseRH.txt'

    try:
        # Read all lines
        with open(rh_file, 'r') as f:
            lines = f.readlines()

        # Filter out comment lines and empty lines
        data_lines = [line for line in lines if not line.strip().startswith('%') and line.strip()]

        # Convert to numpy array
        data = np.array([list(map(float, line.split())) for line in data_lines])

        # Track number is the first column (0-based index)
        matches = data[:, 0] == track

        if np.any(matches):
            row = data[matches][0]
            return float(row[1]), int(row[2]), float(row[3])  # RH, SatNu, MeanAz
        else:
            print(f"Warning: No RH data found for station {station}, track {track}")
            return None, None, None

    except Exception as e:
        print(f"Error loading RH data: {e}")
        return None, None, None


def format_time_range(seconds_array):
    """Convert seconds of day to formatted time range string"""
    start_time = timedelta(seconds=float(seconds_array[0]))
    end_time = timedelta(seconds=float(seconds_array[-1]))
    return f"{str(start_time).split('.')[0]} - {str(end_time).split('.')[0]}"


def fit_sinusoid(elev_angles, snr, RH):
    """Fit sinusoid to SNR data using least squares"""
    L2 = 24.421  # L2 wavelength [cm]
    h2wL2 = 4 * np.pi / L2
    x = np.sin(elev_angles * np.pi / 180)
    omega = RH * h2wL2

    A = np.zeros((len(x), 2))
    A[:, 0] = np.sin(omega * x)
    A[:, 1] = np.cos(omega * x)

    C = A.T @ A
    cvec = A.T @ snr
    xvec = inv(C) @ cvec

    As, Ac = xvec
    A = np.sqrt(As ** 2 + Ac ** 2)
    phi = np.arctan2(Ac, As)

    yc = A * np.sin(omega * x + phi)

    return x, yc, A, phi


def compare_arcs(station, year, doy1, doy2, track=1, results_dir='results/'):
    """Run GNSS-IR analysis and compare specific arc waveforms between two days."""
    station = station.lower()

    # Load full year VWC data
    doys, vwc_values = load_full_vwc_data(station, year)
    if doys is None:
        print("Error: Could not load VWC data")
        return

    # Load VWC data for the specific days
    vwc1 = load_vwc_data(station, year, doy1)
    vwc2 = load_vwc_data(station, year, doy2)

    if vwc1 is None or vwc2 is None:
        print("Warning: Missing VWC data for comparison days, analysis may be incomplete")
        vwc1 = vwc1 if vwc1 is not None else 0.0
        vwc2 = vwc2 if vwc2 is not None else 0.0

    # Load RH data for the specified track
    RH, sat_num, mean_az = load_rh_data(station, year, track)
    if RH is None:
        print("Error: Could not load RH data for specified track")
        return

    # Create results directory and copy arcs
    arc_results_dir = os.path.join(results_dir, f'{station}_{year}_{doy1:03d}v{doy2:03d}')
    for doy in (doy1, doy2):
        gnssir_cl.gnssir(station, year, doy, savearcs=True)
        copy_arcs(station, year, doy, os.path.join(arc_results_dir, f'{doy:03d}'))

    target_arc = f'sat{sat_num:03d}_L2_G_az{int(mean_az):03d}.txt'
    arc_file1 = os.path.join(arc_results_dir, f'{doy1:03d}', target_arc)
    arc_file2 = os.path.join(arc_results_dir, f'{doy2:03d}', target_arc)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(12, 12), dpi=300)
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.5, 1.5])
    ax_vwc = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # Plot VWC time series
    ax_vwc.plot(doys, vwc_values, 'b-', alpha=0.6, label='VWC')
    ax_vwc.scatter([doy1, doy2], [vwc1, vwc2], c=['r', 'g'], s=100, zorder=5,
                   label='Comparison days')
    ax_vwc.grid(True, linestyle='--', alpha=0.7)
    ax_vwc.set_xlabel('Day of Year', fontsize=12)
    ax_vwc.set_ylabel('VWC', fontsize=12)
    ax_vwc.set_title(f'Volumetric Water Content - {station.upper()} {year}', fontsize=12, pad=10)
    ax_vwc.legend()

    # Lists to store overall min/max values for SNR plots
    all_x_values = []
    all_snr_values = []
    plot_data = []

    # First pass: collect all data and determine ranges
    RH = RH * 100  # Convert to centimeters
    for arc_file, doy, vwc in [(arc_file1, doy1, vwc1), (arc_file2, doy2, vwc2)]:
        if os.path.exists(arc_file):
            data = np.loadtxt(arc_file, skiprows=1)
            elev_angle = data[:, 0]
            snr = data[:, 1]
            time_secs = data[:, 2]

            x, yc, A, phi = fit_sinusoid(elev_angle, snr, RH)

            all_x_values.extend(x)
            all_snr_values.extend(snr)
            all_snr_values.extend(yc)

            plot_data.append({
                'x': x,
                'snr': snr,
                'yc': yc,
                'time_secs': time_secs,
                'A': A,
                'phi': phi,
                'doy': doy,
                'vwc': vwc
            })

    # Determine overall ranges for SNR plots
    x_min, x_max = min(all_x_values), max(all_x_values)
    snr_min, snr_max = min(all_snr_values), max(all_snr_values)

    # Add some padding to the ranges
    snr_range = snr_max - snr_min
    snr_padding = snr_range * 0.05
    x_range = x_max - x_min
    x_padding = x_range * 0.05

    # Second pass: create SNR plots with consistent scales
    for ax, data, color in zip([ax1, ax2], plot_data, ['r', 'g']):
        # Format time range
        time_range = format_time_range(data['time_secs'])

        ax.plot(data['x'], data['snr'], '.', color=color, alpha=0.8, label='SNR Data')
        ax.plot(data['x'], data['yc'], 'k-', linewidth=1.5, alpha=0.8, label='Model Fit')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylabel('SNR (V/V)', fontsize=12)
        ax.set_title(f'Day {data["doy"]}, {year}', fontsize=12, pad=10)
        ax.legend(loc='lower left')

        # Set consistent axis limits
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(snr_min - snr_padding, snr_max + snr_padding)

        # Add text box with satellite and timing info
        info_text = (f'Track: {track}\n'
                     f'Satellite: {sat_num:d} (G)\n'
                     f'Mean Az: {mean_az:.1f}°\n'
                     f'RH: {RH / 100:.3f} m\n'
                     f'VWC: {data["vwc"]:.3f}\n'
                     f'Time: {time_range}\n'
                     f'Amplitude: {data["A"]:.2f}\n'
                     f'Phase: {np.degrees(data["phi"]):.2f}°')

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

    ax2.set_xlabel('sin(elevation angle)', fontsize=12)
    plt.tight_layout()

    # Create a descriptive filename with station, year, days, and satellite info
    fig_filename = f'arc_comparison_{station}_{year}_d{doy1:03d}v{doy2:03d}_sat{sat_num:03d}.png'
    fig_path = os.path.join(arc_results_dir, fig_filename)
    plt.savefig(fig_path, bbox_inches='tight')
    print(f'Analysis and visualization complete! Saved as: {fig_filename}')


if __name__ == '__main__':
    compare_arcs('p038', 2017, 211, 212, track=3)