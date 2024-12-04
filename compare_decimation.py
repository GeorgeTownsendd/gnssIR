#!/usr/bin/env python3

"""
Compare GNSS-IR arc files across different decimation rates.
This script builds on the gnssir_arc_plot.py functionality to visualize
how decimation affects SNR patterns.
"""

from pathlib import Path
import re
import matplotlib.pyplot as plt

# Import functions from original script
from plot_arcs import (
    filter_arcs,
    parse_arc_filename,
    read_arc_file,
    make_plot_directory
)


def compare_decimation_arcs(results_dir, station, year, doy, sat=None, freq=None, const=None,
                            direction='both', include_failed=False, save_plot=True):
    """
    Plot arcs from different decimation rates as vertically stacked subplots.
    Plots detrended SNR against sin(elevation), ordered from lowest to highest sample rate.

    Args:
        results_dir (str): Path to results directory
        station (str): 4-character station ID
        year (int): Year
        doy (int): Day of year
        sat (str, optional): Comma-separated list of satellite numbers
        freq (str, optional): Comma-separated list of frequencies
        const (str, optional): Comma-separated list of constellations
        direction (str, optional): Show 'ascending', 'descending', or 'both' arcs
        include_failed (bool, optional): Include arcs that failed QC
        save_plot (bool, optional): Whether to save the plot to disk

    Returns:
        tuple: (fig, axes) matplotlib figure and list of axes objects
    """
    import numpy as np

    # Find all arc directories for this station/year/doy
    base_pattern = f"{station}_{year}_{doy:03d}_dec"
    results_path = Path(results_dir)

    # Get directories and sort by decimation rate (highest to lowest)
    arc_dirs = []
    for d in results_path.glob(f"{base_pattern}*_arcs"):
        dec_match = re.search(r'dec(\d+)', d.name)
        if dec_match:
            dec_rate = int(dec_match.group(1))
            arc_dirs.append((dec_rate, d))

    # Sort by decimation rate (highest to lowest) and extract directories
    arc_dirs = [d for _, d in sorted(arc_dirs, key=lambda x: x[0], reverse=True)]

    if not arc_dirs:
        print(f"No arc directories found matching pattern: {base_pattern}")
        return None, None

    # Create figure with subplots
    fig, axes = plt.subplots(len(arc_dirs), 1, figsize=(12, 4 * len(arc_dirs)), sharex=True)
    # Ensure axes is always a list even with single subplot
    if len(arc_dirs) == 1:
        axes = [axes]

    # Track decimation rates and SNR ranges for consistent y-axis
    dec_rates = []
    valid_arc_counts = {}
    all_snr_values = []

    # First pass to collect SNR ranges
    for arc_dir in arc_dirs:
        arc_files = list(arc_dir.glob('sat*.txt'))
        if include_failed:
            failqc_dir = arc_dir / 'failQC'
            if failqc_dir.exists():
                arc_files.extend(list(failqc_dir.glob('sat*.txt')))

        arc_files = filter_arcs(arc_files, sat, freq, const, direction)

        for arc_file in arc_files:
            _, dsnr, _ = read_arc_file(arc_file)
            if dsnr is not None:
                all_snr_values.extend(dsnr)

    # Calculate global SNR range if we have values
    if all_snr_values:
        snr_min = min(all_snr_values)
        snr_max = max(all_snr_values)
        y_margin = 0.1 * (snr_max - snr_min)  # 10% margin
        y_limits = (snr_min - y_margin, snr_max + y_margin)
    else:
        y_limits = None

    # Process each decimation directory
    for ax, arc_dir in zip(axes, arc_dirs):
        # Extract decimation rate from directory name
        dec_match = re.search(r'dec(\d+)', arc_dir.name)
        if not dec_match:
            continue
        dec_rate = int(dec_match.group(1))
        dec_rates.append(dec_rate)

        # Get arc files
        arc_files = list(arc_dir.glob('sat*.txt'))
        if include_failed:
            failqc_dir = arc_dir / 'failQC'
            if failqc_dir.exists():
                arc_files.extend(list(failqc_dir.glob('sat*.txt')))

        # Filter files based on input parameters
        arc_files = filter_arcs(arc_files, sat, freq, const, direction)

        if not arc_files:
            print(f'No matching arcs found in {arc_dir}')
            continue

        # Plot arcs
        valid_arc_count = 0
        for arc_file in arc_files:
            sat_num, freq_arc, const_arc, azim = parse_arc_filename(arc_file)
            elev, dsnr, secs = read_arc_file(arc_file)

            if elev is not None:
                valid_arc_count += 1
                sin_elev = np.sin(np.deg2rad(elev))
                label = f'Sat{sat_num} az{azim:03d}'
                ax.plot(sin_elev, dsnr, '.-', label=label, alpha=0.6)

        valid_arc_counts[dec_rate] = valid_arc_count
        print(f'Decimation rate {dec_rate}: {valid_arc_count} valid arcs')

        # Set up subplot styling
        ax.set_ylabel('Detrended SNR\n(volts/volts)')
        ax.grid(True)
        if y_limits:
            ax.set_ylim(y_limits)

        # Add decimation rate as text on right side of subplot
        ax.text(1.02, 0.5, f'Dec {dec_rate}',
                transform=ax.transAxes,
                rotation=270,
                verticalalignment='center')

        # Only show legend if there aren't too many arcs
        if valid_arc_count <= 8:  # Adjusted threshold for subplots
            ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')

    # Set up shared x-axis label and overall title
    axes[-1].set_xlabel('sin(elevation)')

    title = f"Detrended SNR Comparison - {station} ({year}/{doy:03d})"
    if const:
        title += f" [{const}]"
    elif sat:
        title += f" [Sat {sat}]"
    title += f" [{freq or 'All Freq.'}]"

    fig.suptitle(title, y=1.02)

    # Adjust layout
    plt.tight_layout()

    if save_plot:
        # Generate plot filename
        plot_dir = make_plot_directory(year, station, doy)
        plot_name = f'decimation_comparison_{station}_{year}_{doy:03d}'
        if sat:
            plot_name += f'_sat{sat}'
        elif const:
            plot_name += f'_{const}'
        if freq:
            plot_name += f'_{freq}'
        if direction != 'both':
            plot_name += f'_{direction}'
        plot_name += '.png'

        # Save plot
        plot_path = plot_dir / plot_name
        plt.savefig(plot_path, bbox_inches='tight')
        print(f'Plot saved to: {plot_path}')

    return fig, axes


if __name__ == '__main__':
    import numpy as np

    # Example usage
    station = 'mchl'
    year = 2024
    doy = 1

    # Compare specific satellite
    fig1, ax1 = compare_decimation_arcs(
        results_dir='results',
        station=station,
        year=year,
        doy=doy,
        sat='12'
    )
    plt.show()
