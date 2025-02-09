#!/usr/bin/env python3
"""
Compare detrended SNR arcs from GNSS-IR arc files across multiple stations
to test antenna performance.

This script reads arc files (created by gnssir with the -savearcs option)
for a given year and day-of-year (DOY) and compares arcs for specific satellites
and/or frequencies across the provided stations.
"""

import os
import sys
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Base data path (adjust if needed)
BASE_PATH = Path("/home/george/Scripts/gnssIR/data/refl_code/2025")


def parse_arc_filename(filename):
    """
    Parse metadata from an arc filename (e.g., sat006_L1_G_az097.txt).

    Returns:
        sat_num (int): Satellite number.
        freq (str): Frequency (e.g., L1).
        const (str): Constellation code (G, R, E, C).
        azim (int): Azimuth (degrees).
    """
    parts = filename.stem.split('_')
    try:
        sat_num = int(parts[0][3:])  # Remove 'sat'
        freq = parts[1]
        const = parts[2]
        azim = int(parts[3][2:])  # Remove 'az'
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None, None, None, None
    return sat_num, freq, const, azim


def read_arc_file(file_path, decimate=None):
    """
    Read elevation angles, dSNR values, and seconds from an arc file.

    Args:
        file_path (Path): Path to the arc file.
        decimate (int): If provided, take every Nth measurement.

    Returns:
        elev (np.array): Elevation angles.
        dsnr (np.array): Detrended SNR values.
        secs (np.array): Seconds.
    """
    try:
        data = np.loadtxt(file_path, skiprows=2)
        if data.size == 0:
            print(f"Warning: Empty file {file_path}")
            return None, None, None
        if decimate and decimate > 1:
            data = data[::decimate]
        return data[:, 0], data[:, 1], data[:, 2]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None


def determine_arc_direction(elev):
    """
    Determine if an arc is ascending or descending based on elevation data.
    """
    if elev is None or len(elev) < 2:
        return 'unknown'
    return 'ascending' if elev[1] > elev[0] else 'descending'


def get_arc_directory(year, station, doy):
    """
    Get the directory containing arc files for the given station and DOY.
    """
    arc_dir = BASE_PATH / "arcs" / station / f"{int(doy):03d}"
    if not arc_dir.exists():
        print(f"Error: Arc directory not found: {arc_dir}")
        sys.exit(1)
    return arc_dir


def filter_arc_file(file_path, sat_filter, freq_filter, const_filter, direction_filter, decimate):
    """
    Return True if the file at file_path passes all filters.
    """
    sat_num, freq, const, azim = parse_arc_filename(file_path)
    if sat_num is None:
        return False
    if sat_filter:
        try:
            sat_nums = [int(s) for s in sat_filter.split(',')]
        except Exception as e:
            print("Error parsing satellite filter:", e)
            sys.exit(1)
        if sat_num not in sat_nums:
            return False
    if freq_filter:
        freqs = [f.strip() for f in freq_filter.split(',')]
        if freq not in freqs:
            return False
    if const_filter:
        consts = [c.strip().upper() for c in const_filter.split(',')]
        if const.upper() not in consts:
            return False
    if direction_filter != 'both':
        elev, dsnr, secs = read_arc_file(file_path, decimate=decimate)
        if elev is None:
            return False
        arc_direction = determine_arc_direction(elev)
        if arc_direction != direction_filter:
            return False
    return True


def main(stations, year, doy, sat, freq, const, direction, include_failed, decimate):
    """
    Compare arcs across the provided stations and plot them.
    """
    # Group arc files by (sat_num, frequency, constellation)
    grouped_arcs = {}
    for station in stations:
        arc_dir = get_arc_directory(year, station, doy)
        print(f"Scanning directory: {arc_dir}")
        files = list(arc_dir.glob("sat*.txt"))
        if include_failed:
            failqc_dir = arc_dir / "failQC"
            if failqc_dir.exists():
                files.extend(list(failqc_dir.glob("sat*.txt")))
        print(f"Found {len(files)} files in station {station}")
        for file in files:
            if not filter_arc_file(file, sat, freq, const, direction, decimate):
                continue
            sat_num, file_freq, file_const, azim = parse_arc_filename(file)
            if sat_num is None:
                continue
            group_key = (sat_num, file_freq, file_const)
            grouped_arcs.setdefault(group_key, []).append((station, file))

    if not grouped_arcs:
        print("No arc files found matching specified criteria.")
        sys.exit(0)

    print(f"Found {len(grouped_arcs)} groups of arcs.")
    # Determine subplot layout based on number of groups
    n_groups = len(grouped_arcs)
    if n_groups == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = [ax]
    else:
        ncols = math.ceil(math.sqrt(n_groups))
        nrows = math.ceil(n_groups / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 12))
        axes = axes.flatten()

    # Plot each group (each group represents a specific satellite/frequency/constellation)
    for ax, (group_key, arc_list) in zip(axes, grouped_arcs.items()):
        sat_num, file_freq, file_const = group_key
        for station, file in arc_list:
            elev, dsnr, secs = read_arc_file(file, decimate=decimate)
            if elev is None:
                continue
            # Plot detrended SNR vs. elevation angle.
            ax.plot(elev, dsnr, label=f'{station}', alpha=1.0)
        title = f"Sat {sat_num} {file_freq} {file_const}"
        ax.set_title(title)
        ax.set_xlabel("Elevation Angle (deg)")
        ax.set_ylabel("Detrended SNR (volts/volts)")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    # Note: plt.show() is a blocking call. The script will wait here until you close the plot window.
    print("Displaying plot... close the plot window to finish.")
    plt.show()
    print("Plot closed.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare arcs from GNSS-IR arc files to test antenna performance."
    )
    parser.add_argument('stations', help='Comma-separated list of station IDs to compare')
    parser.add_argument('year', type=int, help='Year')
    parser.add_argument('doy', type=int, help='Day of year')
    parser.add_argument('-sat', help='Comma-separated list of satellite numbers to include')
    parser.add_argument('-freq', help='Comma-separated list of frequencies (e.g., L1,L2) to include')
    parser.add_argument('-const', help='Comma-separated list of constellations (G,R,E,C) to include')
    parser.add_argument('-direction', choices=['ascending', 'descending', 'both'],
                        default='both', help='Show only ascending, descending, or both arcs')
    parser.add_argument('-include_failed', action='store_true',
                        help='Include arcs that failed QC')
    parser.add_argument('-decimate', type=int,
                        help='Take every Nth measurement (e.g., 5 means use every 5th point)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    station_list = [s.strip() for s in args.stations.split(',')]
    main(station_list, args.year, args.doy, args.sat, args.freq, args.const,
         args.direction, args.include_failed, args.decimate)
