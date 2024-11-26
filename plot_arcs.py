#!/usr/bin/env python3

"""
Plot detrended SNR (signal-to-noise ratio) arcs from GNSS-IR arc files.

This utility reads arc files created by gnssir with the -savearcs option
and creates plots showing detrended SNR vs elevation angle.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def validate_satellite_number(sat_num):
    """
    Validate if a satellite number corresponds to a known GNSS constellation.
    Returns (is_valid, constellation_name) tuple.
    """
    if 1 <= sat_num <= 32:
        return True, "GPS"
    elif 101 <= sat_num <= 132:
        return True, "GLONASS"
    elif 201 <= sat_num <= 236:
        return True, "Galileo"
    elif 301 <= sat_num <= 332:
        return True, "BeiDou"
    return False, None


def validate_constellation(const):
    """Validate if a constellation code is valid."""
    valid_consts = {'G': 'GPS', 'R': 'GLONASS', 'E': 'Galileo', 'C': 'BeiDou'}
    return const in valid_consts, valid_consts.get(const)


def validate_inputs(sat=None, const=None):
    """Validate satellite numbers and constellation codes."""
    if sat and const:
        print('Error: Specify either --sat or --const, but not both.')
        print('Satellite numbers already encode constellation information:')
        print('  GPS: 1-32')
        print('  GLONASS: 101-132')
        print('  Galileo: 201-236')
        print('  BeiDou: 301-332')
        sys.exit(1)

    if sat:
        try:
            sat_nums = [int(s) for s in sat.split(',')]
        except ValueError:
            print(f'Error: Invalid satellite number format. Must be comma-separated integers.')
            sys.exit(1)

        invalid_sats = []
        for sat_num in sat_nums:
            is_valid, const_name = validate_satellite_number(sat_num)
            if not is_valid:
                invalid_sats.append(sat_num)

        if invalid_sats:
            print('Error: Invalid satellite number(s):')
            print('  Invalid:', ', '.join(str(s) for s in invalid_sats))
            print('Valid ranges are:')
            print('  GPS: 1-32')
            print('  GLONASS: 101-132')
            print('  Galileo: 201-236')
            print('  BeiDou: 301-332')
            sys.exit(1)

    if const:
        const_codes = const.split(',')
        invalid_consts = []
        for code in const_codes:
            is_valid, const_name = validate_constellation(code)
            if not is_valid:
                invalid_consts.append(code)

        if invalid_consts:
            print('Error: Invalid constellation code(s):')
            print('  Invalid:', ', '.join(invalid_consts))
            print('Valid codes are:')
            print('  G: GPS')
            print('  R: GLONASS')
            print('  E: Galileo')
            print('  C: BeiDou')
            sys.exit(1)


def filter_arcs(arc_files, sat=None, freq=None, const=None, direction='both'):
    """Filter arc files based on satellite, frequency, constellation, and direction."""
    validate_inputs(sat, const)

    filtered_files = arc_files

    if sat:
        sat_nums = [int(s) for s in sat.split(',')]
        filtered_files = [f for f in filtered_files
                          if parse_arc_filename(f)[0] in sat_nums]

    if freq:
        freqs = freq.split(',')
        filtered_files = [f for f in filtered_files
                          if parse_arc_filename(f)[1] in freqs]

    if const:
        consts = const.split(',')
        filtered_files = [f for f in filtered_files
                          if parse_arc_filename(f)[2] in consts]

    if direction != 'both':
        filtered_files = [f for f in filtered_files
                          if determine_arc_direction(read_arc_file(f)[0]) == direction]

    return filtered_files


def check_environ_variables():
    """Check if REFL_CODE environment variable exists."""
    refl_code = os.environ.get('REFL_CODE')
    if refl_code is None:
        print('Error: REFL_CODE environment variable must be set.')
        sys.exit(1)
    return refl_code


def make_plot_directory(year, station, doy):
    """Create and return plot directory path following gnssir conventions."""
    refl_code = check_environ_variables()
    plot_dir = Path(refl_code) / str(year) / 'plots' / station / f'{doy:03d}'
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def get_arc_directory(year, station, doy, include_failed=False):
    """Get directory containing arc files for given station and date."""
    refl_code = check_environ_variables()
    arc_dir = Path(refl_code) / str(year) / 'arcs' / station / f'{doy:03d}'

    if not arc_dir.exists():
        print(f'Error: Arc directory not found: {arc_dir}')
        sys.exit(1)

    return arc_dir


def get_constellation_from_sat(sat_num):
    """Determine constellation based on satellite number."""
    if 1 <= sat_num <= 32:
        return 'G'  # GPS
    elif 101 <= sat_num <= 132:
        return 'R'  # GLONASS
    elif 201 <= sat_num <= 236:
        return 'E'  # Galileo
    elif 301 <= sat_num <= 332:
        return 'C'  # BeiDou
    else:
        return 'Unknown'


def parse_arc_filename(filename):
    """
    Parse metadata from arc filename (e.g., sat006_L1_G_az097.txt).

    Satellite numbers encode constellation:
    GPS (G): 1-32
    GLONASS (R): 101-132
    Galileo (E): 201-236
    BeiDou (C): 301-332
    """
    parts = filename.stem.split('_')
    sat_num = int(parts[0][3:])  # Remove 'sat' prefix
    freq = parts[1]
    const = parts[2]  # G, R, E, C
    azim = int(parts[3][2:])  # Remove 'az' prefix

    return sat_num, freq, const, azim


def read_arc_file(file_path):
    """Read elevation angles, dSNR values, and seconds from arc file."""
    try:
        data = np.loadtxt(file_path, skiprows=2)
        if data.size == 0:
            print(f'Warning: Empty file {file_path}')
            return None, None, None
        return data[:, 0], data[:, 1], data[:, 2]  # elev, dSNR, seconds
    except Exception as e:
        print(f'Error reading {file_path}: {e}')
        return None, None, None


def determine_arc_direction(elev):
    """Determine if arc is ascending or descending based on elevation angles."""
    if elev is None or len(elev) < 2:
        return 'unknown'
    return 'ascending' if elev[1] > elev[0] else 'descending'


def plot_arc(ax, elev, dsnr, sat_num, freq, const, azim):
    """
    Plot a single arc with direction in legend.
    """
    direction = determine_arc_direction(elev)
    label = f'Sat {sat_num} {freq} {const} az={azim}° ({direction})'
    ax.plot(elev, dsnr, '.-', label=label)


def main(station, year, doy, sat=None, freq=None, const=None,
         direction='both', include_failed=False):
    """Plot detrended SNR arcs for specified parameters."""
    # Get directories
    arc_dir = get_arc_directory(year, station, doy, include_failed)
    plot_dir = make_plot_directory(year, station, doy)

    # Get list of arc files
    arc_files = list(arc_dir.glob('sat*.txt'))
    if include_failed:
        failqc_dir = arc_dir / 'failQC'
        if failqc_dir.exists():
            arc_files.extend(list(failqc_dir.glob('sat*.txt')))

    if not arc_files:
        print(f'No arc files found for {station} {year} {doy}')
        return

    # If specific satellites were requested, check which ones have data
    if sat:
        requested_sats = [int(s) for s in sat.split(',')]
        found_sats = set()
        for arc_file in arc_files:
            sat_num = parse_arc_filename(arc_file)[0]
            if sat_num in requested_sats:
                found_sats.add(sat_num)

        missing_sats = set(requested_sats) - found_sats
        if missing_sats:
            print(f'Warning: No data found for satellite(s): {", ".join(str(s) for s in sorted(missing_sats))}')

    # Filter files based on input parameters
    arc_files = filter_arcs(arc_files, sat, freq, const, direction)

    if not arc_files:
        print(f'No arcs found matching specified criteria')
        return

    # Get unique frequencies from filtered files
    freqs_found = sorted(set(parse_arc_filename(f)[1] for f in arc_files))
    freq_str = ','.join(freqs_found)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count number of valid arcs
    valid_arc_count = 0
    for arc_file in arc_files:
        elev, dsnr, secs = read_arc_file(arc_file)
        if elev is not None:
            valid_arc_count += 1

    # Plot individual arcs
    for arc_file in arc_files:
        sat_num, freq_arc, const_arc, azim = parse_arc_filename(arc_file)
        elev, dsnr, secs = read_arc_file(arc_file)

        if elev is not None:
            plot_arc(ax, elev, dsnr, sat_num, freq_arc, const_arc, azim)

    # Set up plot styling and title
    title = f"GNSS-IR SNR Arcs - {station} ({year}/{doy:03d})"
    if const:
        title += f" [{const}]"
    elif sat:
        title += f" [Sat {sat}]"
    title += f" [{freq or 'All Freq.'}]"

    ax.set_title(title)
    ax.set_xlabel('Elevation Angle (degrees)')
    ax.set_ylabel('Detrended SNR (volts/volts)')
    ax.grid(True)

    if valid_arc_count <= 10:
        ax.legend()

    # Generate plot filename
    plot_name = f'arcs_{station}_{year}_{doy:03d}'
    if sat:
        plot_name += f'_sat{sat}'
    elif const:
        plot_name += f'_{const}'
    if freq:
        plot_name += f'_{freq}'
    if direction != 'both':
        plot_name += f'_{direction}'
    plot_name += '.png'

    # Save and display plot
    plot_path = plot_dir / plot_name
    plt.savefig(plot_path)
    print(f'Plot saved to: {plot_path}')
    print(f'Number of arcs plotted: {valid_arc_count}')
    plt.show()


def parse_arguments():
    """Parse command line arguments for arc plotting utility."""
    parser = argparse.ArgumentParser(
        description='Plot detrended SNR vs elevation angle from GNSS-IR arc files.'
    )
    parser.add_argument('station', help='4-character station ID')
    parser.add_argument('year', type=int, help='Year')
    parser.add_argument('doy', type=int, help='Day of year')
    parser.add_argument('-sat', help='Comma-separated list of satellite numbers')
    parser.add_argument('-freq', help='Comma-separated list of frequencies (e.g., L1,L2)')
    parser.add_argument('-const', help='Comma-separated list of constellations (G,R,E,C)')
    parser.add_argument('-direction', choices=['ascending', 'descending', 'both'],
                      default='both', help='Show only ascending, descending, or both arcs')
    parser.add_argument('-include_failed', action='store_true',
                      help='Include arcs that failed QC')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args.station, args.year, args.doy, args.sat, args.freq,
         args.const, args.direction, args.include_failed)