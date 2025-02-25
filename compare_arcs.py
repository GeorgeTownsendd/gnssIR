import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
BASE_PATH = Path("/home/george/Scripts/gnssIR/data/refl_code/2025/arcs")
AZIMUTH_TOLERANCE = 20
STATIONS = ['g1s1', 'g2s1']  # Current active stations
DECIMATE = 15

# Antenna mapping for better labels
ANTENNA_MAP = {
    "g1s1": "Patch (V)",
    "g1s2": "Blue Cable",
    "g2s1": "Patch (H)",
    "g2s2": "Red Cable",
    "g3s1": "GPS500 (H)",
    "g3s2": "GPS500 (V)",
    "g4s1": "GPS500 (V)"
}

# Host mapping if needed
HOST_MAP = {
    "g1s1": "gnsshost-1",
    "g1s2": "gnsshost-1",
    "g2s1": "gnsshost-2",
    "g2s2": "gnsshost-2",
    "g3s1": "gnsshost-3",
    "g3s2": "gnsshost-3",
    "g4s1": "gnsshost-4"
}


def get_station_label(station_id):
    """Creates a descriptive label combining host and antenna information."""
    antenna = ANTENNA_MAP.get(station_id, "Unknown Antenna")
    return f"{station_id}: {antenna}"


def parse_arc_filename(filename):
    """Extracts satellite number, frequency, constellation, and azimuth from filename."""
    parts = filename.stem.split('_')
    try:
        return int(parts[0][3:]), parts[1], parts[2], int(parts[3][2:])
    except (IndexError, ValueError):
        return None, None, None, None


def read_arc_file(file_path):
    """Reads elevation, dSNR, and time from an arc file, with decimation."""
    try:
        data = np.loadtxt(file_path, skiprows=2)
        if data.size:
            return data[::DECIMATE, 0], data[::DECIMATE, 1], data[::DECIMATE, 2]
        else:
            return None, None, None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None, None


def find_matching_arcs(target_file):
    """Finds matching arcs, ensuring correct station association."""
    target_file = Path(target_file)
    sat, freq, const, azim = parse_arc_filename(target_file)
    if sat is None:
        print("Error: Invalid target file format.")
        sys.exit(1)

    matching_files = {}
    for station in STATIONS:
        matching_files[station] = None

    for station in STATIONS:
        potential_files = list(BASE_PATH.glob(f"{station}/**/sat{sat:03d}_{freq}_{const}_az*.txt")) + \
                          list(BASE_PATH.glob(f"{station}/**/failQC/sat{sat:03d}_{freq}_{const}_az*.txt"))

        best_match = None
        best_diff = AZIMUTH_TOLERANCE + 1
        for file in potential_files:
            _, _, _, file_azim = parse_arc_filename(file)
            if file_azim is not None:
                diff = abs(file_azim - azim)
                if diff <= AZIMUTH_TOLERANCE and diff < best_diff:
                    best_match = file
                    best_diff = diff

        matching_files[station] = best_match

    return matching_files


def plot_arcs(matching_files, target_file):
    """Plots detrended SNR arcs against sin(elevation), using descriptive labels."""
    plt.figure(figsize=(10, 6))

    for station, file in matching_files.items():
        if file is not None:
            elev, dsnr, _ = read_arc_file(file)
            if elev is not None and dsnr is not None:
                sin_elev = np.sin(np.radians(elev))
                label = get_station_label(station)
                plt.plot(sin_elev, dsnr, label=label)
            else:
                print(f"Error reading data from {file}")
        else:
            print(f"No matching file found for {station}")

    plt.xlabel("sin(Elevation Angle)")
    plt.ylabel("Detrended SNR (volts/volts)")
    title_suffix = Path(target_file).name
    plt.title(f"Matching GNSS-IR Arcs - {title_suffix}")
    plt.legend(loc='upper left')  # Moved legend outside
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to accommodate legend
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_arcs.py <arc_file>")
        sys.exit(1)

    target_file = sys.argv[1]
    matching_files = find_matching_arcs(target_file)
    if not matching_files:
        print("No matching arcs found.")
        sys.exit(0)

    plot_arcs(matching_files, target_file)


if __name__ == '__main__':
    main()