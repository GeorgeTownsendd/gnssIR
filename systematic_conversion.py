#!/usr/bin/env python3
"""
extract_rinex_segments.py

This script reads a JSON configuration file (segments_config.json) that defines one or more
time segments for extracting GNSS products from a series of compressed RINEX files.
For each segment, the script:
  - Locates all compressed RINEX files (*.crx.gz) in the specified host folder.
  - Decompresses these files into a temporary folder.
  - Computes the time span (end_time – start_time) and converts the start time into teqc’s format.
  - Builds a teqc command that extracts:
       + a RINEX observation file,
       + a GPS RINEX navigation file, and
       + a meteorological file.
    The teqc command uses:
       - "-st" to set the start time (format: YYYY_MM_DD:HH:MM:SS),
       - "-tbin" to set the output time span (e.g. "6h" for 6 hours),
       - optionally "-O.dec" if decimation is desired.
  - Writes the output files (named <segment_name>.obs, <segment_name>.gps, and <segment_name>.met)
    into an output subdirectory within the host folder.

Requirements:
  • teqc must be installed and available at /home/george/Scripts/gnssIR/bin/teqc
  • The input files (compressed RINEX files) must have the extension ".crx.gz".

Usage:
  1. Create a JSON configuration file (see sample below).
  2. Run the script:
         python extract_rinex_segments.py
"""

import json
import subprocess
import datetime
from pathlib import Path
import sys
import gzip
import shutil
import tempfile

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Base directory containing host subdirectories with compressed RINEX files.
BASE_RINEX_DIR = Path("/home/george/Scripts/gnssIR/field_test_1/processed/rinex")
# Subdirectory (inside each host folder) where teqc output will be stored.
OUTPUT_SUBDIR = "extracted_teqc"
# Full path to the teqc executable.
TEQC_PATH = "/home/george/Scripts/gnssIR/bin/teqc"


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def format_time_teqc(dt):
    """
    Convert a datetime object into teqc's expected start time format:
      YYYY_MM_DD:HH:MM:SS
    For example, 2025-02-06T20:00:00 becomes "2025_02_06:20:00:00".
    """
    return dt.strftime("%Y_%m_%d:%H:%M:%S")


def compute_tbin(timespan_seconds):
    """
    Given a time span in seconds, return a string for the -tbin option.
    For example, if the time span is 6 hours (21600 sec), return "6h".
    (If the span is less than one hour, minutes are used, e.g. "30m".)
    """
    if timespan_seconds < 3600:
        minutes = int(timespan_seconds // 60)
        return f"{minutes}m"
    elif timespan_seconds < 86400:
        hours = int(timespan_seconds // 3600)
        return f"{hours}h"
    else:
        days = int(timespan_seconds // 86400)
        return f"{days}d"


def extract_segment_teqc(host, segment_name, start_time, end_time, decimation=None):
    """
    For the given host and segment parameters, use teqc to extract:
      - a RINEX observation file,
      - a GPS navigation file, and
      - a meteorological file.

    The command uses:
      - -st <start_time> (in teqc format: YYYY_MM_DD:HH:MM:SS)
      - -tbin <span> where <span> is computed from (end_time - start_time)
      - optionally, -O.dec <decimation> if a decimation value is provided.

    The three output files are saved in the OUTPUT_SUBDIR inside the host folder.
    """
    host_dir = BASE_RINEX_DIR / host
    if not host_dir.exists():
        print(f"[ERROR] Host directory {host_dir} does not exist.")
        return

    # Find all compressed RINEX files (*.crx.gz) in the host folder.
    rinex_files = sorted(host_dir.glob("*.crx.gz"))
    if not rinex_files:
        print(f"[WARN] No compressed RINEX files found in {host_dir}.")
        return

    # Decompress the .crx.gz files into a temporary directory.
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_dir = Path(tmpdirname)
        decompressed_files = []
        for rfile in rinex_files:
            if rfile.name.endswith('.crx.gz'):
                # Remove the .gz extension so that the file is recognized as .crx.
                base_name = rfile.name[:-3]
                tmp_file = tmp_dir / base_name
                try:
                    with gzip.open(rfile, 'rb') as fin, open(tmp_file, 'wb') as fout:
                        shutil.copyfileobj(fin, fout)
                    decompressed_files.append(tmp_file)
                except Exception as e:
                    print(f"[ERROR] Failed to decompress {rfile}: {e}")
            else:
                decompressed_files.append(rfile)

        # Compute the time span between start and end times.
        timespan_seconds = (end_time - start_time).total_seconds()
        tbin_str = compute_tbin(timespan_seconds)
        # Convert start time into teqc format.
        start_time_str = format_time_teqc(start_time)

        # Define the output file paths.
        output_dir = host_dir / OUTPUT_SUBDIR
        output_dir.mkdir(exist_ok=True)
        obs_file = output_dir / f"{segment_name}.obs"
        nav_file = output_dir / f"{segment_name}.gps"
        met_file = output_dir / f"{segment_name}.met"

        # Build the teqc command.
        # The command will be similar to:
        #   teqc -st <start_time> -tbin <span> [ -O.dec <value> ] +obs <obs_file> +nav <nav_file> +met <met_file> <input_files...>
        teqc_cmd = [
            TEQC_PATH,
            "-st", start_time_str,
            "-tbin", tbin_str
        ]
        if decimation:
            teqc_cmd.extend(["-O.dec", str(decimation)])
        teqc_cmd.extend([
            "+obs", str(obs_file),
            "+nav", str(nav_file),
            "+met", str(met_file)
        ])
        # Append all decompressed input files.
        for df in decompressed_files:
            teqc_cmd.append(str(df))

        # Print the full command for debugging.
        print("Running teqc command:")
        print(" ".join(teqc_cmd))

        # Execute teqc.
        try:
            result = subprocess.run(teqc_cmd, check=True, capture_output=True, text=True)
            print("[INFO] teqc output:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("[ERROR] teqc failed:")
            print(e.stderr)
            return

        print(f"[INFO] Segment '{segment_name}' for host '{host}' extracted:")
        print(f"  Observation file: {obs_file}")
        print(f"  Navigation file: {nav_file}")
        print(f"  Meteorological file: {met_file}")


# -----------------------------------------------------------------------------
# Main Routine
# -----------------------------------------------------------------------------
def main():
    # Load the JSON configuration file.
    config_file = Path("segments_config.json")
    if not config_file.exists():
        print(f"[ERROR] Configuration file '{config_file}' not found.")
        sys.exit(1)
    try:
        with config_file.open("r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load configuration: {e}")
        sys.exit(1)

    segments = config.get("segments", [])
    if not segments:
        print("[ERROR] No segments defined in configuration.")
        sys.exit(1)

    # Process each segment.
    for seg in segments:
        try:
            host = seg["host"]
            segment_name = seg["name"]
            start_time_str = seg["start_time"]
            end_time_str = seg["end_time"]
            # Optionally, a decimation value (in seconds) can be provided.
            decimation = seg.get("decimation", None)
            # Parse the ISO-formatted times into datetime objects.
            start_time = datetime.datetime.fromisoformat(start_time_str)
            end_time = datetime.datetime.fromisoformat(end_time_str)
        except KeyError as e:
            print(f"[ERROR] Missing required key in segment: {e}")
            continue
        except Exception as e:
            print(f"[ERROR] Failed to parse times for segment '{seg}': {e}")
            continue

        print(f"[INFO] Extracting segment '{segment_name}' for host '{host}' from {start_time} to {end_time}")
        extract_segment_teqc(host, segment_name, start_time, end_time, decimation)


if __name__ == "__main__":
    main()
