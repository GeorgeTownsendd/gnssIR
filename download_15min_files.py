#!/usr/bin/env python3
import requests
from datetime import datetime
import os
from pathlib import Path
import subprocess


def download_rinex_files(station, date, work_dir):
    """Download 15-minute RINEX files if they don't already exist"""
    base_url = "https://data.geonet.org.nz/gnss/rinex1Hz"
    downloaded_files = []

    print(f"Downloading files for {station} on {date.strftime('%Y-%m-%d')}...")
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            timestamp = date.replace(hour=hour, minute=minute)
            filename = f"{station}00NZL_S_{timestamp.strftime('%Y%j%H%M')}_15M_01S_MO.rnx"
            output_file = work_dir / filename

            # Skip if file already downloaded
            if output_file.exists():
                downloaded_files.append(output_file)
                print(f"Found existing file: {filename}")
                continue

            url = f"{base_url}/{date.strftime('%Y/%j')}/{filename}"
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                downloaded_files.append(output_file)
                print(f"Downloaded: {filename}")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {filename}: {e}")

    return downloaded_files


def combine_and_convert(station, date, input_files, output_dir, work_dir):
    """
    Combine RINEX files and convert to RINEX 2.11 format.
    Uses gfzrnx with direct splicing mode.

    Args:
        station (str): Station name (e.g., 'KTIA')
        date (datetime): Date of the files
        input_files (list): List of paths to input RINEX files
        output_dir (str): Directory for output file
        work_dir (str): Working directory containing input files

    Returns:
        Path: Path to the output file
    """
    # Setup output filename in RINEX 2.11 format (e.g., ktia0010.24o)
    year = date.strftime('%Y')
    doy = date.strftime('%j')
    final_file = Path(output_dir) / f"{station.lower()}{doy}0.{year[2:]}o"

    # Change to working directory
    orig_dir = os.getcwd()
    os.chdir('/home/george/Scripts/gnssIR')

    try:
        # Build command list
        cmd = ['bin/gfzrnx', '-finp']
        cmd.extend('rinex_files/tmp/' + f.name for f in sorted(input_files))
        cmd.extend([
            '-fout', str(final_file),
            '-vo', '2',
            '-splice_direct',
            '-kv'
        ])

        print(' '.join(cmd))

        print(f"\nCombining files with command:\n{' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    finally:
        os.chdir(orig_dir)

    return final_file

def cleanup_files(work_dir, input_files):
    """Remove temporary files and directory"""
    for file in input_files:
        file.unlink()
    work_dir.rmdir()


# Main script
if __name__ == "__main__":
    # Set your parameters here
    station = "KTIA"
    date = datetime(2024, 12, 1)  # Year, month, day
    output_dir = "rinex_files"

    # Setup directories
    work_dir = Path(output_dir) / "tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Check if final file already exists
    final_filename = f"{station.lower()}{date.strftime('%j')}0.{date.strftime('%y')}o.gz"
    final_path = Path(output_dir) / final_filename

    if final_path.exists():
        print(f"Final file {final_filename} already exists. Skipping processing.")
        exit(0)

    try:
        # Download files
        downloaded_files = download_rinex_files(station, date, work_dir)
        if not downloaded_files:
            print("No files were downloaded or found.")
            exit(1)

        # Combine and convert files
        final_file = combine_and_convert(station, date, downloaded_files, output_dir, work_dir)

        # Compress the final output
        subprocess.run(['gzip', str(final_file)], check=True)
        print(f"\nCreated combined file: {final_file}.gz")

        # Cleanup
        cleanup_files(work_dir, downloaded_files)
        print("\nProcessing completed successfully!")

    except (subprocess.CalledProcessError, IOError) as e:
        print(f"Error during processing: {e}")
        exit(1)