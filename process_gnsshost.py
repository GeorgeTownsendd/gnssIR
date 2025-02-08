import os
from pathlib import Path
import subprocess
from tqdm import tqdm
from datetime import datetime

# Configure your base directory here
BASE_DIR = "/home/george/Scripts/gnssIR/field_test_1"


def convert_ubx_files(base_dir):
    """
    Convert all UBX files to RINEX format and compress using Hatanaka compression
    """
    base_path = Path(base_dir)
    rinex_dir = base_path / 'processed' / 'rinex'

    # Create rinex directory if it doesn't exist
    rinex_dir.mkdir(parents=True, exist_ok=True)

    # Process each host directory
    for host_dir in sorted(base_path.glob('gnsshost-*')):
        # Create host-specific output directory
        host_rinex_dir = rinex_dir / host_dir.name
        host_rinex_dir.mkdir(exist_ok=True)

        # Get list of UBX files
        ubx_files = list(host_dir.glob('*.ubx'))
        print(f"\nProcessing {host_dir.name}: {len(ubx_files)} UBX files")

        # Convert each UBX file
        for ubx_file in tqdm(ubx_files, desc=f"Converting {host_dir.name}"):
            try:
                # Generate a standard RINEX name (YYYYMMDD_HHMMSS.23o format)
                # Parse the datetime from the original filename
                dt = datetime.strptime(ubx_file.stem, '%Y-%m-%d_%H-%M-%S_GNSS-1')
                rinex_name = f"{dt.strftime('%Y%m%d_%H%M%S')}.obs"
                standard_rinex_name = f"{dt.strftime('%Y%m%d_%H%M%S')}.{dt.strftime('%y')}o"

                # First convert UBX to RINEX
                cmd_convbin = [
                    'convbin',
                    '-r', 'ubx',
                    '-v', '3.04',
                    '-od',  # Include Doppler
                    '-os',  # Include SNR
                    '-hc', f"Data from {host_dir.name}",
                    str(ubx_file),
                    '-d', str(host_rinex_dir)
                ]
                result = subprocess.run(cmd_convbin,
                                        check=True,
                                        capture_output=True,
                                        text=True)

                # Rename the file to standard RINEX naming convention for rnx2crx
                original_rinex = host_rinex_dir / rinex_name
                standard_rinex = host_rinex_dir / standard_rinex_name

                if original_rinex.exists():
                    original_rinex.rename(standard_rinex)

                    # Compress using Hatanaka compression using pipe method
                    compressed_file = host_rinex_dir / (standard_rinex.stem + '.crx')
                    with open(standard_rinex, 'rb') as infile, open(compressed_file, 'wb') as outfile:
                        process = subprocess.run(['rnx2crx', '-'],
                                                 input=infile.read(),
                                                 stdout=outfile,
                                                 stderr=subprocess.PIPE)

                    if process.returncode == 0 and compressed_file.exists():
                        # Remove the intermediate RINEX file
                        standard_rinex.unlink()

                        # Rename the compressed file back to our desired naming convention
                        final_name = host_rinex_dir / (ubx_file.stem + '.crx')
                        compressed_file.rename(final_name)

                        # Optionally gzip the Hatanaka compressed file
                        cmd_gzip = ['gzip', str(final_name)]
                        subprocess.run(cmd_gzip, check=True)
                    else:
                        print(f"\nError in Hatanaka compression for {ubx_file.name}:")
                        print(f"Error output: {process.stderr.decode()}")

            except subprocess.CalledProcessError as e:
                print(f"\nError processing {ubx_file.name}:")
                print(f"Error output: {e.stderr}")
            except Exception as e:
                print(f"\nUnexpected error processing {ubx_file.name}:")
                print(f"Error: {str(e)}")


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    print(f"Working in directory: {BASE_DIR}")
    convert_ubx_files(BASE_DIR)