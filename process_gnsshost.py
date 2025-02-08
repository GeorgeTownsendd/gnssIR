import os
from pathlib import Path
import subprocess
from tqdm import tqdm

# Configure your base directory here
BASE_DIR = "/home/george/Scripts/gnssIR/field_test_1"


def convert_ubx_files(base_dir):
    """
    Convert all UBX files to RINEX format
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
                cmd = [
                    'convbin',
                    '-r', 'ubx',
                    '-v', '3.04',
                    '-od',  # Include Doppler
                    '-os',  # Include SNR
                    '-hc', f"Data from {host_dir.name}",
                    str(ubx_file),
                    '-d', str(host_rinex_dir)
                ]
                result = subprocess.run(cmd,
                                        check=True,
                                        capture_output=True,
                                        text=True)

            except subprocess.CalledProcessError as e:
                print(f"\nError converting {ubx_file.name}:")
                print(f"Error output: {e.stderr}")


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    print(f"Working in directory: {BASE_DIR}")
    convert_ubx_files(BASE_DIR)