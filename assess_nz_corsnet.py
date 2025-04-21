#!/usr/bin/env python3
import csv
import subprocess
import os
import shutil
import sys
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # For progress bars

# Configuration
CSV_FILE = "/home/george/Documents/Work/nz_corsnet_assessment/nz-government-gnss-cors.csv"
YEAR = 2024
DAY = 1
ARCHIVE = "nz"
RESULTS_DIR = "aggregate_results"
LOG_LEVEL = "INFO"  # Change to "DEBUG" for verbose output
PARALLEL = False  # Set to True to process stations in parallel
MAX_WORKERS = 4  # Maximum parallel workers when PARALLEL is True

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(f"gnss_processing_{YEAR}_{DAY}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# For capturing subprocess output
if LOG_LEVEL == "DEBUG":
    subprocess_output = None  # Show all output
else:
    subprocess_output = subprocess.DEVNULL  # Hide output


class StationProcessor:
    def __init__(self, station_data):
        self.original_code = station_data['geodetic_code']
        self.station_code = self.original_code.lower()
        self.station_name = station_data['current_mark_name']
        self.network = station_data['network']
        self.success = False
        self.error_msg = None
        self.result_file = None

        # Paths
        self.refl_code = os.environ.get('REFL_CODE', os.path.expanduser('~/Scripts/gnssIR/data/refl_code'))
        self.expected_result_file = os.path.join(
            self.refl_code, str(YEAR), 'results',
            self.station_code, f"{str(DAY).zfill(3)}.txt"
        )

    def _run_command(self, cmd, step_name):
        """Run a command with proper error handling"""
        logger.debug(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess_output,
                stderr=subprocess_output
            )
            return True, None
        except subprocess.CalledProcessError as e:
            return False, f"{step_name} failed: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error during {step_name}: {str(e)}"

    def process(self):
        """Process the station through all steps"""
        logger.info(f"Processing: {self.original_code} ({self.station_name}, {self.network})")

        # Step 1: Create SNR file
        success, error = self._run_command(
            ['rinex2snr', self.station_code, str(YEAR), str(DAY), '-archive', ARCHIVE],
            "SNR creation"
        )
        if not success:
            self.error_msg = error
            logger.warning(f"{self.original_code}: {error}")
            return False

        # Step 2: Set analysis parameters with frequency 20
        success, error = self._run_command(
            ['gnssir_input', self.station_code, '-fr', '20'],
            "Analysis parameter setup with L2C frequency"
        )
        if not success:
            self.error_msg = error
            logger.warning(f"{self.original_code}: {error}")
            return False

        # Step 3: Run GNSS-IR analysis
        success, error = self._run_command(
            ['gnssir', self.station_code, str(YEAR), str(DAY)],
            "GNSS-IR analysis"
        )
        if not success:
            self.error_msg = error
            logger.warning(f"{self.original_code}: {error}")
            return False

        # Step 4: Check and copy results
        if os.path.exists(self.expected_result_file):
            dest_dir = os.path.join(RESULTS_DIR, self.network)
            os.makedirs(dest_dir, exist_ok=True)

            dest_file = os.path.join(dest_dir, f"{self.original_code}_{str(DAY).zfill(3)}.txt")
            shutil.copy2(self.expected_result_file, dest_file)

            self.result_file = dest_file
            self.success = True
            logger.info(f"{self.original_code}: Successfully processed and saved to {dest_file}")
            return True
        else:
            self.error_msg = "No results file generated"
            logger.warning(f"{self.original_code}: No results file found at {self.expected_result_file}")
            return False


def process_all_stations():
    """Process all stations in the CSV file"""
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Read the CSV file
    stations = []
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stations.append(row)

    # Process stations
    results = []
    start_time = datetime.now()

    if PARALLEL:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            processors = [StationProcessor(station) for station in stations]

            # Use tqdm for progress bar
            for processor in tqdm(processors, desc="Processing stations"):
                processor.process()
                results.append(processor)
    else:
        for station in tqdm(stations, desc="Processing stations"):
            processor = StationProcessor(station)
            processor.process()
            results.append(processor)

    # Summarize results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    processing_time = datetime.now() - start_time

    logger.info("\n" + "=" * 60)
    logger.info(f"Processing complete in {processing_time}")
    logger.info(f"Total stations: {len(stations)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info("=" * 60)

    if successful:
        logger.info("\nSuccessful stations:")
        for p in successful:
            logger.info(f"  ✓ {p.original_code} ({p.station_name})")

    if failed:
        logger.info("\nFailed stations:")
        for p in failed:
            logger.info(f"  ✗ {p.original_code} ({p.station_name}): {p.error_msg}")

    logger.info(f"\nResults have been saved to: {RESULTS_DIR}")
    return successful, failed


if __name__ == "__main__":
    process_all_stations()