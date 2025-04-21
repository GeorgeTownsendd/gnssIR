#!/usr/bin/env python3

"""
RinexProcessor: Process RINEX and UBX GNSS observation files.

This module provides functionality to:
- Convert UBX files to RINEX observation (OBS) format
- Process and merge RINEX observation files
- Handle file overlaps and gaps
- Generate merged observation segments
"""

import os
import sys
import glob
import subprocess
import logging
from datetime import datetime, timedelta
import re
from pathlib import Path
import tempfile
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union


@dataclass
class RinexConfig:
    """Configuration for RINEX processing"""
    teqc_path: str = "/home/george/Scripts/gnssIR/bin/teqc"
    gap_threshold: timedelta = timedelta(minutes=1)
    log_level: int = logging.INFO


def setup_logging(name: str = "rinex") -> logging.Logger:
    """Setup logging configuration with both file and console output"""
    logger = logging.getLogger(name)

    if not logger.handlers:  # Only add handlers if none exist
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name}_processor_{timestamp}.log'

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)

    return logger


class RinexProcessor:
    """Process RINEX and UBX GNSS observation files"""

    def __init__(self, config: Optional[RinexConfig] = None):
        """Initialize processor with optional configuration"""
        self.config = config or RinexConfig()
        self.logger = setup_logging()
        self._temp_dir = None
        self.file_times = {}

    def convert_ubx_to_obs(self, ubx_files: List[str]) -> List[str]:
        """
        Convert UBX files to OBS format using convbin.

        Args:
            ubx_files: List of UBX file paths to convert

        Returns:
            List of converted OBS file paths
        """
        self.logger.info(f"Found {len(ubx_files)} UBX files to convert")
        converted_files = []

        for ubx_file in sorted(ubx_files):
            base_name = Path(ubx_file).stem
            output_file = os.path.join(self._temp_dir, f"{base_name}.obs")

            cmd = ['convbin', '-od', '-os', '-v', '2.11', '-r', 'ubx',
                   '-ro', '"-TADJ=1.0"',
                   ubx_file, '-o', output_file]
            self.logger.info(f"Converting {ubx_file} to {output_file}")

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                self.logger.debug(f"convbin output:\n{result.stdout}")
                if result.stderr:
                    self.logger.warning(f"convbin warnings:\n{result.stderr}")
                converted_files.append(output_file)
                self.logger.info(f"Successfully converted {ubx_file}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error converting {ubx_file}: {e}")
                self.logger.error(f"convbin error output:\n{e.stderr}")
                continue

        return converted_files

    def parse_teqc_meta(self, filename: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Extract start and end times from teqc +meta output.

        Args:
            filename: Path to RINEX file

        Returns:
            Tuple of (start_time, end_time) or (None, None) if parsing fails
        """
        try:
            cmd = ['/home/george/Scripts/gnssIR/bin/teqc', '+meta', filename]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            start_match = re.search(
                r'start date & time:\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})',
                result.stdout
            )
            end_match = re.search(
                r'final date & time:\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})',
                result.stdout
            )

            if start_match and end_match:
                start_time = datetime.strptime(start_match.group(1), '%Y-%m-%d %H:%M:%S.%f')
                end_time = datetime.strptime(end_match.group(1), '%Y-%m-%d %H:%M:%S.%f')
                return start_time, end_time

            return None, None
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running teqc on {filename}: {e}")
            return None, None

    def find_continuous_segments(self, file_times: Dict[str, Tuple[datetime, datetime]]) -> List[List[str]]:
        """
        Group files into continuous segments based on time gaps.

        Args:
            file_times: Dictionary mapping filenames to (start_time, end_time) tuples

        Returns:
            List of lists, where each inner list contains filenames for a continuous segment
        """
        segments = []
        current_segment = []

        files = sorted(file_times.keys())
        if not files:
            return segments

        current_segment.append(files[0])

        for i in range(len(files) - 1):
            current_file = files[i]
            next_file = files[i + 1]

            current_end = file_times[current_file][1]
            next_start = file_times[next_file][0]

            gap = next_start - current_end

            if gap > self.config.gap_threshold:
                segments.append(current_segment)
                current_segment = []
                self.logger.info(f"Found gap of {gap} between:")
                self.logger.info(f"  {current_file} ending at {current_end}")
                self.logger.info(f"  {next_file} starting at {next_start}")

            current_segment.append(next_file)

        if current_segment:
            segments.append(current_segment)

        return segments

    def process_file(self, input_file: str, file_times: Dict[str, Tuple[datetime, datetime]],
                     prev_file: Optional[str] = None, next_file: Optional[str] = None) -> Optional[str]:
        """
        Process a single RINEX file, handling overlaps with adjacent files.

        Args:
            input_file: Path to input RINEX file
            file_times: Dictionary of file time ranges
            prev_file: Previous file in sequence (optional)
            next_file: Next file in sequence (optional)

        Returns:
            Path to processed file or None if processing fails
        """
        actual_start, actual_end = file_times[input_file]
        self.logger.info(f"Processing: {input_file}")
        self.logger.info(f"Original time range: {actual_start} -> {actual_end}")

        start_time = actual_start
        end_time = actual_end

        # Handle overlap with previous file
        if prev_file:
            prev_end = file_times[prev_file][1]
            if start_time <= prev_end:
                overlap = prev_end - start_time
                midpoint = start_time + (overlap / 2)
                start_time = midpoint
                self.logger.info(f"Overlap with previous file: {overlap}")
                self.logger.info(f"Using midpoint: {start_time}")

        # Handle overlap with next file
        if next_file:
            next_start = file_times[next_file][0]
            if end_time >= next_start:
                overlap = end_time - next_start
                midpoint = next_start + (overlap / 2)
                end_time = midpoint
                self.logger.info(f"Overlap with next file: {overlap}")
                self.logger.info(f"Using midpoint: {end_time}")

        output_file = os.path.join(self._temp_dir, f"processed_{os.path.basename(input_file)}")
        start_str = start_time.strftime('%Y%m%d%H%M%S.%f')[:15]
        end_str = end_time.strftime('%Y%m%d%H%M%S.%f')[:15]

        cmd = [self.config.teqc_path, '+st', start_str, '+e', end_str, input_file]
        self.logger.info(f"Trimming to: {start_time} -> {end_time}")

        try:
            with open(output_file, 'w') as f:
                subprocess.run(cmd, stdout=f, check=True)
            self.logger.info(f"Successfully processed {input_file} -> {output_file}")

            # Verify processed file
            proc_start, proc_end = self.parse_teqc_meta(output_file)
            if proc_start and proc_end:
                self.logger.info(f"Processed file spans: {proc_start} -> {proc_end}")

            return output_file
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error processing {input_file}: {e}")
            return None

    def merge_segment(self, processed_files: List[str], segment_num: int, output_dir: str) -> bool:
        """
        Merge a segment of processed files into a single file.
        """
        if not processed_files:
            return False

        # Get start time from first file and end time from last file
        first_start, _ = self.parse_teqc_meta(processed_files[0])
        _, last_end = self.parse_teqc_meta(processed_files[-1])

        # Generate filename with time range
        filename_start = first_start.strftime('%Y%m%d_%H%M')
        filename_end = last_end.strftime('%Y%m%d_%H%M')
        output_file = os.path.join(output_dir, f"obs_{filename_start}_to_{filename_end}.obs")

        self.logger.info(f"Merging segment {segment_num} to {output_file}")
        self.logger.info(f"Files to merge: {' '.join(processed_files)}")

        cmd = [self.config.teqc_path, '+rec_mnm'] + processed_files
        try:
            with open(output_file, 'w') as f:
                subprocess.run(cmd, stdout=f, check=True)

            # Verify merged file
            start, end = self.parse_teqc_meta(output_file)
            self.logger.info(f"Successfully created {output_file}")
            self.logger.info(f"Merged file spans: {start} -> {end}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error merging segment {segment_num}: {e}")
            return False

    def process_directory(self, input_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> bool:
        """
        Process all observation files in a directory.

        Args:
            input_dir: Directory containing input files
            output_dir: Directory for output files (defaults to input_dir)

        Returns:
            True if processing successful, False otherwise
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir

        self.logger.info(f"Processing directory: {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")

        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            self._temp_dir = temp_dir
            self.logger.info(f"Using temporary directory: {temp_dir}")

            # Change to input directory
            original_dir = os.getcwd()
            os.chdir(input_dir)

            try:
                # First check for .obs files
                obs_files = sorted(glob.glob('*.obs'))
                obs_files = [f for f in obs_files
                             if not f.startswith(('processed_', 'trimmed_', 'merged'))]

                # If no .obs files found, look for .ubx files
                if not obs_files:
                    ubx_files = glob.glob('*.ubx')
                    if ubx_files:
                        self.logger.info("No OBS files found, converting UBX files")
                        obs_files = self.convert_ubx_to_obs(ubx_files)
                    else:
                        self.logger.error("No OBS or UBX files found in directory")
                        return False

                if not obs_files:
                    self.logger.error("No valid observation files found after conversion")
                    return False

                self.logger.info(f"Found {len(obs_files)} observation files to process")

                # Get file times
                self.file_times = {}
                for f in obs_files:
                    print(str(f))
                    start, end = self.parse_teqc_meta(f)
                    if start and end:
                        self.file_times[f] = (start, end)
                        self.logger.info(f"File {f} spans: {start} -> {end}")

                # Find continuous segments
                segments = self.find_continuous_segments(self.file_times)
                self.logger.info(f"Found {len(segments)} continuous segments")

                # Process each segment
                for segment_num, segment_files in enumerate(segments, 1):
                    self.logger.info(f"Processing segment {segment_num} ({len(segment_files)} files)")
                    processed_files = []

                    for i, obs_file in enumerate(segment_files):
                        prev_file = segment_files[i - 1] if i > 0 else None
                        next_file = segment_files[i + 1] if i < len(segment_files) - 1 else None

                        processed_file = self.process_file(obs_file, self.file_times,
                                                           prev_file, next_file)
                        if processed_file:
                            processed_files.append(processed_file)

                    if processed_files:
                        self.merge_segment(processed_files, segment_num, output_dir)

                return True

            finally:
                os.chdir(original_dir)
                self._temp_dir = None


def main():
    """Command-line interface for standalone use"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Process RINEX and UBX GNSS observation files'
    )
    parser.add_argument('input_dir', type=str,
                        help='Directory containing input files')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Directory for output files (defaults to input directory)')
    parser.add_argument('-t', '--teqc_path', type=str,
                        help='Path to teqc executable (defaults to "teqc" in PATH)')
    parser.add_argument('-g', '--gap_threshold', type=int, default=1,
                        help='Gap threshold in minutes (default: 1)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Create configuration
    config = RinexConfig(
        teqc_path=args.teqc_path or "teqc",
        gap_threshold=timedelta(minutes=args.gap_threshold),
        log_level=logging.DEBUG if args.verbose else logging.INFO
    )

    # Initialize processor and run
    processor = RinexProcessor(config)
    success = processor.process_directory(args.input_dir, args.output_dir)

    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()