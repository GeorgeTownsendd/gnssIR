#!/usr/bin/env python3

"""
SegmentProcessor: Process GNSS observation file segments by host.

This module processes segments host by host, ensuring each host's segments
are processed in chronological order with sequential session numbering.
"""

import os
import sys
import json
import subprocess
import logging
from datetime import datetime
import re
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
import glob


class Segment(NamedTuple):
    """Represents a segment to be processed"""
    host: str
    name: str
    start_time: datetime
    end_time: datetime


@dataclass
class SegmentConfig:
    """Configuration for segment processing"""
    teqc_path: str = "/home/george/Scripts/gnssIR/bin/teqc"
    field_test_dir: str = "field_test_1"
    refl_code_dir: str = "/home/george/Scripts/gnssIR/data/refl_code"
    log_level: int = logging.INFO


def setup_logging(name: str = "segment") -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'{name}_processor_{timestamp}.log'
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)
    return logger


class SegmentProcessor:
    """Process GNSS observation file segments host by host"""

    def __init__(self, config: Optional[SegmentConfig] = None):
        self.config = config or SegmentConfig()
        self.logger = setup_logging()

    def parse_teqc_meta(self, filename: Path) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract start and end times from teqc +meta output"""
        try:
            cmd = [self.config.teqc_path, '+meta', str(filename)]
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

    def find_merged_file(self, host_dir: Path, target_start: datetime,
                         target_end: datetime) -> Tuple[Optional[Path], Optional[datetime], Optional[datetime]]:
        """Find the appropriate merged file containing the target time range"""
        merged_files = glob.glob(str(host_dir / "merged_*.obs"))

        for merged_file in merged_files:
            merged_path = Path(merged_file)
            start_time, end_time = self.parse_teqc_meta(merged_path)
            if start_time and end_time:
                if start_time <= target_start and end_time >= target_end:
                    return merged_path, start_time, end_time

        return None, None, None

    def group_segments_by_host(self, segments: List[dict]) -> Dict[str, List[Segment]]:
        """
        Group and sort segments by host.

        Returns a dictionary where:
        - key: host identifier (e.g., 'gnsshost-1')
        - value: list of Segment objects sorted by start time
        """
        host_segments: Dict[str, List[Segment]] = {}

        for seg_dict in segments:
            segment = Segment(
                host=seg_dict['host'],
                name=seg_dict['name'],
                start_time=datetime.fromisoformat(seg_dict['start_time']),
                end_time=datetime.fromisoformat(seg_dict['end_time'])
            )

            if segment.host not in host_segments:
                host_segments[segment.host] = []
            host_segments[segment.host].append(segment)

        # Sort each host's segments by start time
        for host in host_segments:
            host_segments[host].sort(key=lambda x: x.start_time)
            self.logger.info(f"\nSegments for {host} (in chronological order):")
            for seg in host_segments[host]:
                self.logger.info(f"  {seg.name}: {seg.start_time}")

        return host_segments

    def process_host_segment(self, segment: Segment, base_dir: Path, session_num: int) -> bool:
        """Process a single segment for a host with known session number"""
        host_dir = base_dir / segment.host
        if not host_dir.exists():
            self.logger.error(f"Host directory not found: {host_dir}")
            return False

        self.logger.info(f"\nProcessing segment: {segment.name}")
        self.logger.info(f"Host: {segment.host}")
        self.logger.info(f"Time range: {segment.start_time} -> {segment.end_time}")
        self.logger.info(f"Session number: {session_num}")

        # Find appropriate merged file
        merged_file, file_start, file_end = self.find_merged_file(
            host_dir, segment.start_time, segment.end_time
        )
        if not merged_file:
            self.logger.error("No suitable merged file found")
            return False

        self.logger.info(f"Found merged file: {merged_file}")
        self.logger.info(f"File spans: {file_start} -> {file_end}")

        # Create temporary file
        temp_file = host_dir / f"segment_{segment.name}.obs"

        try:
            # Extract segment
            cmd = [
                self.config.teqc_path,
                '+st', segment.start_time.strftime('%Y%m%d%H%M%S'),
                '+e', segment.end_time.strftime('%Y%m%d%H%M%S'),
                str(merged_file)
            ]

            self.logger.info(f"Extracting segment to: {temp_file}")
            with open(temp_file, 'w') as f:
                subprocess.run(cmd, stdout=f, check=True)

            # Verify extraction
            segment_start, segment_end = self.parse_teqc_meta(temp_file)
            if not (segment_start and segment_end):
                raise ValueError("Failed to verify extracted segment")

            # Generate destination information
            host_num = int(segment.host.split('-')[-1])
            host_id = f"g{host_num}"
            doy = segment.start_time.timetuple().tm_yday
            year = str(segment.start_time.year)[-2:]

            # Create destination path
            dest_dir = Path(self.config.refl_code_dir) / str(
                segment.start_time.year) / "rinex" / f"{host_id}s{session_num}"
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Generate RINEX filename
            rinex_name = f"{host_id}s{session_num}{doy:03d}0.{year}o"
            dest_file = dest_dir / rinex_name

            # Move file to destination
            self.logger.info(f"Moving segment to: {dest_file}")
            shutil.copy2(temp_file, dest_file)
            os.remove(temp_file)

            return True

        except Exception as e:
            self.logger.error(f"Error processing segment: {e}")
            if temp_file.exists():
                os.remove(temp_file)
            return False

    def process_directory(self, base_dir: Union[str, Path]) -> bool:
        """Process all segments in directory, host by host"""
        base_dir = Path(base_dir)
        self.logger.info(f"Processing directory: {base_dir}")

        # Load configuration
        config_path = base_dir / 'segments_config.json'
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False

        # Group segments by host and sort by time
        host_segments = self.group_segments_by_host(config['segments'])

        # Process each host's segments in order
        successful = 0
        failed = 0

        for host in sorted(host_segments.keys()):
            self.logger.info(f"\nProcessing host: {host}")
            segments = host_segments[host]

            # Process each segment for this host in chronological order
            for session_num, segment in enumerate(segments, 1):
                if self.process_host_segment(segment, base_dir, session_num):
                    successful += 1
                else:
                    failed += 1

        self.logger.info(f"\nProcessing complete")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")

        return successful > 0


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Process GNSS observation file segments based on configuration'
    )
    parser.add_argument('base_dir', type=str, nargs='?',
                        default=None,
                        help='Base directory containing segment configuration')
    parser.add_argument('-t', '--teqc_path', type=str,
                        help='Path to teqc executable')
    parser.add_argument('-r', '--refl_code_dir', type=str,
                        help='Base directory for organized files')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    config = SegmentConfig(
        teqc_path=args.teqc_path or SegmentConfig.teqc_path,
        field_test_dir=args.base_dir or SegmentConfig.field_test_dir,
        refl_code_dir=args.refl_code_dir or SegmentConfig.refl_code_dir,
        log_level=logging.DEBUG if args.verbose else logging.INFO
    )

    processor = SegmentProcessor(config)
    base_dir = Path(config.field_test_dir)

    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    success = processor.process_directory(base_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()