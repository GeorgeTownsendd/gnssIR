#!/usr/bin/env python3
"""
GNSS Observation File Generator

This script orchestrates the processing of GNSS observation files by:
1. Synchronizing remote GNSS host data
2. Converting raw GNSS data to RINEX format
3. Processing and merging RINEX files for each host
4. Processing host segments in chronological order
5. Organizing files into appropriate directories
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
import importlib.util
from typing import Set, Optional, Dict, Any
from dataclasses import dataclass

# Import the data sync module with the refactored sync_all_hosts function
import pull_gnsshost_data


@dataclass
class GeneratorConfig:
    """Configuration for observation file generation."""
    field_test_dir: Path
    rinex_processor_path: Path
    segment_processor_path: Path
    log_level: int = logging.INFO

    @classmethod
    def from_paths(cls, field_test_dir: str, rinex_processor_path: str,
                   segment_processor_path: str) -> 'GeneratorConfig':
        """Create configuration from string paths."""
        return cls(
            field_test_dir=Path(field_test_dir),
            rinex_processor_path=Path(rinex_processor_path),
            segment_processor_path=Path(segment_processor_path),
            log_level=logging.INFO
        )


class ObservationGenerator:
    """Generate and process GNSS observation files."""

    def __init__(self, config: GeneratorConfig):
        """Initialize generator with configuration."""
        self.config = config
        self.logger = self._setup_logging()

        # Import processing modules
        self.rinex_processor = self._import_rinex_processor()
        self.segment_processor = self._import_segment_processor()

        if not (self.rinex_processor and self.segment_processor):
            raise ImportError("Failed to import required processing modules")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('obs_generator')
        if not logger.handlers:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'generate_obs_{timestamp}.log'
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            logger.setLevel(self.config.log_level)
            logger.info(f"Log file: {log_file}")

        return logger

    def _import_rinex_processor(self):
        """Import the RinexProcessor class."""
        try:
            spec = importlib.util.spec_from_file_location(
                "rinex_processor",
                self.config.rinex_processor_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            self.logger.error(f"Failed to import RinexProcessor: {e}")
            return None

    def _import_segment_processor(self):
        """Import the segment processor module."""
        try:
            spec = importlib.util.spec_from_file_location(
                "segment_processor",
                self.config.segment_processor_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            self.logger.error(f"Failed to import segment processor: {e}")
            return None

    def get_host_directories(self, config: Dict[str, Any]) -> Set[str]:
        """Extract unique host directories from config."""
        return set(segment['host'] for segment in config['segments'])

    def verify_config(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Verify the segments configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            required_keys = ['segments']
            for key in required_keys:
                if key not in config:
                    raise KeyError(f"Missing required key: {key}")
            for segment in config['segments']:
                for key in ['host', 'name', 'start_time', 'end_time']:
                    if key not in segment:
                        raise KeyError(f"Segment missing required key: {key}")
            return config
        except Exception as e:
            self.logger.error(f"Config verification failed: {e}")
            return None

    def verify_paths(self) -> bool:
        """Verify that all required paths exist."""
        paths_to_check = [
            (self.config.field_test_dir, "Field test directory"),
            (self.config.rinex_processor_path, "RINEX processor module"),
            (self.config.segment_processor_path, "Segment processor script"),
            (self.config.field_test_dir / 'segments_config.json', "Configuration file")
        ]
        for path, name in paths_to_check:
            if not path.exists():
                self.logger.error(f"{name} does not exist: {path}")
                return False
        return True

    def process(self) -> bool:
        """Process observation files according to configuration."""
        self.logger.info("Starting OBS file generation process")

        # Verify required paths and configuration file
        if not self.verify_paths():
            return False
        config_path = self.config.field_test_dir / 'segments_config.json'
        config = self.verify_config(config_path)
        if not config:
            self.logger.error("Configuration verification failed")
            return False

        # Synchronize remote GNSS host data
        self.logger.info("Synchronizing GNSS host data...")
        try:
            synced_hosts = pull_gnsshost_data.sync_all_hosts()
            if not synced_hosts:
                self.logger.error("No hosts were successfully synchronized.")
                return False
            self.logger.info(f"Synchronized hosts: {synced_hosts}")
        except Exception as e:
            self.logger.error(f"Data synchronization failed: {e}")
            return False

        # Process RINEX files for each host
        processor = self.rinex_processor.RinexProcessor()
        successful_hosts = 0
        host_dirs = self.get_host_directories(config)
        for host in sorted(host_dirs):
            host_path = self.config.field_test_dir / host
            host_path.mkdir(exist_ok=True)
            self.logger.info(f"Processing RINEX files for host: {host}")
            if processor.process_directory(host_path):
                successful_hosts += 1
            else:
                self.logger.error(f"Failed to process RINEX files for host: {host}")

        self.logger.info(f"Processed RINEX files for {successful_hosts} of {len(host_dirs)} host directories")

        # Process segments if any host was successfully processed
        if successful_hosts > 0:
            original_dir = os.getcwd()
            os.chdir(self.config.field_test_dir.parent)
            segment_config = self.segment_processor.SegmentConfig(
                field_test_dir=self.config.field_test_dir.name,
                log_level=self.config.log_level
            )
            segment_processor = self.segment_processor.SegmentProcessor(segment_config)
            self.logger.info("Starting segment processing")
            success = segment_processor.process_directory(self.config.field_test_dir)
            os.chdir(original_dir)
            if not success:
                self.logger.error("Segment processing failed")
                return False

        self.logger.info("OBS file generation complete")
        return successful_hosts > 0


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate and process GNSS observation files'
    )
    parser.add_argument('-d', '--field_test_dir',
                        default="/home/george/Scripts/gnssIR/field_test_1",
                        help='Base directory containing raw data')
    parser.add_argument('-r', '--rinex_processor',
                        default="/home/george/Scripts/gnssIR/rinex_processor.py",
                        help='Path to RinexProcessor module')
    parser.add_argument('-s', '--segment_processor',
                        default="/home/george/Scripts/gnssIR/segment_processor.py",
                        help='Path to segment processor script')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Create configuration from command-line arguments
    config = GeneratorConfig.from_paths(
        field_test_dir=args.field_test_dir,
        rinex_processor_path=args.rinex_processor,
        segment_processor_path=args.segment_processor
    )
    if args.verbose:
        config.log_level = logging.DEBUG

    try:
        generator = ObservationGenerator(config)
        success = generator.process()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error initializing generator: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
