import os
import glob
import subprocess
from datetime import datetime, timedelta
import re
import logging
from pathlib import Path

TEQC_PATH = "/home/george/Scripts/gnssIR/bin/teqc"
GAP_THRESHOLD = timedelta(minutes=1)


def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'rinex_processor_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def convert_ubx_to_obs(ubx_files):
    """Convert UBX files to OBS format using convbin"""
    logging.info(f"Found {len(ubx_files)} UBX files to convert")
    converted_files = []

    for ubx_file in sorted(ubx_files):
        base_name = Path(ubx_file).stem
        output_file = f"{base_name}.obs"

        if os.path.exists(output_file):
            logging.info(f"Skipping {ubx_file} as {output_file} already exists")
            converted_files.append(output_file)
            continue

        cmd = ['convbin', '-od', '-os', '-v', '2.11', '-r', 'ubx', ubx_file, '-o', output_file]
        logging.info(f"Converting {ubx_file} to {output_file}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logging.info(f"Successfully converted {ubx_file}")
            logging.debug(f"convbin output:\n{result.stdout}")
            if result.stderr:
                logging.warning(f"convbin warnings:\n{result.stderr}")
            converted_files.append(output_file)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error converting {ubx_file}: {e}")
            logging.error(f"convbin error output:\n{e.stderr}")
            continue

    return converted_files


def parse_teqc_meta(filename):
    """Extract start and end times from teqc +meta output"""
    try:
        cmd = [TEQC_PATH, '+meta', filename]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        start_match = re.search(r'start date & time:\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})', result.stdout)
        end_match = re.search(r'final date & time:\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})', result.stdout)

        if start_match and end_match:
            start_time = datetime.strptime(start_match.group(1), '%Y-%m-%d %H:%M:%S.%f')
            end_time = datetime.strptime(end_match.group(1), '%Y-%m-%d %H:%M:%S.%f')
            return start_time, end_time

        return None, None
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running teqc on {filename}: {e}")
        return None, None


def find_continuous_segments(file_times):
    """Group files into continuous segments based on gaps"""
    segments = []
    current_segment = []

    files = sorted(file_times.keys())
    current_segment.append(files[0])

    for i in range(len(files) - 1):
        current_file = files[i]
        next_file = files[i + 1]

        current_end = file_times[current_file][1]
        next_start = file_times[next_file][0]

        gap = next_start - current_end

        if gap > GAP_THRESHOLD:
            segments.append(current_segment)
            current_segment = []
            logging.info(f"Found gap of {gap} between:")
            logging.info(f"  {current_file} ending at {current_end}")
            logging.info(f"  {next_file} starting at {next_start}")

        current_segment.append(next_file)

    if current_segment:
        segments.append(current_segment)

    return segments


def process_file(input_file, file_times, prev_file=None, next_file=None):
    """Process a single RINEX file, handling overlaps"""
    actual_start, actual_end = file_times[input_file]
    logging.info(f"Processing: {input_file}")
    logging.info(f"Original time range: {actual_start} -> {actual_end}")

    start_time = actual_start
    end_time = actual_end

    if prev_file:
        prev_end = file_times[prev_file][1]
        if start_time <= prev_end:
            overlap = prev_end - start_time
            midpoint = start_time + (overlap / 2)
            start_time = midpoint
            logging.info(f"Overlap with previous file: {overlap}")
            logging.info(f"Using midpoint: {start_time}")

    if next_file:
        next_start = file_times[next_file][0]
        if end_time >= next_start:
            overlap = end_time - next_start
            midpoint = next_start + (overlap / 2)
            end_time = midpoint
            logging.info(f"Overlap with next file: {overlap}")
            logging.info(f"Using midpoint: {end_time}")

    output_file = f"processed_{input_file}"
    start_str = start_time.strftime('%Y%m%d%H%M%S.%f')[:15]
    end_str = end_time.strftime('%Y%m%d%H%M%S.%f')[:15]

    cmd = [TEQC_PATH, '+st', start_str, '+e', end_str, input_file]
    logging.info(f"Trimming to: {start_time} -> {end_time}")

    try:
        with open(output_file, 'w') as f:
            subprocess.run(cmd, stdout=f, check=True)
        logging.info(f"Successfully processed {input_file} -> {output_file}")

        proc_start, proc_end = parse_teqc_meta(output_file)
        if proc_start and proc_end:
            logging.info(f"Processed file spans: {proc_start} -> {proc_end}")

        return output_file
    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing {input_file}: {e}")
        return None


def merge_segment(processed_files, segment_num):
    """Merge a segment of processed files"""
    if not processed_files:
        return False

    output_file = f"merged_{segment_num}.obs"
    logging.info(f"Merging segment {segment_num} to {output_file}")
    logging.info(f"Files to merge: {' '.join(processed_files)}")

    cmd = [TEQC_PATH, '+rec_mnm'] + processed_files
    try:
        with open(output_file, 'w') as f:
            subprocess.run(cmd, stdout=f, check=True)

        # Verify the merged file
        start, end = parse_teqc_meta(output_file)
        logging.info(f"Successfully created {output_file}")
        logging.info(f"Merged file spans: {start} -> {end}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error merging segment {segment_num}: {e}")
        return False


def main():
    log_file = setup_logging()
    logging.info("Starting RINEX processing script")

    # Clean up existing files
    for f in glob.glob('processed_*.obs'):
        os.remove(f)
    for f in glob.glob('merged_*.obs'):
        os.remove(f)

    # First check for .obs files
    obs_files = sorted(glob.glob('*.obs'))
    obs_files = [f for f in obs_files if not f.startswith(('processed_', 'trimmed_', 'merged'))]

    # If no .obs files found, look for .ubx files
    if not obs_files:
        ubx_files = glob.glob('*.ubx')
        if ubx_files:
            logging.info("No OBS files found, converting UBX files")
            obs_files = convert_ubx_to_obs(ubx_files)
        else:
            logging.error("No OBS or UBX files found in current directory")
            return

    if not obs_files:
        logging.error("No valid observation files found after conversion")
        return

    logging.info(f"Found {len(obs_files)} observation files to process")

    # Get file times
    file_times = {}
    for f in obs_files:
        start, end = parse_teqc_meta(f)
        if start and end:
            file_times[f] = (start, end)
            logging.info(f"File {f} spans: {start} -> {end}")

    # Find continuous segments
    segments = find_continuous_segments(file_times)
    logging.info(f"Found {len(segments)} continuous segments")

    # Process each segment
    for segment_num, segment_files in enumerate(segments, 1):
        logging.info(f"Processing segment {segment_num} ({len(segment_files)} files)")
        processed_files = []

        for i, obs_file in enumerate(segment_files):
            prev_file = segment_files[i - 1] if i > 0 else None
            next_file = segment_files[i + 1] if i < len(segment_files) - 1 else None

            processed_file = process_file(obs_file, file_times, prev_file, next_file)
            if processed_file:
                processed_files.append(processed_file)

        if processed_files:
            merge_segment(processed_files, segment_num)

    logging.info("Processing complete")
    logging.info(f"Log file saved as: {log_file}")


if __name__ == "__main__":
    main()