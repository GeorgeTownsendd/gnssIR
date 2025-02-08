import os
import glob
import subprocess
from datetime import datetime, timedelta
import re

TEQC_PATH = "/home/george/Scripts/gnssIR/bin/teqc"
GAP_THRESHOLD = timedelta(minutes=1)  # Gap threshold for splitting segments


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
        print(f"Error running teqc on {filename}: {e}")
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
            print(f"\nFound gap of {gap} between:")
            print(f"  {current_file} ending at {current_end}")
            print(f"  {next_file} starting at {next_start}")

        current_segment.append(next_file)

    if current_segment:
        segments.append(current_segment)

    return segments


def process_file(input_file, file_times, prev_file=None, next_file=None):
    """Process a single RINEX file, handling overlaps"""
    actual_start, actual_end = file_times[input_file]
    print(f"\nProcessing: {input_file}")
    print(f"Original time range: {actual_start} -> {actual_end}")

    start_time = actual_start
    end_time = actual_end

    if prev_file:
        prev_end = file_times[prev_file][1]
        if start_time <= prev_end:
            overlap = prev_end - start_time
            midpoint = start_time + (overlap / 2)
            start_time = midpoint
            print(f"Overlap with previous file: {overlap}")
            print(f"Using midpoint: {start_time}")

    if next_file:
        next_start = file_times[next_file][0]
        if end_time >= next_start:
            overlap = end_time - next_start
            midpoint = next_start + (overlap / 2)
            end_time = midpoint
            print(f"Overlap with next file: {overlap}")
            print(f"Using midpoint: {end_time}")

    output_file = f"processed_{input_file}"
    start_str = start_time.strftime('%Y%m%d%H%M%S.%f')[:15]
    end_str = end_time.strftime('%Y%m%d%H%M%S.%f')[:15]

    cmd = [TEQC_PATH, '+st', start_str, '+e', end_str, input_file]
    print(f"Trimming to: {start_time} -> {end_time}")

    try:
        with open(output_file, 'w') as f:
            subprocess.run(cmd, stdout=f, check=True)
        print(f"Successfully processed {input_file} -> {output_file}")

        proc_start, proc_end = parse_teqc_meta(output_file)
        if proc_start and proc_end:
            print(f"Processed file spans: {proc_start} -> {proc_end}")

        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_file}: {e}")
        return None


def merge_segment(processed_files, segment_num):
    """Merge a segment of processed files"""
    if not processed_files:
        return False

    output_file = f"merged_{segment_num}.obs"
    print(f"\nMerging segment {segment_num} to {output_file}")
    print(f"Files to merge: {' '.join(processed_files)}")

    cmd = [TEQC_PATH, '+rec_mnm'] + processed_files
    try:
        with open(output_file, 'w') as f:
            subprocess.run(cmd, stdout=f, check=True)

        # Verify the merged file
        start, end = parse_teqc_meta(output_file)
        print(f"Successfully created {output_file}")
        print(f"Merged file spans: {start} -> {end}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error merging segment {segment_num}: {e}")
        return False


def main():
    # Clean up existing files
    for f in glob.glob('processed_*.obs'):
        os.remove(f)
    for f in glob.glob('merged_*.obs'):
        os.remove(f)

    # Get all .obs files and sort them
    obs_files = sorted(glob.glob('*.obs'))
    obs_files = [f for f in obs_files if not f.startswith(('processed_', 'trimmed_', 'merged'))]

    if not obs_files:
        print("No valid .obs files found in current directory")
        return

    print(f"Found {len(obs_files)} observation files to process")

    # Get file times
    file_times = {}
    for f in obs_files:
        start, end = parse_teqc_meta(f)
        if start and end:
            file_times[f] = (start, end)
            print(f"File {f} spans: {start} -> {end}")

    # Find continuous segments
    segments = find_continuous_segments(file_times)
    print(f"\nFound {len(segments)} continuous segments")

    # Process each segment
    for segment_num, segment_files in enumerate(segments, 1):
        print(f"\nProcessing segment {segment_num} ({len(segment_files)} files)")
        processed_files = []

        for i, obs_file in enumerate(segment_files):
            prev_file = segment_files[i - 1] if i > 0 else None
            next_file = segment_files[i + 1] if i < len(segment_files) - 1 else None

            processed_file = process_file(obs_file, file_times, prev_file, next_file)
            if processed_file:
                processed_files.append(processed_file)

        if processed_files:
            merge_segment(processed_files, segment_num)


if __name__ == "__main__":
    main()