import os
import gzip
from pathlib import Path


def round_to_quarter(x):
    """Round to nearest 0.25"""
    return round(float(x) * 4) / 4


def format_snr(value):
    """Format SNR value to match original format: always showing two decimal places"""
    return f"{value:5.2f}"


def process_line(line, rounding_type):
    """Process a single line of the file using exact positions for SNR values"""
    if len(line.strip()) == 0:
        return line

    try:
        # Extract the three SNR values using their exact positions
        snr1 = float(line[48:54].strip())
        snr2 = float(line[55:61].strip())
        snr3 = float(line[62:].strip())

        # Apply rounding
        if rounding_type == 'integer':
            rounded = [round(x) for x in [snr1, snr2, snr3]]
        else:  # quarter
            rounded = [round_to_quarter(x) for x in [snr1, snr2, snr3]]

        # Reconstruct line preserving exact format
        new_line = line[:49]  # Keep everything up to first SNR value unchanged

        # Format each SNR with proper spacing
        new_line += format_snr(rounded[0])
        new_line += "  "  # Two spaces
        new_line += format_snr(rounded[1])
        new_line += "  "  # Three spaces
        new_line += format_snr(rounded[2])
        new_line += "\n"

        return new_line
    except (ValueError, IndexError):
        return line  # Return original line if any parsing fails


def process_file(input_path, output_path, rounding_type):
    """Process a single SNR file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Read from gzipped file and write to new gzipped file
    with gzip.open(input_path, 'rt') as f_in:
        with gzip.open(output_path.with_suffix('.snr66.gz'), 'wt') as f_out:
            for line in f_in:
                processed_line = process_line(line, rounding_type)
                f_out.write(processed_line)


def main():
    input_dir = Path("data/refl_code/2017/snr/p038")
    base_output_dir = Path("data/refl_code/2017/snr")

    integer_dir = base_output_dir / "p038_integer"
    quarter_dir = base_output_dir / "p038_quarter"

    os.makedirs(integer_dir, exist_ok=True)
    os.makedirs(quarter_dir, exist_ok=True)

    for input_file in input_dir.glob("*.snr66.gz"):
        base_name = input_file.name[:-3]  # Remove .gz

        integer_output = integer_dir / base_name
        process_file(input_file, integer_output, 'integer')

        quarter_output = quarter_dir / base_name
        process_file(input_file, quarter_output, 'quarter')

        print(f"Processed {input_file.name}")


if __name__ == "__main__":
    main()