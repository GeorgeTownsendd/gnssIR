import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime


def read_vwc_file(file_path):
    """
    Read a VWC file and return a pandas DataFrame.
    Skips the header lines starting with '%'.
    """
    # Count number of header rows
    header_rows = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('%'):
                header_rows += 1
            else:
                break

    # Read the file, skipping the header rows
    df = pd.read_csv(file_path,
                     skiprows=header_rows,
                     delim_whitespace=True,
                     names=['FracYr', 'Year', 'DOY', 'VWC', 'Month', 'Day'])

    # Create datetime column for better plotting
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    return df


def plot_vwc_timeseries(file_list, master_file=None, output_path=None):
    """
    Plot VWC time series for multiple files and master file.

    Args:
        file_list (list): List of paths to VWC files
        master_file (str): Path to master/true VWC file
        output_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 6))

    # Plot master file first in black
    if master_file:
        master_df = read_vwc_file(master_file)
        plt.plot(master_df['Date'], master_df['VWC'], 'k-',
                label='Half Sky All Satellites', alpha=0.7, linewidth=2)

    # Plot other files
    for file_path in file_list:
        df = read_vwc_file(file_path)
        station_name = Path(file_path).stem
        plt.plot(df['Date'], df['VWC'], label=station_name, alpha=0.7)

    plt.xlabel('Date')
    plt.ylabel('VWC')
    plt.title('VWC Comparison (original data @ 0.1 precision)')
    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(output_path)
    plt.show()


def calculate_errors(test_files, master_file):
    """
    Calculate errors between test files and a master file.

    Args:
        test_files (list): List of paths to test VWC files
        master_file (str): Path to master/true VWC file

    Returns:
        dict: Dictionary of errors for each file
    """
    master_df = read_vwc_file(master_file)
    errors = {}

    for test_file in test_files:
        test_df = read_vwc_file(test_file)

        # Merge dataframes on Date to compare VWC values
        merged = pd.merge(master_df[['Date', 'VWC']],
                          test_df[['Date', 'VWC']],
                          on='Date',
                          suffixes=('_master', '_test'))

        # Calculate errors
        errors[Path(test_file).stem] = merged['VWC_test'] - merged['VWC_master']

    return errors


def plot_error_histogram(errors, output_path=None):
    """
    Plot histogram of errors for each file.

    Args:
        errors (dict): Dictionary of errors for each file
        output_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(12, 6))

    for station, error_values in errors.items():
        plt.hist(error_values, bins=30, alpha=0.5, label=station)

    plt.xlabel('Error (VWC difference)')
    plt.ylabel('Frequency')
    plt.title('VWC Error Distribution vs Half Sky All Satellites (original data @ 0.1 precision)')
    plt.legend()
    plt.grid(True)

    if output_path:
        plt.savefig(output_path)
    plt.show()


def main():
    # Example usage
    vwc_files = ['/home/george/Documents/Work/original_single_satellites/p038_sat7_vwc.txt',
                 '/home/george/Documents/Work/original_single_satellites/p038_sat1_vwc.txt',
                 '/home/george/Documents/Work/original_single_satellites/p038_sat8_vwc.txt']
    master_file = '/home/george/Documents/Work/original_single_satellites/half_sky_vwc.txt'

    # Plot time series with master file
    plot_vwc_timeseries(vwc_files, master_file)

    # Calculate and plot errors
    errors = calculate_errors(vwc_files, master_file)
    plot_error_histogram(errors)


if __name__ == '__main__':
    main()