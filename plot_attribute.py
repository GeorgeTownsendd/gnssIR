#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# ============================
# Configuration Variables
# ============================

# Target directory containing the *_snr_timeseries.csv files for a GNSS host.
# Change this path as needed.
gnss_host = 'gnsshost-4'
TARGET_DIRECTORY = f"/home/george/Scripts/gnssIR/field_test_1/processed/rinex/{gnss_host}"

# Define the start and end time strings (in UTC/GPS time) for plotting.
# You can manually adjust these as needed.
START_TIME_STR = "2025-02-06T22:45:00"
END_TIME_STR = "2025-02-07T00:55:00"

# Choose the attribute to plot: "snr" for average SNR or "count" for number of observations.
PLOT_ATTRIBUTE = "snr"


# ============================
# Functions
# ============================

def load_and_concatenate_csv(csv_dir):
    """
    Load all CSV files matching '*_snr_timeseries.csv' from the provided directory,
    concatenate them into a single DataFrame, and sort by time.
    """
    csv_dir = Path(csv_dir)
    csv_files = sorted(csv_dir.glob('*_snr_timeseries.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    df_list = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, parse_dates=['time'])
        df_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True)
    full_df.sort_values('time', inplace=True)
    return full_df


def resample_data(df):
    """
    Given a DataFrame with a datetime 'time' column and a numeric 'snr' column,
    set 'time' as the index and resample the data into 1-minute bins.

    Returns a DataFrame with:
      - 'snr': the mean SNR in each minute,
      - 'count': the number of observations in each minute.
    """
    df = df.set_index('time')
    resampled = df.resample('1T').agg({'snr': 'mean'})
    resampled['count'] = df.resample('1T').size()
    return resampled


def plot_attribute(resampled, attribute="snr"):
    """
    Plot the chosen attribute from the resampled data.

    Parameters:
      resampled : DataFrame indexed by time with columns 'snr' and 'count'
      attribute : either 'snr' (for average SNR) or 'count' (for observation count)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(resampled.index, resampled[attribute], marker='o', linestyle='-')
    plt.xlabel("Time (UTC)")
    if attribute == 'snr':
        plt.ylabel("Average SNR")
        plt.title(f"{gnss_host} average SNR")
    elif attribute == 'count':
        plt.ylabel("Observation Count")
        plt.title("Observation Count over Time (1-minute intervals)")
    plt.grid(True)

    start_time = pd.to_datetime(START_TIME_STR)
    end_time = pd.to_datetime(END_TIME_STR)

    plt.xlim(
        min(start_time, resampled.index.min()),  # Use start_time if earlier
        max(end_time, resampled.index.max())  # Use end_time if later
    )

    plt.tight_layout()
    plt.show()


def main():
    # Load the full dataset from the target directory.
    try:
        print(TARGET_DIRECTORY)
        df = load_and_concatenate_csv(TARGET_DIRECTORY)
    except FileNotFoundError as e:
        print(e)
        return

    # Convert the start/end times from strings to Timestamps.
    start_time = pd.to_datetime(START_TIME_STR)
    end_time = pd.to_datetime(END_TIME_STR)

    # Filter the DataFrame to the specified time range.
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    if df.empty:
        print("No data available in the specified time range.")
        return

    # Resample the data into 1-minute intervals.
    resampled = resample_data(df)
    # Plot the chosen attribute.
    plot_attribute(resampled, attribute=PLOT_ATTRIBUTE)


if __name__ == "__main__":
    main()
