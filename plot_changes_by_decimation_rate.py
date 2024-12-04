import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
import numpy as np


def find_matching_arcs(dataframes, time_tolerance=5 / 60.0):
    """
    Find arcs that appear in all decimation rates by matching satellite number,
    rise/set flag, and UTC time within tolerance.

    Args:
        dataframes: List of dataframes for each decimation rate
        time_tolerance: Time tolerance in hours (default 5 minutes = 5/60 hours)

    Returns:
        List of filtered dataframes containing only matching arcs
    """
    # Start with arcs from the first decimation rate as reference
    ref_df = dataframes[0]

    # Create key identifiers for reference arcs
    ref_keys = set([f"{int(row['sat'])}_{int(row['rise'])}" for _, row in ref_df.iterrows()])

    # For each reference arc, find matching arcs in all other dataframes
    matching_indices = []
    for dec_df in dataframes:
        dec_matches = []

        # Group by satellite and rise/set flag
        for key in ref_keys:
            sat, rise = key.split('_')
            sat, rise = int(sat), int(rise)

            # Get reference times for this satellite/rise combination
            ref_times = ref_df[(ref_df['sat'].astype(int) == sat) &
                               (ref_df['rise'].astype(int) == rise)]['UTCtime'].values

            # Find matches in current decimation rate
            for ref_time in ref_times:
                # Find rows within time tolerance
                matches = dec_df[(dec_df['sat'].astype(int) == sat) &
                                 (dec_df['rise'].astype(int) == rise) &
                                 (abs(dec_df['UTCtime'] - ref_time) <= time_tolerance)]

                if not matches.empty:
                    dec_matches.extend(matches.index.tolist())

        matching_indices.append(dec_matches)

    # Filter each dataframe to keep only matching arcs
    filtered_dfs = [df.loc[indices] for df, indices in zip(dataframes, matching_indices)]

    return filtered_dfs


def plot_variable_by_decimation(results_dir='results/', variable_name=None):
    """
    Load result files and plot a specified variable against decimation rate,
    considering only arcs that are present in all decimation rates.

    Args:
        results_dir (str): Path to results directory
        variable_name (str): Name of the variable to plot (must match column headers)
    """
    # Define column names based on the file structure
    columns = ['year', 'doy', 'RH', 'sat', 'UTCtime', 'Azim', 'Amp', 'eminO',
               'emaxO', 'NumbOf', 'freq', 'rise', 'EdotF', 'PkNoise', 'DelT',
               'MJD', 'refr_model']

    if variable_name not in columns:
        print(f"Error: variable_name must be one of: {', '.join(columns)}")
        return

    # Get all txt files (excluding arc directories)
    result_files = sorted([f for f in Path(results_dir).glob('*.txt')])

    # Data structure to hold results
    dataframes = []
    decimation_rates = []

    for file_path in result_files:
        match = re.search(r'dec(\d+)\.txt$', str(file_path))
        if not match:
            continue

        dec_rate = int(match.group(1))

        try:
            df = pd.read_csv(file_path, comment='%', names=columns, delim_whitespace=True)
            df['decimation'] = dec_rate
            dataframes.append(df)
            decimation_rates.append(dec_rate)
            print(f"Loaded {file_path.name}: {len(df)} rows")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not dataframes:
        print("No valid data files found")
        return

    # Find matching arcs across all decimation rates
    filtered_dfs = find_matching_arcs(dataframes)

    # Combine all filtered dataframes
    combined_df = pd.concat(filtered_dfs)

    # Count unique arcs (using satellite, rise/set, and rounded time)
    n_arcs = len(set([f"{int(row['sat'])}_{int(row['rise'])}_{round(row['UTCtime'], 2)}"
                      for _, row in filtered_dfs[0].iterrows()]))

    # Create figure with specific size and DPI for publication quality
    plt.figure(figsize=(8, 6), dpi=300)

    # Create box plot with customized style
    box_plot = plt.boxplot([group[variable_name].values for name, group in combined_df.groupby('decimation')],
                           positions=sorted(combined_df['decimation'].unique()),
                           widths=0.7,
                           patch_artist=True,
                           medianprops=dict(color="black", linewidth=1.5),
                           flierprops=dict(marker='o', markerfacecolor='gray', markersize=4))

    # Customize box colors
    for patch in box_plot['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Customize plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Decimation Rate', fontsize=12)
    plt.ylabel(variable_name, fontsize=12)
    plt.title(f'Distribution of {variable_name} vs Decimation Rate\n(n={n_arcs} matching arcs)',
              fontsize=12, pad=20)

    # Print summary statistics
    print("\nSummary statistics:")
    summary = combined_df.groupby('decimation')[variable_name].agg(['count', 'mean', 'std', 'min', 'max'])
    print(summary)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    print(f"\nTotal number of matching arcs across all decimation rates: {n_arcs}")
    print("\nDecimation rates included:", sorted(decimation_rates))

    plt.show()

    return combined_df

# Example usage:
plot_variable_by_decimation(variable_name='RH')