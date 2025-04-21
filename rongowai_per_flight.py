import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import re


def extract_date_from_filename(filename):
    """Extract date from filename with format YYYYMMDD-HHMMSS_DEPARTURE-ARRIVAL_L1.nc.gpkg"""
    match = re.match(r'(\d{8})-(\d{6})_.*', filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        return datetime.strptime(f"{date_str}T{time_str}", '%Y%m%dT%H%M%S')
    return None


def load_and_prepare_data(file_path, incident_angle_threshold=30):
    """Load and prepare data from GeoJSON file with an incident angle filter"""
    print(f"Reading file: {file_path}")
    data = gpd.read_file(file_path)
    print(f"Loaded {len(data)} records")

    # Add timestamp column
    data['timestamp'] = data['filename'].apply(extract_date_from_filename)
    data = data.dropna(subset=['timestamp'])  # Remove records with missing timestamps

    # Add date column for grouping
    data['date'] = data['timestamp'].dt.date

    # Filter out invalid data (negative reflectivity values)
    valid_mask = data['surface_reflectivity_peak'] >= 0
    print(f"Removing {len(data) - sum(valid_mask)} records with negative reflectivity values")
    data = data[valid_mask].copy()

    # Filter by incident angle
    angle_mask = data['sp_inc_angle'] >= incident_angle_threshold
    print(
        f"Applying incident angle threshold >= {incident_angle_threshold}° ({sum(angle_mask)}/{len(data)} records kept)")
    data = data[angle_mask].copy()

    # Create land cover type mapping for reference
    landcover_mapping = {
        -1: 'Ocean',
        1: 'Artificial',
        2: 'Barely vegetated',
        3: 'Inland water',
        4: 'Crop',
        5: 'Grass',
        6: 'Shrub',
        7: 'Forest'
    }

    data['land_type'] = data['sp_surface_type'].map(landcover_mapping)

    # Identify unique flight passes
    data['flight_id'] = data['filename'].str.split('_').str[0]

    return data


def compute_flight_statistics(data):
    """Compute statistics for each flight pass"""
    # Filter data by polarization type
    lhcp_data = data[data['ddm_ant'] == 2].copy()
    rhcp_data = data[data['ddm_ant'] == 3].copy()

    print(f"LHCP measurements: {len(lhcp_data)}")
    print(f"RHCP measurements: {len(rhcp_data)}")

    # Initialize dataframe for combined results
    flight_stats = pd.DataFrame()

    # Process LHCP data
    if not lhcp_data.empty:
        lhcp_flights = lhcp_data.groupby(['flight_id', 'timestamp'])['surface_reflectivity_peak'].agg(
            ['mean', 'std', 'count']).reset_index()
        lhcp_flights = lhcp_flights.rename(columns={
            'mean': 'lhcp_mean',
            'std': 'lhcp_std',
            'count': 'lhcp_count'
        })
        flight_stats = lhcp_flights

    # Process RHCP data and prepare ratio data
    if not rhcp_data.empty:
        rhcp_flights = rhcp_data.groupby(['flight_id', 'timestamp'])['surface_reflectivity_peak'].agg(
            ['mean', 'std', 'count']).reset_index()
        rhcp_flights = rhcp_flights.rename(columns={
            'mean': 'rhcp_mean',
            'std': 'rhcp_std',
            'count': 'rhcp_count'
        })

        # Merge with LHCP data if available
        if not flight_stats.empty:
            # Merge on flight_id and timestamp
            flight_stats = pd.merge(flight_stats, rhcp_flights, on=['flight_id', 'timestamp'], how='outer')

            # Calculate ratio where both measurements exist
            ratio_mask = (~flight_stats['lhcp_mean'].isna()) & (~flight_stats['rhcp_mean'].isna()) & (
                        flight_stats['rhcp_mean'] != 0)
            flight_stats.loc[ratio_mask, 'lhcp_rhcp_ratio'] = flight_stats.loc[ratio_mask, 'lhcp_mean'] / \
                                                              flight_stats.loc[ratio_mask, 'rhcp_mean']
            flight_stats.loc[ratio_mask, 'ratio_std'] = np.sqrt(
                (flight_stats.loc[ratio_mask, 'lhcp_std'] / flight_stats.loc[ratio_mask, 'lhcp_mean']) ** 2 +
                (flight_stats.loc[ratio_mask, 'rhcp_std'] / flight_stats.loc[ratio_mask, 'rhcp_mean']) ** 2) * \
                                                        flight_stats.loc[ratio_mask, 'lhcp_rhcp_ratio']
        else:
            flight_stats = rhcp_flights

    # Sort by timestamp
    flight_stats = flight_stats.sort_values('timestamp')

    return flight_stats


def plot_lhcp_data(stats, location_name):
    """Create a figure showing LHCP statistics over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    # Top panel: Mean reflectivity with error bars
    ax1.errorbar(stats['timestamp'], stats['lhcp_mean'], yerr=stats['lhcp_std'],
                 fmt='o-', color='blue', ecolor='lightblue',
                 elinewidth=1, capsize=3, label='LHCP Reflectivity')

    # Bottom panel: Sample counts
    ax2.bar(stats['timestamp'], stats['lhcp_count'], color='teal', alpha=0.7, label='LHCP Observations')

    # Format date axis
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax2.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add labels and styling
    ax1.set_ylabel('LHCP Surface Reflectivity')
    ax1.set_title(f'LHCP Surface Reflectivity Over Time ({location_name})')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Flight Date')
    ax2.set_ylabel('Observation Count')
    ax2.set_title('Number of Observations per Flight')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ratio_data(stats, location_name):
    """Create a figure showing LHCP/RHCP ratio statistics over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)

    ratio_mask = ~stats['lhcp_rhcp_ratio'].isna()

    if sum(ratio_mask) > 0:
        # Top panel: Ratio with error bars
        ax1.errorbar(stats.loc[ratio_mask, 'timestamp'], stats.loc[ratio_mask, 'lhcp_rhcp_ratio'],
                     yerr=stats.loc[ratio_mask, 'ratio_std'],
                     fmt='o-', color='purple', ecolor='plum',
                     elinewidth=1, capsize=3, label='LHCP/RHCP Ratio')

        # Bottom panel: Sample counts (minimum of LHCP and RHCP counts)
        stats['min_count'] = stats[['lhcp_count', 'rhcp_count']].min(axis=1)
        ax2.bar(stats.loc[ratio_mask, 'timestamp'], stats.loc[ratio_mask, 'min_count'],
                color='darkmagenta', alpha=0.7, label='Observations with Both Polarizations')

        # Add horizontal line at ratio = 1 for reference
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal LHCP/RHCP')
    else:
        ax1.text(0.5, 0.5, 'No valid ratio data available',
                 ha='center', va='center', transform=ax1.transAxes)

    # Format date axis
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax2.xaxis.set_major_formatter(date_format)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add labels and styling
    ax1.set_ylabel('LHCP/RHCP Reflectivity Ratio')
    ax1.set_title(f'LHCP/RHCP Reflectivity Ratio Over Time ({location_name})')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Flight Date')
    ax2.set_ylabel('Observation Count')
    ax2.set_title('Number of Observations with Both Polarizations per Flight')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_site(file_path, site_name, incident_angle_threshold=30):
    """Analyze data for a single site and return figures"""
    # Load and prepare data
    data = load_and_prepare_data(file_path, incident_angle_threshold)

    # Compute statistics
    flight_stats = compute_flight_statistics(data)

    # Plot figures
    lhcp_fig = plot_lhcp_data(flight_stats, site_name)
    ratio_fig = plot_ratio_data(flight_stats, site_name)

    return lhcp_fig, ratio_fig, flight_stats


def main():
    # File paths
    warkworth_path = '/home/george/Documents/Work/rongowai_timeseries/wark_data_spatial.geojson'
    taupo_path = '/home/george/Documents/Work/rongowai_timeseries/taup_data_spatial.geojson'

    # Set incident angle threshold
    incident_angle_threshold = 30

    # Analyze Warkworth data
    print("\n--- Analyzing Warkworth data ---")
    wark_lhcp_fig, wark_ratio_fig, wark_stats = analyze_site(warkworth_path, "Warkworth", incident_angle_threshold)

    # Analyze Taupo data
    print("\n--- Analyzing Taupo data ---")
    taup_lhcp_fig, taup_ratio_fig, taup_stats = analyze_site(taupo_path, "Taupo", incident_angle_threshold)

    # Display the figures
    plt.figure(wark_lhcp_fig.number)
    plt.show()
    plt.figure(wark_ratio_fig.number)
    plt.show()
    plt.figure(taup_lhcp_fig.number)
    plt.show()
    plt.figure(taup_ratio_fig.number)
    plt.show()

    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Warkworth: {len(wark_stats)} flights analyzed")
    print(f"Taupo: {len(taup_stats)} flights analyzed")


if __name__ == "__main__":
    main()