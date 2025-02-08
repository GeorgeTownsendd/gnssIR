import georinex as gr
from pathlib import Path
import pandas as pd


def process_rinex_file(rinex_file):
    """
    Process a single RINEX file and extract the SNR time series from the S1C observation channel.

    The routine loads the file using GeoRinex (filtering for the GPS constellation with `use='G'`),
    converts the time coordinate to a Pandas datetime object, and extracts the S1C data.
    The resulting DataArray is converted to a DataFrame, and the satellite identifier (typically a string like "G12")
    is split into its constituent parts:

        - 'svtype' : The constellation letter (e.g. 'G' for GPS)
        - 'svid'   : The satellite number (e.g. '12')

    The final DataFrame is then sorted in time.
    """
    try:
        # Load observation data (only GPS satellites, i.e. use 'G')
        obs = gr.load(rinex_file, use='G')

        # Ensure the time coordinate is in pandas datetime format (if not already)
        obs = obs.assign_coords(time=pd.to_datetime(obs.time.values))

        # Extract SNR data for the S1C channel.
        # (This will typically yield a DataArray with dimensions ("time", "sv").)
        snr_data = obs['S1C']

        # Convert the DataArray to a DataFrame.
        # Typically, the resulting DataFrame has columns: ['time', 'sv', 'S1C'].
        df = snr_data.to_dataframe().reset_index()

        # If the satellite identifier is provided as a single column ("sv"), split it into
        # "svtype" (the constellation letter) and "svid" (the numeric part).
        if 'sv' in df.columns:
            df['svtype'] = df['sv'].str[0]
            df['svid'] = df['sv'].str[1:]
            # Rename the observation column to "snr"
            df.rename(columns={'S1C': 'snr'}, inplace=True)
            # Drop the original "sv" column and order the columns appropriately.
            df = df[['time', 'svtype', 'svid', 'snr']]
        else:
            # If the expected "sv" column is not present, assume the DataFrame already
            # contains the desired columns.
            df.rename(columns={'S1C': 'snr'}, inplace=True)

        # Sort the DataFrame by time
        df.sort_values('time', inplace=True)
        return df

    except Exception as e:
        print(f"Error processing {rinex_file}: {e}")
        return None


def process_rinex_directory(directory_path):
    """
    Process all RINEX files within the given directory.

    This routine searches for files with the ".crx.gz" extension (typically indicating Hatanaka-compressed RINEX files),
    processes each file to extract the S1C SNR time series, and writes the resulting DataFrame as a CSV.
    It also prints summary information including time range, total observations, and unique satellites.
    """
    directory = Path(directory_path)
    rinex_files = sorted(directory.glob('*.crx.gz'))
    print(f"Found {len(rinex_files)} RINEX files in {directory}")

    for rinex_file in rinex_files:
        # Construct a CSV output filename.
        # Since rinex_file.stem for a file like "foo.crx.gz" returns "foo.crx", we remove the extra suffix.
        filename = rinex_file.name
        if filename.endswith('.crx.gz'):
            base = filename[:-7]  # remove ".crx.gz"
        else:
            base = rinex_file.stem
        output_csv = rinex_file.parent / f"{base}_snr_timeseries.csv"

        # Process the RINEX file
        df = process_rinex_file(rinex_file)
        if df is not None:
            # Save the DataFrame to CSV
            df.to_csv(output_csv, index=False)
            print(f"Saved time series for {rinex_file.name} to {output_csv.name}")
            print(f"Time range: {df['time'].min()} to {df['time'].max()}")
            print(f"Total observations: {len(df)}")
            # Summarize unique satellites (grouped by constellation and satellite id)
            unique_satellites = df.groupby(['svtype', 'svid']).size()
            print("Unique satellites:")
            print(unique_satellites)
            print("\n")


def main():
    """
    Main execution routine.

    The script iterates over a predefined list of directories (each corresponding to a GNSS host) and processes all
    RINEX files found within those directories.
    """
    host_dirs = [
        '/home/george/Scripts/gnssIR/field_test_1/processed/rinex/gnsshost-2',
        '/home/george/Scripts/gnssIR/field_test_1/processed/rinex/gnsshost-3',
        '/home/george/Scripts/gnssIR/field_test_1/processed/rinex/gnsshost-4'
    ]

    for host_dir in host_dirs:
        print(f"\nProcessing directory: {host_dir}")
        process_rinex_directory(host_dir)


if __name__ == "__main__":
    main()
