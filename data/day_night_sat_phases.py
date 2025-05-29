# Import necessary libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta, date
from matplotlib.dates import DateFormatter
from scipy import stats # For linear regression (slope calculation)
import seaborn as sns   # For enhanced plotting (box plot)
import calendar # To check for leap years

# --- Configuration ---

# Base path where year directories are located
base_path = "/home/george/Scripts/gnssIR/data/refl_code/"
STATION_ID = "p038" # Station identifier used in the path

# Specify the year(s) to process
YEARS_TO_PROCESS = [2024] # Start with 2022 as requested
# YEARS_TO_PROCESS = range(2021, 2025) # For 2021-2024 later

# Path to the DAILY rainfall/ET data file
# Assumes columns: DateTime, Precipitation, Ensemble_ET
rainfall_path = "/home/george/Documents/Work/p038_longterm/p038_et_precip_with_vwc.csv"

# Site Location (Example: US Mountain Time)
# *** IMPORTANT: Update these to your actual site coordinates and timezone ***
SITE_LATITUDE = 34.147255
SITE_LONGITUDE = -103.407338
SITE_ELEVATION = 1213 # Elevation in meters
TIMEZONE = 'America/Denver' # Olson timezone name for the site

# Analysis Parameters
MIN_TRACK_DAYS = 3       # Min distinct dates a track must appear for normalization
MIN_POINTS_FOR_SLOPE = 4 # Min data points in a Day/Night period to calculate slope

# --- Filtering Options ---
# NOTE: Hourly filtering is not possible with daily rainfall data.
# Daily Filter: Removes slope results for entire DAYS if rain occurred that day OR the previous day
FILTER_DAYS_AFTER_RAIN = True # Set to False to disable the daily filter
# --- End Filtering Options ---

# Plotting Options
# Limit the number of daily detail plots generated (set to None for all)
MAX_DAILY_PLOTS = 5 # e.g., only plot the first 5 days with valid filtered results


# --- Libraries for Day/Night Calculation ---
try:
    from astral.sun import sun
    from astral.location import LocationInfo
    import pytz
    astral_loaded = True
    local_tz = pytz.timezone(TIMEZONE)
except ImportError:
    print("Error: 'astral' or 'pytz' library not found. Install: pip install astral pytz")
    print("Day/Night analysis and local time conversions will be limited.")
    astral_loaded = False
    local_tz = pytz.utc # Fallback

# --- Data Loading ---

# GNSS Column names
gnss_column_names = [
    "Year", "DOY", "Hour", "Phase", "Nv", "Azimuth", "Sat", "Ampl",
    "emin", "emax", "DelT", "aprioriRH", "freq", "estRH", "pk2noise", "LSPAmp"
]

# Initialize list for GNSS data
dfs = []

print("--- Loading GNSS Data ---")
print(f"Station ID: {STATION_ID}")
# Loop through specified years/days for GNSS data
for year in YEARS_TO_PROCESS:
    print(f"Processing Year: {year}")
    days_in_year = 366 if calendar.isleap(year) else 365
    for doy in range(1, days_in_year + 1):
        file_path = os.path.join(base_path, str(year), 'phase', STATION_ID, f"{doy:03d}.txt")
        try:
            df = pd.read_csv(file_path, sep=r'\s+', skiprows=2, names=gnss_column_names, engine='python')
            if not df.empty:
                 dfs.append(df)
                 if doy % 50 == 0 or doy == 1 or doy == days_in_year :
                      print(f"  Loaded DOY {doy:03d} ({len(df)} rows)")
        except FileNotFoundError:
             pass # Silently ignore missing days
        except pd.errors.EmptyDataError:
             print(f"  Warning: File is empty {file_path} (Skipping)")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

# --- Daily Rainfall Data Loading ---
print("\n--- Loading Daily Rainfall Data ---")
rainfall_loaded = False
rainfall_df = None
try:
    rainfall_df = pd.read_csv(rainfall_path)
    if 'DateTime' not in rainfall_df.columns or 'Precipitation' not in rainfall_df.columns:
         raise ValueError("Rainfall CSV must contain 'DateTime' and 'Precipitation' columns.")

    # Check if Ensemble_ET exists for plotting later
    has_et_data = 'Ensemble_ET' in rainfall_df.columns
    if not has_et_data:
         print("Warning: 'Ensemble_ET' column not found in rainfall file. ET plot will be skipped.")

    rainfall_df.rename(columns={'DateTime': 'Timestamp_Local', 'Precipitation': 'Rainfall_mm'}, inplace=True)
    rainfall_df['Timestamp_Local'] = pd.to_datetime(rainfall_df['Timestamp_Local'])

    if astral_loaded:
         rainfall_df['Timestamp_Local'] = rainfall_df['Timestamp_Local'].dt.tz_localize(local_tz)
         print(f"Successfully loaded and localized daily rainfall data: {len(rainfall_df)} days")
    else:
         print(f"Successfully loaded daily rainfall data: {len(rainfall_df)} days (Warning: Could not localize timezone)")

    # Check date range coverage
    min_gnss_date_dt_naive = datetime(min(YEARS_TO_PROCESS), 1, 1)
    last_day_of_period_naive = datetime(max(YEARS_TO_PROCESS), 12, 31)

    if astral_loaded: # Make comparison dates aware if possible
         min_gnss_date_dt = local_tz.localize(min_gnss_date_dt_naive)
         max_gnss_date_dt_endofday = local_tz.localize(last_day_of_period_naive)
         min_rain_date = rainfall_df['Timestamp_Local'].min() # Already localized
         max_rain_date = rainfall_df['Timestamp_Local'].max() # Already localized
    else: # Compare naive if localization failed
         min_gnss_date_dt = min_gnss_date_dt_naive
         max_gnss_date_dt_endofday = last_day_of_period_naive
         # Convert rain dates to naive for comparison IF they were successfully loaded but not localized
         if 'Timestamp_Local' in rainfall_df.columns:
              min_rain_date = rainfall_df['Timestamp_Local'].min().replace(tzinfo=None)
              max_rain_date = rainfall_df['Timestamp_Local'].max().replace(tzinfo=None)
         else: # Should not happen if loaded, but safety check
              min_rain_date = max_gnss_date_dt_endofday + timedelta(days=1) # Ensure check fails
              max_rain_date = min_gnss_date_dt - timedelta(days=1)


    if min_rain_date > min_gnss_date_dt or max_rain_date < max_gnss_date_dt_endofday:
         print(f"Warning: Rainfall data range ({min_rain_date.date()} to {max_rain_date.date()}) "
               f"may not fully cover the GNSS processing period ({min(YEARS_TO_PROCESS)} to {max(YEARS_TO_PROCESS)}).")
         print("Daily rainfall filtering might be incomplete.")
    rainfall_loaded = True

except FileNotFoundError:
    print(f"Error: Rainfall file not found at {rainfall_path}")
    FILTER_DAYS_AFTER_RAIN = False; print("Daily rainfall filtering disabled.")
except Exception as e:
    print(f"Error loading or processing rainfall data: {e}")
    FILTER_DAYS_AFTER_RAIN = False; print("Daily rainfall filtering disabled.")


# --- Calculate Daily Rainfall Summary (for Daily Filter) ---
print("\n--- Pre-calculating Daily Rainfall Summary (Local Time) ---")
rainy_days_set = set()
if FILTER_DAYS_AFTER_RAIN and rainfall_loaded:
    try:
        daily_summary_df = rainfall_df[['Timestamp_Local', 'Rainfall_mm']].copy()
        daily_summary_df['Date_Local'] = daily_summary_df['Timestamp_Local'].dt.date
        daily_summary_df = daily_summary_df.sort_values(by='Date_Local')
        daily_summary_df['Rain_on_Day'] = daily_summary_df['Rainfall_mm'] > 0.0
        daily_summary_df['Rain_Today_Or_Yesterday'] = daily_summary_df['Rain_on_Day'] \
                                                            .rolling(window=2, min_periods=1) \
                                                            .max().astype(bool)
        rainy_days_set = set(daily_summary_df[daily_summary_df['Rain_Today_Or_Yesterday']]['Date_Local'])
        print(f"Identified {len(rainy_days_set)} local dates to potentially exclude due to recent rain.")
    except Exception as e:
        print(f"Error calculating daily rainfall summary: {e}. Disabling daily rain filter.")
        FILTER_DAYS_AFTER_RAIN = False
elif not rainfall_loaded:
     print("Skipping daily rainfall calculation as rainfall data was not loaded.")
     FILTER_DAYS_AFTER_RAIN = False


# --- Initial GNSS Data Processing ---
if not dfs:
    print("\nNo GNSS data loaded. Exiting.")
    exit()

combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined GNSS data: {len(combined_df)} total measurements.")

# 1. Create datetime_utc column
def create_datetime(row):
    try:
        year = int(row['Year']); doy = int(row['DOY']); hour_fraction = row['Hour']
        if year not in YEARS_TO_PROCESS: return pd.NaT
        if not (1 <= doy <= (366 if calendar.isleap(year) else 365)): return pd.NaT
        base_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        time_delta = timedelta(hours=hour_fraction)
        return base_date + time_delta
    except Exception: return pd.NaT
combined_df['datetime_utc'] = combined_df.apply(create_datetime, axis=1)
original_rows = len(combined_df)
combined_df.dropna(subset=['datetime_utc'], inplace=True)
if len(combined_df) < original_rows:
     print(f"Warning: Removed {original_rows - len(combined_df)} rows due to errors in datetime conversion.")
if combined_df.empty: print("Error: No valid GNSS data remaining. Exiting."); exit()
combined_df['datetime_utc'] = combined_df['datetime_utc'].dt.tz_localize('UTC')

# 2. Wrap Phase
combined_df['Phase_Wrapped'] = combined_df['Phase'] % 360

# 3. Azimuth Bins
def assign_azimuth_bin(azimuth):
    azimuth = azimuth % 360; return int(azimuth // 30) * 30
combined_df['Azimuth_Bin'] = combined_df['Azimuth'].apply(assign_azimuth_bin)

# 4. Arc Type
combined_df['Arc_Type'] = np.where((combined_df['Azimuth_Bin'] >= 0) & (combined_df['Azimuth_Bin'] < 180), 'Rising', 'Setting')
combined_df.loc[combined_df['Azimuth_Bin'] >= 360, 'Arc_Type'] = 'Unknown'

# 5. Sat_Arc Identifier
combined_df['Sat_Arc'] = combined_df['Sat'].astype(str) + '_Az' + combined_df['Azimuth_Bin'].astype(str) + '_' + combined_df['Arc_Type']

# 6. Track Eligibility
combined_df['Date_For_Check'] = combined_df['datetime_utc'].dt.date
track_dates = combined_df.groupby('Sat_Arc')['Date_For_Check'].nunique()
eligible_tracks = track_dates[track_dates >= MIN_TRACK_DAYS].index.tolist()
all_tracks = combined_df['Sat_Arc'].unique().tolist()
print(f"\nTotal unique satellite tracks identified: {len(all_tracks)}")
print(f"Tracks present on at least {MIN_TRACK_DAYS} different dates: {len(eligible_tracks)}")
if not eligible_tracks:
    print(f"Warning: No tracks met the {MIN_TRACK_DAYS}-date requirement.")
    eligible_tracks = all_tracks

# --- Phase Normalization ---
def circular_mean(angles):
    angles_rad = np.deg2rad(angles.dropna())
    if len(angles_rad) == 0: return np.nan
    mean_sin = np.mean(np.sin(angles_rad)); mean_cos = np.mean(np.cos(angles_rad))
    return np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360

print("\n--- Normalizing Phase Data per Track ---")
all_normalized_data = []
processed_tracks_count = 0
for track_id in eligible_tracks:
    track_data = combined_df[combined_df['Sat_Arc'] == track_id].copy()
    if not track_data.empty:
        mean_phase = circular_mean(track_data['Phase_Wrapped'])
        if not np.isnan(mean_phase):
             track_data['Phase_Normalized'] = [(phase - mean_phase + 180) % 360 - 180 for phase in track_data['Phase_Wrapped']]
             all_normalized_data.append(track_data)
             processed_tracks_count += 1

if not all_normalized_data: print("Error: No normalized data could be generated."); exit()

normalized_df = pd.concat(all_normalized_data, ignore_index=True)
normalized_df = normalized_df.sort_values(by='datetime_utc')
print(f"Generated normalized phase data for {processed_tracks_count} tracks.")

# --- Day/Night Period Calculation ---
print("\n--- Calculating Day/Night Periods ---")
if astral_loaded:
    site_location = LocationInfo(f"Site_{STATION_ID}", "Region", TIMEZONE, SITE_LATITUDE, SITE_LONGITUDE)
    def get_day_night(utc_timestamp, location):
        if pd.isna(utc_timestamp): return 'Unknown'
        local_timestamp = utc_timestamp.astimezone(local_tz)
        try:
            s = sun(location.observer, date=local_timestamp.date(), tzinfo=local_tz)
            if s['sunrise'] <= local_timestamp < s['sunset']: return 'Day'
            else: return 'Night'
        except Exception: return 'Unknown'
    normalized_df['Time_Period'] = normalized_df['datetime_utc'].apply(lambda dt: get_day_night(dt, site_location))
    print("Added 'Time_Period' column based on sunrise/sunset.")
    print(normalized_df['Time_Period'].value_counts())
else:
    normalized_df['Time_Period'] = 'Unknown'
    print("Skipped Day/Night calculation (astral/pytz missing).")

# --- No Hourly Filter Applied ---
print("\n--- Skipping Hourly Rainfall Filter (Daily data provided) ---")
# Data for slope calculation is the full normalized dataset
analysis_df = normalized_df.copy()

# --- Rate of Change (Slope) Calculation ---
print("\n--- Calculating Rate of Change (Slope) per Day/Night Period ---")
if analysis_df.empty: print("Error: No data available for slope calculation."); exit()

start_time_analysis = analysis_df['datetime_utc'].min()
analysis_df['Time_Hours_Since_Start'] = (analysis_df['datetime_utc'] - start_time_analysis).dt.total_seconds() / 3600.0
try: analysis_df['Date_Local'] = analysis_df['datetime_utc'].dt.tz_convert(local_tz).dt.date
except TypeError: print("Warning: Could not convert datetime to local tz. Using UTC date."); analysis_df['Date_Local'] = analysis_df['datetime_utc'].dt.date

slope_results = []
if 'Date_Local' not in analysis_df.columns or 'Time_Period' not in analysis_df.columns or analysis_df[['Date_Local', 'Time_Period']].isna().all().all():
     print("Error: Cannot group for slope calculation (Missing Date_Local/Time_Period)."); exit()

grouped = analysis_df.groupby(['Date_Local', 'Time_Period'])
for name, group in grouped:
    date_local, period = name
    if period == 'Unknown': continue
    if len(group) >= MIN_POINTS_FOR_SLOPE:
        group_clean = group.dropna(subset=['Time_Hours_Since_Start', 'Phase_Normalized'])
        if len(group_clean) >= MIN_POINTS_FOR_SLOPE:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    group_clean['Time_Hours_Since_Start'], group_clean['Phase_Normalized']
                )
                slope_results.append({'Date': date_local, 'Period': period, 'Slope_Deg_per_Hour': slope,
                                      'Intercept': intercept, 'R_squared': r_value**2, 'P_value': p_value,
                                      'Std_Err': std_err, 'N_points': len(group_clean),
                                      'Start_Time_UTC': group_clean['datetime_utc'].min(),
                                      'End_Time_UTC': group_clean['datetime_utc'].max()})
            except Exception as e: slope_results.append({'Date': date_local, 'Period': period, 'Slope_Deg_per_Hour': np.nan, 'N_points': len(group)})
        else: slope_results.append({'Date': date_local, 'Period': period, 'Slope_Deg_per_Hour': np.nan, 'N_points': len(group)})
    else: slope_results.append({'Date': date_local, 'Period': period, 'Slope_Deg_per_Hour': np.nan, 'N_points': len(group),
                                 'Start_Time_UTC': group['datetime_utc'].min() if not group.empty else pd.NaT,
                                 'End_Time_UTC': group['datetime_utc'].max() if not group.empty else pd.NaT})

if not slope_results: print("Error: No slope results generated."); exit()
slope_results_df = pd.DataFrame(slope_results)
slope_results_df.dropna(subset=['Slope_Deg_per_Hour'], inplace=True)
slope_results_df = slope_results_df.sort_values(by=['Date', 'Period'])
print(f"Calculated valid slopes for {len(slope_results_df)} periods (before daily rain filter).")

# --- Apply Daily Rain Filter ---
print("\n--- Applying Daily Rainfall Filter (Days with recent rain) ---")
if FILTER_DAYS_AFTER_RAIN and rainfall_loaded:
    if not rainy_days_set:
        print("No days identified for exclusion by the daily filter.")
        final_slope_results_df = slope_results_df.copy()
    else:
        original_count = len(slope_results_df)
        try:
             if not slope_results_df.empty and not isinstance(slope_results_df['Date'].iloc[0], date):
                  slope_results_df['Date'] = pd.to_datetime(slope_results_df['Date']).dt.date
             final_slope_results_df = slope_results_df[~slope_results_df['Date'].isin(rainy_days_set)].copy()
             removed_count = original_count - len(final_slope_results_df)
             print(f"Removed {removed_count} slope results from {len(rainy_days_set)} dates due to recent rain.")
             print(f"Slope results remaining after daily filter: {len(final_slope_results_df)}")
        except IndexError:
             print("Slope results dataframe is empty, cannot apply daily filter.")
             final_slope_results_df = slope_results_df.copy()
else:
    final_slope_results_df = slope_results_df.copy()
    if not FILTER_DAYS_AFTER_RAIN: print("Daily rainfall filtering is disabled.")
    elif not rainfall_loaded: print("Daily rainfall filtering skipped (rainfall data not loaded).")

# --- Statistical Comparison and Visualization ---
if final_slope_results_df.empty:
    print("\nNo valid slope data available for comparison after filtering.")
else:
    print("\n--- Comparing Day vs. Night Slopes (Post-Filtering) ---")
    day_slopes = final_slope_results_df[final_slope_results_df['Period'] == 'Day']['Slope_Deg_per_Hour']
    night_slopes = final_slope_results_df[final_slope_results_df['Period'] == 'Night']['Slope_Deg_per_Hour']

    if day_slopes.empty or night_slopes.empty:
        print("Insufficient Day or Night slope data remaining for statistical comparison.")
    else:
        # T-test and interpretation...
        t_stat, p_val_ttest = stats.ttest_ind(day_slopes, night_slopes, equal_var=False, nan_policy='omit')
        print(f"Mean Day Slope:   {day_slopes.mean():.4f} +/- {day_slopes.std():.4f} deg/hr (N={len(day_slopes)})")
        print(f"Mean Night Slope: {night_slopes.mean():.4f} +/- {night_slopes.std():.4f} deg/hr (N={len(night_slopes)})")
        print(f"\nWelch's t-test results: T={t_stat:.4f}, p={p_val_ttest:.4f}")
        alpha = 0.05
        if p_val_ttest < alpha:
            print(f"Difference is statistically significant (p < {alpha}).")
            if day_slopes.mean() < night_slopes.mean(): print("Interpretation: Drying faster during Day.")
            else: print("Interpretation: Drying slower (or wetting more) during Day.")
        else: print(f"Difference is not statistically significant (p >= {alpha}).")

        # Plotting setup
        plot_title_suffix = "(Filters: "
        filters_active = []
        if FILTER_DAYS_AFTER_RAIN: filters_active.append("Daily>24h")
        if not filters_active: plot_title_suffix += "None)"
        else: plot_title_suffix += ", ".join(filters_active) + ")"
        years_str = "-".join(map(str, sorted(YEARS_TO_PROCESS))) if len(YEARS_TO_PROCESS) > 1 else str(YEARS_TO_PROCESS[0])

        # Box Plot
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=final_slope_results_df, x='Period', y='Slope_Deg_per_Hour', hue='Period',
                    order=['Day', 'Night'], palette=['#FFC107', '#3F51B5'], legend=False)
        sns.stripplot(data=final_slope_results_df, x='Period', y='Slope_Deg_per_Hour', order=['Day', 'Night'], color='black', alpha=0.3, size=4)
        plt.title(f'Distribution of Phase Slopes: Day vs Night\nSite: {STATION_ID} ({years_str}) {plot_title_suffix}', fontsize=14)
        plt.ylabel('Slope (Phase Deviation degrees / hour)', fontsize=12); plt.xlabel('Time Period', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        if not day_slopes.empty and not night_slopes.empty:
             plt.text(0.95, 0.05, f'p-value = {p_val_ttest:.3f}', transform=plt.gca().transAxes, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        plt.tight_layout(); plt.show()

        # Time Series Plot (with Ensemble ET if available)
        plt.figure(figsize=(14, 7))
        ax1 = plt.gca() # Primary axis for slopes
        plot_df = final_slope_results_df.copy()

        if 'Start_Time_UTC' in plot_df.columns and not plot_df['Start_Time_UTC'].isna().all() and astral_loaded:
             plot_df['Start_Time_Local'] = plot_df['Start_Time_UTC'].dt.tz_convert(local_tz)
             day_data = plot_df[plot_df['Period'] == 'Day']
             line1, = ax1.plot(day_data['Start_Time_Local'], day_data['Slope_Deg_per_Hour'], marker='o', linestyle='-', color='#FFC107', markersize=5, label='Day Slope')
             night_data = plot_df[plot_df['Period'] == 'Night']
             line2, = ax1.plot(night_data['Start_Time_Local'], night_data['Slope_Deg_per_Hour'], marker='s', linestyle='-', color='#3F51B5', markersize=5, label='Night Slope')

             ax1.set_title(f'Day vs Night Slopes & Daily ET Over Time\nSite: {STATION_ID} ({years_str}) {plot_title_suffix}', fontsize=14)
             ax1.set_ylabel('Slope (Phase Deviation degrees / hour)', fontsize=12, color='black')
             ax1.set_xlabel('Date (Local Time)', fontsize=12)
             ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
             ax1.tick_params(axis='y', labelcolor='black')
             ax1.set_ylim(-1, 1) # Set y-limit for slopes
             ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d', tz=local_tz))
             plt.gcf().autofmt_xdate()

             # Add Ensemble ET on secondary axis if available
             if rainfall_loaded and has_et_data:
                  ax2 = ax1.twinx()
                  et_plot_data = rainfall_df[['Timestamp_Local', 'Ensemble_ET']].copy()
                  et_plot_data = et_plot_data.sort_values(by='Timestamp_Local')
                  et_plot_data = et_plot_data[et_plot_data['Timestamp_Local'].dt.year.isin(YEARS_TO_PROCESS)]
                  et_color = 'green'
                  line3, = ax2.plot(et_plot_data['Timestamp_Local'], et_plot_data['Ensemble_ET'], color=et_color, linestyle=':', marker='.', markersize=4, label='Daily ET')
                  ax2.set_ylabel('Ensemble ET (mm/day?)', color=et_color, fontsize=12)
                  ax2.tick_params(axis='y', labelcolor=et_color)
                  ax2.set_ylim(bottom=0); ax2.grid(False)
                  # Combine legends
                  lines = [line1, line2, line3]; labels = [l.get_label() for l in lines]
                  ax1.legend(lines, labels, loc='best', fontsize=10)
             else: ax1.legend(fontsize=10) # Legend for slopes only

             plt.tight_layout(); plt.show()
        else: print("Skipping time series plot (missing start time data or timezone library).")

        # --- [REVISED v3] Detailed Daily Fit Plotting (Single Plot, Full Day + Night Period) ---
        print("\n--- Generating Detailed Daily Fit Plots (Full Day + Following Night) ---")

        # Prepare lookup for slope/intercept results
        try:
            if not final_slope_results_df.empty and not isinstance(final_slope_results_df['Date'].iloc[0], date):
                final_slope_results_df['Date'] = pd.to_datetime(final_slope_results_df['Date']).dt.date
            fit_lookup = final_slope_results_df.set_index(['Date', 'Period'])
        except (KeyError, IndexError):
            print("Warning: Could not set index on slope results. Skipping daily plots.")
            fit_lookup = None

        # Get unique dates to plot from filtered results
        if 'Date' in final_slope_results_df.columns and not final_slope_results_df.empty:
            dates_to_plot = sorted(final_slope_results_df['Date'].unique())
        else:
            dates_to_plot = []

        if not dates_to_plot:
            print("No dates with valid slope results to generate daily plots.")
        elif fit_lookup is None:
            print("Skipping daily plots due to lookup issue.")
        elif not astral_loaded:
            print("Skipping daily plots (astral/pytz needed).")
        else:
            output_plot_dir = f"./{STATION_ID}_daily_fits_full_period_{years_str}"
            os.makedirs(output_plot_dir, exist_ok=True)
            print(f"Saving daily plots to: {output_plot_dir} (Limit: {MAX_DAILY_PLOTS})")
            plot_count = 0
            site_info = LocationInfo(f"Site_{STATION_ID}", "Region", TIMEZONE, SITE_LATITUDE, SITE_LONGITUDE)

            # Ensure analysis_df['Date_Local'] is available if needed elsewhere, though filtering uses UTC
            if 'Date_Local' not in analysis_df.columns:
                try:
                    analysis_df['Date_Local'] = analysis_df['datetime_utc'].dt.tz_convert(local_tz).dt.date
                except TypeError:
                    print("Warn: Could not create Date_Local column reliably.")

            for current_date in dates_to_plot:
                if MAX_DAILY_PLOTS is not None and plot_count >= MAX_DAILY_PLOTS:
                    print(f"Reached plot limit ({MAX_DAILY_PLOTS}).");
                    break

                try:  # Calculate precise period boundaries for the plot window and shading
                    # Window starts at sunrise today
                    sun_today = sun(site_info.observer, date=current_date, tzinfo=local_tz)
                    plot_start_local = sun_today['sunrise']
                    sunset_today_local = sun_today['sunset']  # Needed for shading Day/Night split

                    # Window ends at sunrise tomorrow (defines end of relevant Night period)
                    next_date = current_date + timedelta(days=1)
                    sun_next_day = sun(site_info.observer, date=next_date, tzinfo=local_tz)
                    plot_end_local = sun_next_day['sunrise']  # This is also the end of Night period

                    # Get UTC boundaries for filtering the main dataframe
                    plot_start_utc = plot_start_local.astimezone(pytz.utc)
                    plot_end_utc = plot_end_local.astimezone(pytz.utc)
                    sunset_today_utc = sunset_today_local.astimezone(pytz.utc)  # For filtering line segments

                except Exception as e:
                    print(f"Warn: Skip plot for {current_date}: Sun calc error: {e}"); continue

                # Select ALL data points within the plot window (Sunrise D to Sunrise D+1)
                plot_data = analysis_df[
                    (analysis_df['datetime_utc'] >= plot_start_utc) &
                    (analysis_df['datetime_utc'] < plot_end_utc)
                    ].copy()

                if plot_data.empty: continue  # Skip if no data in this specific window

                # Convert time for plotting axis
                plot_data['datetime_local'] = plot_data['datetime_utc'].dt.tz_convert(local_tz)

                # Create figure
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                # Title refers to the primary date for which the Day/Night fits were calculated
                fig.suptitle(f"Daily Phase Fit (Day {current_date.strftime('%Y-%m-%d')} + Following Night)",
                             fontsize=14)

                # Plot all points in the window
                ax.scatter(plot_data['datetime_local'], plot_data['Phase_Normalized'],
                           label='Data Points', alpha=0.6, s=15, c='grey', zorder=2)

                # Background shading: Day = sunrise to sunset; Night = sunset to next sunrise
                ax.axvspan(plot_start_local, sunset_today_local, facecolor='gold', alpha=0.2, zorder=1,
                           label='Day Period')
                ax.axvspan(sunset_today_local, plot_end_local, facecolor='navy', alpha=0.15, zorder=1,
                           label='Night Period')

                lines_for_legend = []

                # Plot Day Fit Line (Sunrise D to Sunset D)
                period = 'Day'
                try:
                    fit_result = fit_lookup.loc[(current_date, period)]
                    slope, intercept, n_points = fit_result['Slope_Deg_per_Hour'], fit_result['Intercept'], fit_result[
                        'N_points']
                    # Select points actually within the Day period boundary
                    day_points_in_plot = plot_data[
                        (plot_data['datetime_utc'] >= plot_start_utc) &  # >= sunrise today
                        (plot_data['datetime_utc'] < sunset_today_utc)  # < sunset today
                        ]
                    if not day_points_in_plot.empty:
                        x_hours = day_points_in_plot['Time_Hours_Since_Start']
                        y_fit = slope * x_hours + intercept
                        line_day, = ax.plot(day_points_in_plot['datetime_local'], y_fit, color='orange', linewidth=2.5,
                                            label=f'Day Fit (S={slope:.3f})', zorder=3)
                        lines_for_legend.append(line_day)
                except KeyError:
                    pass  # No Day fit for this date

                # Plot Night Fit Line (Sunset D to Sunrise D+1)
                period = 'Night'
                try:
                    fit_result = fit_lookup.loc[(current_date, period)]  # Fit associated with current_date
                    slope, intercept, n_points = fit_result['Slope_Deg_per_Hour'], fit_result['Intercept'], fit_result[
                        'N_points']
                    # Select points actually within the Night period boundary
                    night_points_in_plot = plot_data[
                        (plot_data['datetime_utc'] >= sunset_today_utc) &  # >= sunset today
                        (plot_data['datetime_utc'] < plot_end_utc)  # < sunrise tomorrow
                        ]
                    if not night_points_in_plot.empty:
                        x_hours = night_points_in_plot['Time_Hours_Since_Start']
                        y_fit = slope * x_hours + intercept
                        line_night, = ax.plot(night_points_in_plot['datetime_local'], y_fit, color='blue',
                                              linewidth=2.5,
                                              label=f'Night Fit (S={slope:.3f})', zorder=3)
                        lines_for_legend.append(line_night)
                except KeyError:
                    pass  # No Night fit for this date

                # Final Plot Formatting
                ax.set_ylabel('Norm. Phase (deg)', fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.6)

                # Manual legend combining lines and shading proxies
                day_patch = plt.Rectangle((0, 0), 1, 1, fc='gold', alpha=0.2)
                night_patch = plt.Rectangle((0, 0), 1, 1, fc='navy', alpha=0.15)
                combined_handles = lines_for_legend + [day_patch, night_patch]
                combined_labels = [h.get_label() for h in lines_for_legend] + ['Day Period', 'Night Period']
                ax.legend(handles=combined_handles, labels=combined_labels, fontsize='small', loc='best')

                # Format x-axis to show date and time if window > 24h, or just time if appropriate
                ax.xaxis.set_major_formatter(DateFormatter('%H:%M\n%Y-%m-%d', tz=local_tz))
                ax.set_xlabel(f'Time ({local_tz.zone})', fontsize=12)
                # Set x-axis limits explicitly to the calculated window
                ax.set_xlim(plot_start_local, plot_end_local)
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")  # Rotate labels for better fit

                # Save or show
                plot_filename = os.path.join(output_plot_dir,
                                             f"daily_fit_full_{STATION_ID}_{current_date.strftime('%Y-%m-%d')}.png")
                try:
                    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust margins
                    plt.savefig(plot_filename, dpi=150);
                    plt.close(fig);
                    plot_count += 1
                except Exception as e:
                    print(f"Error saving plot {plot_filename}: {e}"); plt.close(fig)

    # --- End of Detailed Daily Fit Plotting ---

# Final completion message
print("\n--- Analysis Complete ---")