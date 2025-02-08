from skyfield.api import load, wgs84
from datetime import datetime, timedelta
import pytz
import pyperclip

# Load GPS satellites only
satellites = load.tle_file('https://celestrak.org/NORAD/elements/gps-ops.txt')
ts = load.timescale()

# Newcastle, Australia coordinates
location = wgs84.latlon(-32.888024, 151.707553, elevation_m=0)

# Set time range - Sydney timezone
sydney_tz = pytz.timezone('Australia/Sydney')
start_time = sydney_tz.localize(datetime(2025, 2, 7, 6, 0, 0))
end_time = sydney_tz.localize(datetime(2025, 2, 7, 18, 0, 0))

# Convert to UTC for calculations
start_time_utc = start_time.astimezone(pytz.UTC)
end_time_utc = end_time.astimezone(pytz.UTC)

# Create time range with 1-minute intervals
time_range = ts.linspace(
    ts.from_datetime(start_time_utc),
    ts.from_datetime(end_time_utc),
    720  # 12 hours * 60 minutes
)


def is_northern_azimuth(az):
    return 270 <= az <= 360 or 0 <= az <= 90


results = []
for sat in satellites:
    # Calculate satellite positions relative to observer
    difference = sat - location
    topocentric = difference.at(time_range)
    alt, az, distance = topocentric.altaz()

    # Find rising and setting events
    for i in range(len(time_range) - 1):
        # Rising event
        if alt.degrees[i] < 0 and alt.degrees[i + 1] >= 0:
            if is_northern_azimuth(az.degrees[i]):
                event_time = time_range[i].utc_datetime()
                sydney_time = event_time.astimezone(sydney_tz)
                results.append({
                    'satellite': sat.name,
                    'event': 'Rise',
                    'time': sydney_time,
                    'azimuth': az.degrees[i]
                })
        # Setting event
        elif alt.degrees[i] >= 0 and alt.degrees[i + 1] < 0:
            if is_northern_azimuth(az.degrees[i]):
                event_time = time_range[i].utc_datetime()
                sydney_time = event_time.astimezone(sydney_tz)
                results.append({
                    'satellite': sat.name,
                    'event': 'Set',
                    'time': sydney_time,
                    'azimuth': az.degrees[i]
                })

# Sort results by time
results.sort(key=lambda x: x['time'])

# Format for spreadsheet
spreadsheet_text = "Satellite\tEvent\tTime (AEDT)\tAzimuth (degrees)\n"
for event in results:
    spreadsheet_text += f"{event['satellite']}\t{event['event']}\t{event['time'].strftime('%H:%M')}\t{event['azimuth']:.1f}\n"

# Copy to clipboard
pyperclip.copy(spreadsheet_text)

# Display preview
print("Results copied to clipboard in spreadsheet format!")
print("\nPreview:")
print(spreadsheet_text)