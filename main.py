import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def plot_sea_level(filename):
    # Define column names
    columns = ['MJD', 'RH', 'year', 'month', 'day', 'hour', 'min', 'sec', 'sea_level']

    # Load the file with specified column names, skipping header rows
    df = pd.read_csv(filename, delimiter=r'\s+', skiprows=8, names=columns)

    # Create datetime objects from the date/time columns
    df['datetime'] = pd.to_datetime({
        'year': df['year'],
        'month': df['month'],
        'day': df['day'],
        'hour': df['hour'],
        'minute': df['min'],
        'second': df['sec']
    })

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['sea_level'], '-o')

    # Customize the plot
    plt.title('Sea Level Variation Over Time')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Sea Level (m)')
    plt.grid(True)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()

# Usage example:
plot_sea_level("/data/refl_code/Files/newe/newe_spline_out.txt")