import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Set the base directory
BASE_DIR = Path('/home/george/Documents/Work/snr_precision_analysis/vwc_comparison')


def read_vwc_file(filepath):
    """Read VWC file and return as DataFrame."""
    df = pd.read_csv(filepath, skiprows=3, delimiter=r'\s+',
                     names=['FracYr', 'Year', 'DOY', 'VWC', 'Month', 'Day'])
    # Convert VWC to float
    df['VWC'] = pd.to_numeric(df['VWC'], errors='coerce')
    return df


def read_phaseRH_file(filepath):
    """Read phaseRH file and return RefH values."""
    df = pd.read_csv(filepath, skiprows=5, delimiter=r'\s+',
                     names=['Track', 'RefH', 'SatNu', 'MeanAz', 'Nval', 'Az1', 'Az2'])
    # Convert RefH to float
    df['RefH'] = pd.to_numeric(df['RefH'], errors='coerce')
    return df['RefH']


def plot_vwc_comparison(original_vwc, quarter_vwc, integer_vwc):
    """Create plot comparing VWC values across datasets."""
    plt.figure(figsize=(12, 6))
    plt.plot(original_vwc['DOY'], original_vwc['VWC'], 'b-', label='Original', alpha=0.7)
    plt.plot(quarter_vwc['DOY'], quarter_vwc['VWC'], 'r--', label='Quarter', alpha=0.7)
    plt.plot(integer_vwc['DOY'], integer_vwc['VWC'], 'g:', label='Integer', alpha=0.7)

    plt.xlim(0,365)

    plt.xlabel('Day of Year')
    plt.ylabel('VWC')
    plt.title('VWC Comparison Across Datasets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(BASE_DIR / 'vwc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_histogram_with_normal(errors_quarter, errors_integer, title, filename):
    """Create histogram of errors with fitted normal distributions."""
    plt.figure(figsize=(10, 6))

    # Clean data
    quarter_clean = errors_quarter.dropna()
    integer_clean = errors_integer.dropna()

    # Fit normal distributions
    quarter_mean, quarter_std = stats.norm.fit(quarter_clean)
    integer_mean, integer_std = stats.norm.fit(integer_clean)

    # Create bins
    all_data = np.concatenate([quarter_clean, integer_clean])
    bins = np.linspace(np.min(all_data), np.max(all_data), 50)

    # Plot histograms with counts (not density)
    plt.hist(quarter_clean, bins=bins, alpha=0.5, label='Quarter Error',
             color='red')
    plt.hist(integer_clean, bins=bins, alpha=0.5, label='Integer Error',
             color='green')

    # Get bin width and total counts for scaling
    bin_width = bins[1] - bins[0]
    n_quarter = len(quarter_clean)
    n_integer = len(integer_clean)

    # Plot fitted normal distributions
    x = np.linspace(np.min(all_data), np.max(all_data), 100)
    quarter_normal = stats.norm.pdf(x, quarter_mean, quarter_std)
    integer_normal = stats.norm.pdf(x, integer_mean, integer_std)

    # Scale normal curves to match histogram area
    # Area = bin_width * height * n_samples
    quarter_normal = quarter_normal * bin_width * n_quarter
    integer_normal = integer_normal * bin_width * n_integer

    plt.plot(x, quarter_normal, 'r-', lw=2,
             label=f'Quarter Normal Fit\nμ={quarter_mean:.6f}, σ={quarter_std:.6f}')
    plt.plot(x, integer_normal, 'g-', lw=2,
             label=f'Integer Normal Fit\nμ={integer_mean:.6f}, σ={integer_std:.6f}')

    plt.xlabel('Error')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Perform Shapiro-Wilk test for normality
    quarter_stat, quarter_p = stats.shapiro(quarter_clean)
    integer_stat, integer_p = stats.shapiro(integer_clean)

    # Add Shapiro-Wilk test results as text
    plt.text(0.02, 0.98, f'Shapiro-Wilk p-values:\nQuarter: {quarter_p:.6f}\nInteger: {integer_p:.6f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.savefig(BASE_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Read VWC files
    original_vwc = read_vwc_file(BASE_DIR / 'original' / 'p038_vwc.txt')
    quarter_vwc = read_vwc_file(BASE_DIR / 'quarter' / 'p038_vwc.txt')
    integer_vwc = read_vwc_file(BASE_DIR / 'integer' / 'p038_vwc.txt')

    # Plot 1: VWC comparison
    plot_vwc_comparison(original_vwc, quarter_vwc, integer_vwc)

    # Calculate VWC errors
    vwc_errors_quarter = quarter_vwc['VWC'] - original_vwc['VWC']
    vwc_errors_integer = integer_vwc['VWC'] - original_vwc['VWC']

    # Plot 2: VWC errors histogram
    plot_error_histogram_with_normal(
        vwc_errors_quarter,
        vwc_errors_integer,
        'VWC Errors Distribution with Normal Fits',
        'vwc_errors_histogram.png'
    )

    # Read phaseRH files and calculate errors
    original_phase = read_phaseRH_file(BASE_DIR / 'original' / 'p038_phaseRH.txt')
    quarter_phase = read_phaseRH_file(BASE_DIR / 'quarter' / 'p038_phaseRH.txt')
    integer_phase = read_phaseRH_file(BASE_DIR / 'integer' / 'p038_phaseRH.txt')

    phase_errors_quarter = quarter_phase - original_phase
    phase_errors_integer = integer_phase - original_phase

    # Plot 3: PhaseRH errors histogram
    plot_error_histogram_with_normal(
        phase_errors_quarter,
        phase_errors_integer,
        'PhaseRH Errors Distribution with Normal Fits',
        'phaseRH_errors_histogram.png'
    )

    # Print summary statistics
    print("\nVWC Error Statistics:")
    print(f"Quarter - Mean: {vwc_errors_quarter.mean():.6f}, Std: {vwc_errors_quarter.std():.6f}")
    print(f"Integer - Mean: {vwc_errors_integer.mean():.6f}, Std: {vwc_errors_integer.std():.6f}")

    print("\nPhaseRH Error Statistics:")
    print(f"Quarter - Mean: {phase_errors_quarter.mean():.6f}, Std: {phase_errors_quarter.std():.6f}")
    print(f"Integer - Mean: {phase_errors_integer.mean():.6f}, Std: {phase_errors_integer.std():.6f}")


if __name__ == "__main__":
    main()