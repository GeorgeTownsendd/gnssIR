import numpy as np
from scipy.signal import lombscargle
from LevenbergEstimator import GNSSMultipathEstimator
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def estimate_height_lsp(elevation_angles, dsnr_values, wavelength):
    """
    Estimate reflector height using LSP and fit amplitude/phase using least squares.

    Parameters:
        elevation_angles (ndarray): Elevation angles in radians
        dsnr_values (ndarray): Detrended SNR values
        wavelength (float): GNSS wavelength in meters

    Returns:
        tuple: (estimated_height, model_predictions, amplitude, phase)
    """
    sin_elv = np.sin(elevation_angles)

    # Create array of candidate reflector heights
    heights = np.linspace(0.5, 8, 1000)

    # Convert heights to frequencies using f = 4πH/λ
    frequencies = 4 * np.pi * heights / wavelength

    # Compute LSP
    pgram = lombscargle(sin_elv, dsnr_values, frequencies)

    # Find height corresponding to maximum power
    best_height = heights[np.argmax(pgram)]
    logger.info(f"LSP best height: {best_height:.3f}m")

    # Compute angular frequency from height
    omega = 4 * np.pi * best_height / wavelength

    # Set up design matrix for least squares
    # Model: A*sin(ωx + φ) = As*sin(ωx) + Ac*cos(ωx)
    A = np.column_stack([
        np.sin(omega * sin_elv),
        np.cos(omega * sin_elv)
    ])

    # Solve least squares problem
    x = np.linalg.solve(A.T @ A, A.T @ dsnr_values)
    As, Ac = x

    # Compute amplitude and phase
    amplitude = np.sqrt(As ** 2 + Ac ** 2)
    phase = np.arctan2(Ac, As)

    # Generate model predictions
    model_predictions = A @ x

    logger.info(f"LSP Amplitude: {amplitude:.3f}")
    logger.info(f"LSP Phase: {phase:.3f} rad")

    return best_height, model_predictions, amplitude, phase


def compute_sse(data, predictions):
    """
    Compute Sum of Squared Errors.
    """
    return np.sum((data - predictions) ** 2)


def plot_comparison(sin_theta, dsnr_values, lm_predictions, lsp_predictions,
                    lm_height, lsp_height, save_path=None):
    """
    Create comparison plot showing both model fits and residuals.
    """
    # Compute SSE for both methods
    lm_sse = compute_sse(dsnr_values, lm_predictions)
    lsp_sse = compute_sse(dsnr_values, lsp_predictions)

    # Create figure with two vertically stacked subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1],
                                   sharex=True)

    # Top subplot: Data and both model fits
    ax1.scatter(sin_theta, dsnr_values, c='gray', alpha=0.5, s=20, label='Data')
    ax1.plot(sin_theta, lm_predictions, 'r-', linewidth=2,
             label=f'LM fit (H={lm_height:.3f}m, SSE={lm_sse:.2f})')
    ax1.plot(sin_theta, lsp_predictions, 'b--', linewidth=2,
             label=f'LSP model (H={lsp_height:.3f}m, SSE={lsp_sse:.2f})')

    ax1.set_ylabel('dSNR')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Model Comparison')

    # Bottom subplot: Residuals
    lm_residuals = dsnr_values - lm_predictions
    lsp_residuals = dsnr_values - lsp_predictions

    ax2.plot(sin_theta, lm_residuals, 'r-', alpha=0.7, label='LM residuals')
    ax2.plot(sin_theta, lsp_residuals, 'b--', alpha=0.7, label='LSP residuals')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    ax2.set_xlabel('sin(θ)')
    ax2.set_ylabel('Residuals')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Model Residuals')

    plt.tight_layout()

    if save_path:
        plt.show()
        #plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")


def main():
    """
    Main function to compare LM and LSP methods.
    """
    filepath = "/home/george/Scripts/gnssIR/data/refl_code/2025/arcs/gns1/105/sat001_L2_G_az035.txt"

    try:
        logger.info(f"Processing file: {filepath}")

        # Levenberg-Marquardt  Analysis
        estimator = GNSSMultipathEstimator()
        estimator.read_data(filepath)

        # Determine wavelength from filename
        wavelength = 0.19029 if 'L1' in filepath else 0.24421  # meters
        logger.info(f"Signal type detected: {'L1' if 'L1' in filepath else 'L2'}")
        logger.info(f"Wavelength: {wavelength:.4f}m")


        lm_params, converged = estimator.fit_parameters(height_bounds=(2.0, 4.0))

        if not converged:
            logger.error("LM fit did not converge")
            return

        # Get LM model predictions (includes dampening)
        lm_predictions = estimator.snr_model(
            estimator.elevation_angles,
            *lm_params
        )

        # LSP Analysis - get height and fit amplitude/phase
        lsp_height, lsp_predictions, amplitude, phase = estimate_height_lsp(
            estimator.elevation_angles,
            estimator.dsnr_values,
            wavelength
        )

        # Log results
        logger.info(f"LM height estimate: {lm_params[2]:.3f}m")
        logger.info(f"LSP height estimate: {lsp_height:.3f}m")
        logger.info(f"Height difference: {abs(lsp_height - lm_params[2]):.3f}m")

        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Generate comparison plot
        plot_comparison(
            np.sin(estimator.elevation_angles),
            estimator.dsnr_values,
            lm_predictions,
            lsp_predictions,
            lm_params[2],
            lsp_height,
            save_path=output_dir / "height_comparison.png"
        )

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()