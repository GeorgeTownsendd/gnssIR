import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.signal import lombscargle
from typing import Tuple, Optional
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNSSMultipathEstimator:
    """
    Estimator for GNSS-IR (Interferometric Reflectometry) signal multipath patterns.
    Implements a cosine model with damping to fit GNSS reflected signals.
    """

    # GNSS wavelength constants in meters
    GPS_L1_WAVELENGTH = 0.1903
    GPS_L2_WAVELENGTH = 0.2442

    def __init__(self):
        self.elevation_angles = None
        self.dsnr_values = None
        self.times = None
        self.params = None
        self.wavelength = None

    def detect_signal_type(self, filename: str) -> float:
        """
        Detect GNSS signal type from filename and return corresponding wavelength.

        Args:
            filename: Input filename containing signal type indicator (L1 or L2)

        Returns:
            float: Signal wavelength in meters
        """
        if '_L1_' in filename:
            logger.info("Detected GPS L1 signal")
            return self.GPS_L1_WAVELENGTH
        elif '_L2_' in filename:
            logger.info("Detected GPS L2 signal")
            return self.GPS_L2_WAVELENGTH
        else:
            raise ValueError("Unable to detect signal type (L1 or L2) from filename")

    def read_data(self, filename: str) -> None:
        """
        Read and preprocess data from input file, detecting signal type.

        Args:
            filename: Path to input file containing elevation angles, dSNR, and times
        """
        try:
            # Detect signal type and set wavelength
            self.wavelength = self.detect_signal_type(filename)

            # Skip header row starting with %
            data = np.genfromtxt(filename, skip_header=1)

            # Extract columns
            self.elevation_angles = np.radians(data[:, 0])  # Convert to radians
            self.dsnr_values = data[:, 1]
            self.times = data[:, 2]

            # Filter invalid entries
            valid_mask = (self.elevation_angles >= 0) & ~np.isnan(self.elevation_angles) & \
                         ~np.isnan(self.dsnr_values)

            self.elevation_angles = self.elevation_angles[valid_mask]
            self.dsnr_values = self.dsnr_values[valid_mask]
            self.times = self.times[valid_mask]

            logger.info(f"Loaded {len(self.elevation_angles)} valid data points")

        except Exception as e:
            logger.error(f"Error reading data file: {str(e)}")
            raise

    def snr_model(self, theta: np.ndarray, A: float, eta: float, H: float, phi: float) -> np.ndarray:
        """
        Implement the SNR multipath model with direct reflector height parameterization.

        Args:
            theta: Elevation angles in radians
            A: Amplitude
            eta: Attenuation factor
            H: Reflector height in meters
            phi: Phase offset

        Returns:
            Model predictions for given parameters
        """
        return A * np.exp(-eta * np.sin(theta)) * np.cos((4 * np.pi * H / self.wavelength) * np.sin(theta) + phi)

    def initial_parameter_estimate(self) -> Tuple[float, float, float, float]:
        """
        Estimate initial parameters using Lomb-Scargle periodogram.

        Returns:
            Tuple of (A, eta, H, phi) initial guesses
        """
        # Amplitude and attenuation initial guesses
        A_init = np.max(np.abs(self.dsnr_values))
        eta_init = 1.0

        # Prepare data for Lomb-Scargle
        sin_theta = np.sin(self.elevation_angles)

        # Create array of candidate reflector heights
        heights = np.linspace(0.5, 5.0, 1000)

        # Convert heights to frequencies using f = 4πH/λ
        frequencies = 4 * np.pi * heights / self.wavelength

        # Compute Lomb-Scargle periodogram
        pgram = lombscargle(sin_theta, self.dsnr_values, frequencies)

        # Find height corresponding to maximum power
        H_init = heights[np.argmax(pgram)]

        # If initial height estimate is unrealistic, use a reasonable default
        if H_init < 1.0 or H_init > 3.0:
            H_init = 2.0  # Use typical reflector height as default

        # Rough phase estimate
        phi_init = 0.0

        logger.info(f"Initial parameter estimates: A={A_init:.2f}, eta={eta_init:.2f}, "
                    f"H={H_init:.2f}m, phi={phi_init:.2f}")

        return A_init, eta_init, H_init, phi_init

    def fit_parameters(self, height_bounds: Tuple[float, float] = (1.0, 3.0)) -> Tuple[np.ndarray, bool]:
        """
        Perform nonlinear least squares fitting of the SNR model with bounds on H.

        Args:
            height_bounds: Min and max bounds for reflector height parameter (meters)

        Returns:
            Tuple of (optimal parameters, convergence flag)
        """
        if self.elevation_angles is None or self.dsnr_values is None:
            raise ValueError("Data must be loaded before fitting")

        # Get initial guesses
        initial_guess = self.initial_parameter_estimate()

        # Define residual function for least squares
        def residual(params):
            return self.dsnr_values - self.snr_model(self.elevation_angles, *params)

        # Set bounds to ensure H stays in physical range
        # Format: (lower_bounds, upper_bounds) for (A, eta, H, phi)
        bounds = ([0, 0, height_bounds[0], -np.inf],  # lower bounds
                  [np.inf, np.inf, height_bounds[1], np.inf])  # upper bounds

        # Perform optimization
        result = optimize.least_squares(
            residual,
            initial_guess,
            method='trf',  # Trust Region Reflective algorithm (supports bounds)
            bounds=bounds,
            ftol=1e-8,
            xtol=1e-8
        )

        self.params = result.x

        # Log the results
        A, eta, H, phi = result.x
        logger.info(f"Fitted parameters: A={A:.4f}, eta={eta:.4f}, H={H:.4f}m, phi={phi:.4f}")
        logger.info(f"Optimization success: {result.success}, status: {result.status}")

        return result.x, result.success