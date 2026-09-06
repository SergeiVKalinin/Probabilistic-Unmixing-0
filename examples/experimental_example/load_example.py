"""Load the experimental-style example into probunmix.

This script intentionally does NOT load synthetic ground truth.
Once the files have been written to disk, they are treated exactly
like experimental measurements.
"""

from pathlib import Path

import numpy as np

import probunmix as pu


# ============================================================
# File location
# ============================================================

HERE = Path(__file__).resolve().parent


# ============================================================
# Load experimental measurements
# ============================================================

E = np.loadtxt(
    HERE / "energy.csv",
    delimiter=",",
    skiprows=1,
)

Y = np.loadtxt(
    HERE / "spectra.csv",
    delimiter=",",
)

sigma_y = np.loadtxt(
    HERE / "spectral_noise.csv",
    delimiter=",",
)


# ============================================================
# Load composition / mixture-fraction information
# ============================================================

composition_table = np.loadtxt(
    HERE / "compositions.csv",
    delimiter=",",
    skiprows=1,
)

# This example has four components.
K = composition_table.shape[1] // 2

W_mean = composition_table[:, :K]
W_sigma = composition_table[:, K:]


# ============================================================
# Construct the common experimental data object
# ============================================================

data = pu.from_arrays(
    E=E,
    Y=Y,
    W_mean=W_mean,
    W_sigma=W_sigma,
    sigma_y=sigma_y,
)


# ============================================================
# Attach Level 0 spectral knowledge
# ============================================================

S_fixed = np.loadtxt(
    HERE / "level0_endmembers.csv",
    delimiter=",",
)

data.set_level0(
    S_fixed
)


# ============================================================
# Attach Level 1 spectral knowledge
# ============================================================

theta_mean = np.loadtxt(
    HERE / "level1_theta_mean.csv",
    delimiter=",",
)

theta_sigma = np.loadtxt(
    HERE / "level1_theta_sigma.csv",
    delimiter=",",
)

data.set_level1(
    model="two_gaussian",
    theta_mean=theta_mean,
    theta_sigma=theta_sigma,
)


# ============================================================
# Attach Level 2 spectral knowledge
# ============================================================

S_library = np.load(
    HERE / "level2_spectral_library.npy",
)

data.set_level2(
    S_library
)


# ============================================================
# Validate and inspect
# ============================================================

pu.validate_data(data)
pu.print_data_summary(data)


if __name__ == "__main__":
    print()
    print("Experimental-style dataset loaded successfully.")
