"""Load the experimental-style example into probunmix.

This script intentionally does NOT load synthetic ground truth.

If the example data files do not yet exist, they are generated
automatically. Once written to disk, the files are reloaded and
treated exactly like experimental measurements.

The resulting UnmixingData object therefore contains measured data
and supplied spectral knowledge, but no S_true, W_true, or Y_clean.
"""

from pathlib import Path

import numpy as np

import probunmix as pu


# ============================================================
# File location
# ============================================================

HERE = Path(__file__).resolve().parent


# ============================================================
# Create example files if they are not present
# ============================================================

required_files = [
    "energy.csv",
    "spectra.csv",
    "spectral_noise.csv",
    "compositions.csv",
    "level0_endmembers.csv",
    "level1_theta_mean.csv",
    "level1_theta_sigma.csv",
    "level2_spectral_library.npy",
]

missing_files = [
    name
    for name in required_files
    if not (HERE / name).exists()
]

if missing_files:

    print(
        "Example data files are missing. "
        "Generating them now..."
    )

    # Importing this module executes the example-data generator
    # and writes the required files into this directory.
    import make_example_files  # noqa: F401


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

# The table contains:
#
# first K columns  -> W_mean
# next K columns   -> W_sigma

if composition_table.ndim != 2:
    raise ValueError(
        "compositions.csv must contain a two-dimensional table."
    )

if composition_table.shape[1] % 2 != 0:
    raise ValueError(
        "compositions.csv must contain an equal number of "
        "composition and uncertainty columns."
    )

K = (
    composition_table.shape[1]
    // 2
)

W_mean = (
    composition_table[:, :K]
)

W_sigma = (
    composition_table[:, K:]
)


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
# Validate the reconstructed experimental dataset
# ============================================================

pu.validate_data(
    data
)


# ============================================================
# Display summary when this file is executed directly
# ============================================================

if __name__ == "__main__":

    pu.print_data_summary(
        data
    )

    print()
    print(
        "Experimental-style dataset loaded successfully."
    )

    print()
    print("Synthetic truth attached:")
    print(
        "  W_true:",
        data.W_true is not None,
    )
    print(
        "  S_true:",
        data.S_true is not None,
    )
    print(
        "  Y_clean:",
        data.Y_clean is not None,
    )
