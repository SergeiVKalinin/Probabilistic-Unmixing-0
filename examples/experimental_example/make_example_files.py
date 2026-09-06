"""Create a small experimental-style dataset for probunmix examples.

The files produced here contain measured quantities and spectral
knowledge in ordinary CSV/NumPy formats.

The purpose is to mimic how a real experimental dataset would enter
the package. The downstream analysis should not depend on the fact
that these files were generated synthetically.
"""

from pathlib import Path

import numpy as np

import probunmix as pu


# ============================================================
# Output directory
# ============================================================

HERE = Path(__file__).resolve().parent


# ============================================================
# Generate a small benchmark
# ============================================================

data = pu.generate_example(
    n_locations=30,
    n_channels=120,
    n_library=50,
    seed=2026,
)


# ============================================================
# Core experimental measurements
# ============================================================

np.savetxt(
    HERE / "energy.csv",
    data.E,
    delimiter=",",
    header="spectral_coordinate",
    comments="",
)

np.savetxt(
    HERE / "spectra.csv",
    data.Y,
    delimiter=",",
)

np.savetxt(
    HERE / "spectral_noise.csv",
    data.sigma_y,
    delimiter=",",
)


# ============================================================
# Composition / mixture-fraction information
# ============================================================

K = data.n_components

composition_table = np.column_stack(
    [
        data.W_mean,
        data.W_sigma,
    ]
)

header = ",".join(
    [f"component_{k + 1}" for k in range(K)]
    +
    [f"component_{k + 1}_sd" for k in range(K)]
)

np.savetxt(
    HERE / "compositions.csv",
    composition_table,
    delimiter=",",
    header=header,
    comments="",
)


# ============================================================
# Level 0 spectral knowledge
# ============================================================

np.savetxt(
    HERE / "level0_endmembers.csv",
    data.S_fixed,
    delimiter=",",
)


# ============================================================
# Level 1 spectral knowledge
# ============================================================

np.savetxt(
    HERE / "level1_theta_mean.csv",
    data.theta_mean,
    delimiter=",",
)

np.savetxt(
    HERE / "level1_theta_sigma.csv",
    data.theta_sigma,
    delimiter=",",
)


# ============================================================
# Level 2 spectral knowledge
# ============================================================

np.save(
    HERE / "level2_spectral_library.npy",
    data.S_library,
)


print("Experimental-style example files created in:")
print(HERE)
