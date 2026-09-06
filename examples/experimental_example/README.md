# Experimental Data Example

This folder contains a minimal example showing how experimental data
can be supplied to `probunmix`.

For the current implementation,

**composition = mixture fraction.**

The experimental measurements are converted into the same
`UnmixingData` object used by the synthetic benchmark.

## Minimal experimental inputs

An experiment requires:

- `E`: spectral coordinate, shape `(M,)`
- `Y`: measured spectra, shape `(N, M)`
- `W_mean`: measured concentrations / mixture fractions, shape `(N, K)`
- `W_sigma`: concentration uncertainties, shape `(N, K)`
- `sigma_y`: spectral measurement uncertainty

where:

- `N` = number of measurement locations
- `M` = number of spectral channels
- `K` = number of mixture components

## Example construction

```python
import probunmix as pu

data = pu.from_arrays(
    E=E,
    Y=Y,
    W_mean=W_mean,
    W_sigma=W_sigma,
    sigma_y=sigma_y,
)
