# Probabilistic Unmixing 0

A compact Python package for probabilistic spectral unmixing with uncertain concentrations and progressively weaker assumptions about the component spectra.

For the current implementation:

**composition = mixture fraction**

The same data object and inference interface are used for synthetic benchmark data and experimental data.

---

## Core idea

We observe spectra

$$
Y_i(E) = \sum_k W_{ik} S_k(E) + \epsilon_i(E)
$$

where:

- `Y` contains the measured mixture spectra
- `W` contains concentrations / mixture fractions
- `S` contains component endmember spectra
- `sigma_y` describes spectral measurement uncertainty

The package considers three levels of prior spectral knowledge.

### Level 0 — Known endmembers

The component spectra are assumed known exactly.

```python
result = pu.fit(data, level=0)
```

### Level 1 — Parametric spectral uncertainty

Each component spectrum has a known parametric form, but its parameters are uncertain.

The current implementation uses a two-Gaussian model.

```python
result = pu.fit(data, level=1)
```

### Level 2 — Functional spectral uncertainty

Prior spectral knowledge is supplied as an ensemble of plausible complete component spectra.

The library is represented in a reduced log-spectrum PCA basis, and the functional coefficients are inferred jointly with the concentrations.

```python
result = pu.fit(data, level=2)
```

---

## Installation

Install directly from GitHub:

```python
!pip install -q --upgrade --no-deps \
git+https://github.com/YOUR_GITHUB_USERNAME/Probabilistic-Unmixing-0.git
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

Then:

```python
import probunmix as pu
```

---

# Synthetic benchmark

Generate one dataset containing everything needed for Level 0, Level 1, and Level 2:

```python
import probunmix as pu

data = pu.generate_example()

pu.validate_data(data)
pu.print_data_summary(data)

pu.plot_data(data);
```

The generated object contains:

- measured spectra
- spectral measurement uncertainty
- measured concentrations / mixture fractions
- concentration uncertainty
- exact endmember spectra for Level 0
- parametric spectral priors for Level 1
- functional spectral libraries for Level 2
- synthetic ground truth for evaluation only

Run the same dataset through all three analyses:

```python
r0 = pu.fit(data, level=0)

r1 = pu.fit(
    data,
    level=1,
)

r2 = pu.fit(
    data,
    level=2,
)
```

Compare the results:

```python
pu.plot_level_comparison(
    data,
    r0,
    r1,
    r2,
);
```

The same measured spectra and concentration information are used for all three levels. Only the assumed spectral knowledge changes.

---

# Experimental data

Experimental measurements enter through the same `UnmixingData` object.

The minimal experimental inputs are:

```text
E         spectral coordinate                  (M,)
Y         measured spectra                     (N, M)
W_mean    measured concentrations/fractions    (N, K)
W_sigma   concentration uncertainty             (N, K)
sigma_y   spectral measurement uncertainty      scalar or (N, M)
```

where:

- `N` = number of measurement locations
- `M` = number of spectral channels
- `K` = number of components

Create the data object:

```python
data = pu.from_arrays(
    E=E,
    Y=Y,
    W_mean=W_mean,
    W_sigma=W_sigma,
    sigma_y=sigma_y,
)
```

Validate and inspect it:

```python
pu.validate_data(data)
pu.print_data_summary(data)
pu.plot_data(data);
```

Then attach the spectral knowledge appropriate for the desired analysis level.

---

## Experimental Level 0

Supply fixed reference endmember spectra:

```python
data.set_level0(
    S_reference
)
```

Run the analysis:

```python
result = pu.fit(
    data,
    level=0,
)
```

---

## Experimental Level 1

Supply a parametric spectral model and uncertain parameters:

```python
data.set_level1(
    model="two_gaussian",
    theta_mean=theta_mean,
    theta_sigma=theta_sigma,
)
```

Run the analysis:

```python
result = pu.fit(
    data,
    level=1,
)
```

---

## Experimental Level 2

Supply an ensemble of plausible complete spectra:

```python
data.set_level2(
    S_library
)
```

where the library has shape

```text
(K, number_of_library_spectra, M)
```

Run the analysis:

```python
result = pu.fit(
    data,
    level=2,
)
```

The downstream analysis interface is therefore identical for synthetic and experimental data.

---

# Common result object

All three analysis levels return the same `UnmixingResult` structure.

Typical quantities include:

```python
result.W_mean
result.W_sd

result.S_mean
result.S_sd

result.Y_pred
result.residuals

result.metrics
result.metadata
```

For Level 0, the endmember spectra are assumed exact, so:

```python
result.S_sd
```

is identically zero.

For Levels 1 and 2, the recovered endmembers have posterior uncertainty.

---

# Visualization

Plot the input dataset:

```python
pu.plot_data(data);
```

Plot an individual analysis result:

```python
pu.plot_result(
    data,
    result,
);
```

Compare several spectral-knowledge levels:

```python
pu.plot_level_comparison(
    data,
    r0,
    r1,
    r2,
);
```

For synthetic data, ground truth is shown when available.

For experimental data, the same plots work without requiring ground truth.

---

# Experimental-style example

The folder

```text
examples/experimental_example/
```

demonstrates the complete experimental-data pathway.

It contains scripts to:

1. generate experimental-style files
2. reload those files without synthetic ground truth
3. construct an `UnmixingData` object
4. attach Level 0, Level 1, and Level 2 spectral knowledge
5. run all three analyses

The important point is that once the data are loaded from disk, the downstream package does not distinguish between synthetic and experimental origins.

---

# Repository structure

```text
Probabilistic-Unmixing-0/
│
├── src/
│   └── probunmix/
│       ├── __init__.py
│       ├── data.py
│       ├── diagnostics.py
│       ├── fit.py
│       ├── io.py
│       ├── level0.py
│       ├── level1.py
│       ├── level2.py
│       ├── plotting.py
│       ├── results.py
│       └── synthetic.py
│
├── examples/
│   └── experimental_example/
│       ├── README.md
│       ├── make_example_files.py
│       ├── load_example.py
│       └── run_all_levels.py
│
├── tests/
│   ├── test_data.py
│   ├── test_level0.py
│   ├── test_level1.py
│   ├── test_level2.py
│   └── test_integration.py
│
├── pyproject.toml
├── LICENSE
└── README.md
```

The notebooks are intentionally kept outside the package implementation and can be developed independently in Google Colab.

---

# Package workflow

The package is organized around one common workflow.

## Synthetic data

```python
data = pu.generate_example()
```

## Experimental data

```python
data = pu.from_arrays(
    E=E,
    Y=Y,
    W_mean=W_mean,
    W_sigma=W_sigma,
    sigma_y=sigma_y,
)
```

Both routes produce the same `UnmixingData` object.

Then:

```python
pu.validate_data(data)
pu.plot_data(data);

result = pu.fit(
    data,
    level=0,   # or 1 or 2
)

pu.plot_result(
    data,
    result,
);
```

Conceptually:

```text
generate_example() ───────┐
                          │
                          ▼
                    UnmixingData
                          ▲
                          │
from_arrays() ────────────┘
                          │
                          ▼
                 fit(data, level)
                          │
                          ▼
                   UnmixingResult
                          │
                          ▼
              diagnostics + visualization
```

---

# Current scope

The present implementation assumes:

- linear spectral mixing
- composition = mixture fraction
- nonnegative fractions that sum to one
- globally shared endmember spectra
- known spectral measurement uncertainty
- measured concentrations with uncertainty
- no spatial coupling between measurement locations

The three levels differ only in the representation of spectral knowledge:

| Level | Spectral knowledge |
|---|---|
| 0 | Fixed exact endmember spectra |
| 1 | Parametric spectral model with uncertain parameters |
| 2 | Ensemble of plausible complete spectra |

---

# Status

Current version: `0.1.0`

Implemented:

- common synthetic / experimental data object
- synthetic benchmark generator
- experimental array loader
- Level 0 inference
- Level 1 parametric inference
- Level 2 functional inference
- common result object
- dataset validation and diagnostics
- input and result visualization
- cross-level comparison
- experimental-style example
- automated tests
````
