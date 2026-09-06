# Probabilistic Unmixing 0

A compact Python package for probabilistic spectral unmixing with
uncertain concentrations and progressively weaker assumptions about
the component spectra.

For the current implementation:

**composition = mixture fraction**

The same data object and inference interface are used for synthetic
benchmark data and experimental data.

---

## Core idea

We observe spectra

\[
Y_i(E) = \sum_k W_{ik} S_k(E) + \epsilon_i(E)
\]

where:

- `Y` contains the measured mixture spectra,
- `W` contains concentrations / mixture fractions,
- `S` contains component endmember spectra,
- `sigma_y` describes spectral measurement uncertainty.

The package studies three levels of prior spectral knowledge.

### Level 0 — Known endmembers

The component spectra are assumed known exactly.

```python
result = pu.fit(data, level=0)
