"""Run Level 0, Level 1, and Level 2 on experimental-style data."""

from load_example import data

import probunmix as pu


# ============================================================
# Level 0
# ============================================================

result0 = pu.fit(
    data,
    level=0,
    n_draws=500,
    seed=101,
)

print()
print("LEVEL 0")
print(result0.metrics)


# ============================================================
# Level 1
# ============================================================

result1 = pu.fit(
    data,
    level=1,
    n_draws=500,
    seed=102,
    max_nfev=200,
)

print()
print("LEVEL 1")
print(result1.metrics)


# ============================================================
# Level 2
# ============================================================

result2 = pu.fit(
    data,
    level=2,
    n_draws=500,
    seed=103,
    max_nfev=200,
)

print()
print("LEVEL 2")
print(result2.metrics)


# ============================================================
# Compare all three
# ============================================================

pu.plot_level_comparison(
    data,
    result0,
    result1,
    result2,
)

print()
print("All three levels completed successfully.")
