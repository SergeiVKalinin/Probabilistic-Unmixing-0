"""Unified inference interface for probabilistic unmixing."""

from .level0 import fit_level0
from .level1 import fit_level1
from .level2 import fit_level2


def fit(data, level, **kwargs):
    """
    Run probabilistic unmixing at the requested knowledge level.

    Parameters
    ----------
    data : UnmixingData
        Dataset containing measurements, concentration information,
        and the spectral knowledge required by the selected level.

    level : int
        Spectral-knowledge level.

        0 : fixed, exactly known endmember spectra
        1 : parametric spectral uncertainty
        2 : functional spectral uncertainty

    **kwargs
        Additional arguments passed to the corresponding inference
        function.

    Returns
    -------
    UnmixingResult
        Common result object returned by all inference levels.
    """

    if level == 0:
        return fit_level0(
            data,
            **kwargs,
        )

    if level == 1:
        return fit_level1(
            data,
            **kwargs,
        )

    if level == 2:
        return fit_level2(
            data,
            **kwargs,
        )

    raise ValueError(
        "level must be 0, 1, or 2."
    )
