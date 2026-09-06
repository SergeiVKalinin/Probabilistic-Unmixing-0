"""Unified inference interface for probabilistic unmixing."""

from .level0 import fit_level0
from .level1 import fit_level1


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
        raise NotImplementedError(
            "Level 2 inference has not been added yet."
        )

    raise ValueError(
        "level must be 0, 1, or 2."
    )
