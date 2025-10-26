"""Parameter uncertainty sampling utilities for the OpenSeesPy frame model."""
from __future__ import annotations

import logging
from typing import Dict, Iterable

import numpy as np

PARAMETER_ORDER = [
    "fys",
    "Es",
    "bs",
    "fc_cover",
    "rho_con",
    "eps_c_cover",
    "eps_cu_cover",
    "fc_core",
    "fcu_core",
    "eps_c_core",
    "eps_cu_core",
    "rho_s",
    "Hc",
    "Ds",
    "L",
    "zeta",
]

_MEAN_VALUES = np.array(
    [
        420.0,      # fys (MPa)
        200000.0,   # Es (MPa)
        0.01,       # bs (-)
        30.0,       # fc_cover (MPa)
        2500.0,     # rho_con (kg/m^3)
        0.0020,     # eps_c_cover
        0.0035,     # eps_cu_cover
        35.0,       # fc_core (MPa)
        20.0,       # fcu_core (MPa)
        0.0022,     # eps_c_core
        0.0040,     # eps_cu_core
        0.02,       # rho_s (-)
        3.0,        # Hc (m)
        0.5,        # Ds (m)
        6.0,        # L (m)
        0.05,       # zeta (-)
    ]
)

_STD_VALUES = np.array(
    [
        42.0,
        10000.0,
        0.002,
        3.0,
        50.0,
        0.0002,
        0.0003,
        3.5,
        2.0,
        0.0002,
        0.0004,
        0.002,
        0.10,
        0.05,
        0.20,
        0.005,
    ]
)


def _build_correlation_matrix() -> np.ndarray:
    """Construct the correlation matrix following the specification."""
    size = len(PARAMETER_ORDER)
    corr = np.eye(size)

    def _set_pairs(pairs: Iterable[tuple[int, int]], value: float) -> None:
        for i, j in pairs:
            corr[i, j] = value
            corr[j, i] = value

    index = {name: idx for idx, name in enumerate(PARAMETER_ORDER)}

    # Steel parameters (strong correlation 0.8)
    steel_pairs = [
        (index["fys"], index["Es"]),
        (index["fys"], index["bs"]),
        (index["Es"], index["bs"]),
    ]
    _set_pairs(steel_pairs, 0.8)

    # Concrete strong correlations
    strong_pairs = [
        (index["fc_core"], index["fcu_core"]),
        (index["eps_c_core"], index["eps_cu_core"]),
        (index["fc_core"], index["fc_cover"]),
        (index["eps_c_cover"], index["eps_cu_cover"]),
        (index["eps_c_core"], index["eps_c_cover"]),
        (index["eps_cu_core"], index["eps_cu_cover"]),
    ]
    _set_pairs(strong_pairs, 0.8)

    # Concrete moderate correlations (0.64)
    moderate_pairs = [
        (index["eps_c_core"], index["eps_cu_cover"]),
        (index["eps_c_cover"], index["eps_cu_core"]),
    ]
    _set_pairs(moderate_pairs, 0.64)

    return corr


_CORRELATION_MATRIX = _build_correlation_matrix()


def _covariance_matrix() -> np.ndarray:
    """Return a symmetric positive definite covariance matrix."""
    base_cov = np.outer(_STD_VALUES, _STD_VALUES) * _CORRELATION_MATRIX
    try:
        np.linalg.cholesky(base_cov)
        return base_cov
    except np.linalg.LinAlgError:
        logging.warning("Covariance matrix not SPD; applying eigenvalue correction.")
        eigvals, eigvecs = np.linalg.eigh(base_cov)
        eigvals[eigvals < 1e-8] = 1e-8
        return eigvecs @ np.diag(eigvals) @ eigvecs.T


_COVARIANCE_MATRIX = _covariance_matrix()


def sample_parameters(seed: int | None = None) -> Dict[str, float]:
    """Sample a parameter set respecting the prescribed correlations.

    Parameters
    ----------
    seed:
        Optional seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing sampled parameter values keyed by parameter name.
    """
    rng = np.random.default_rng(seed)
    try:
        lower = np.linalg.cholesky(_COVARIANCE_MATRIX)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - safeguard
        logging.error("Cholesky factorization failed: %s", exc)
        raise

    standard_normal = rng.standard_normal(len(PARAMETER_ORDER))
    sampled = _MEAN_VALUES + lower @ standard_normal

    result: Dict[str, float] = {}
    for name, value in zip(PARAMETER_ORDER, sampled):
        if name in {"fys", "Es", "fc_cover", "fc_core", "fcu_core", "rho_con"}:
            value = max(value, 1e-6)
        if name in {"bs"}:
            value = float(np.clip(value, 0.002, 0.05))
        elif name == "rho_s":
            value = float(np.clip(value, 0.005, 0.05))
        elif name == "zeta":
            value = float(np.clip(value, 0.01, 0.10))
        elif name in {"Hc", "Ds", "L"}:
            value = max(value, 0.2)
        elif name.startswith("eps"):
            value = max(value, 1e-4)
        result[name] = float(value)

    logging.info("Sampled parameter set: %s", {k: round(v, 6) for k, v in result.items()})
    return result
