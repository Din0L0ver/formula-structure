"""Fit simple one-dimensional transform families to ALE curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class ShapeFit:
    family: str
    score: float
    parameter: float | None
    monotonicity: float
    power_alpha: float | None
    power_r2: float | None
    power_alpha_low: float | None
    power_alpha_high: float | None


def _weighted_r2(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    mean = float(np.average(y, weights=weights))
    total = float(np.sum(weights * (y - mean) ** 2))
    residual = float(np.sum(weights * (y - prediction) ** 2))
    return 1.0 - residual / total if total > 1e-20 else 0.0


def _linear_fit(phi: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    design = np.column_stack([np.ones(len(phi)), phi])
    root_w = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)
    prediction = design @ beta
    return _weighted_r2(y, prediction, weights), prediction


def fit_shape(x: np.ndarray, effect: np.ndarray, counts: np.ndarray) -> ShapeFit:
    """Return the best low-complexity transform for a first-order ALE curve."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(effect, dtype=float)
    weights = np.maximum(np.asarray(counts, dtype=float), 1.0)
    order = np.argsort(x)
    monotonicity = float(np.corrcoef(np.arange(len(x)), y[order])[0, 1]) if len(x) > 2 else 0.0
    candidates: list[tuple[str, float, float | None]] = []

    fixed: dict[str, np.ndarray] = {"linear": x, "quadratic": x**2}
    power_alpha: float | None = None
    power_r2: float | None = None
    power_alpha_low: float | None = None
    power_alpha_high: float | None = None
    if np.all(x > 0):
        fixed.update({"sqrt": np.sqrt(x), "log": np.log(x), "inverse": 1.0 / x})
    for name, phi in fixed.items():
        score, _ = _linear_fit(phi, y, weights)
        candidates.append((name, score, None))

    if np.all(x > 0):
        scale = float(np.median(x))
        normalized = x / max(scale, 1e-12)

        def objective(alpha: float) -> float:
            phi = np.log(normalized) if abs(alpha) < 1e-6 else normalized**alpha
            score, _ = _linear_fit(phi, y, weights)
            return -score

        optimized = minimize_scalar(objective, bounds=(-2.0, 3.0), method="bounded")
        alpha = float(optimized.x)
        power_score = float(-optimized.fun)
        power_alpha = alpha
        power_r2 = power_score
        alpha_grid = np.linspace(-2.0, 3.0, 501)
        grid_scores = np.array([-objective(value) for value in alpha_grid])
        practically_equivalent = alpha_grid[grid_scores >= power_score - 0.002]
        if len(practically_equivalent):
            power_alpha_low = float(practically_equivalent.min())
            power_alpha_high = float(practically_equivalent.max())
        # Canonicalize exponents that are numerically indistinguishable from
        # common simpler forms.  Otherwise retain a free exponent only when it
        # explains materially more ALE variation than a named transform.
        if abs(alpha - 1.0) <= 0.08:
            candidates.append(("linear", power_score, None))
        elif abs(alpha - 0.5) <= 0.06:
            candidates.append(("sqrt", power_score, None))
        else:
            candidates.append(("power", power_score, alpha))

    # A small preference for named, lower-complexity shapes prevents tiny ALE
    # approximation gains from turning every recommendation into x**alpha.
    penalties = {
        "linear": 0.0,
        "sqrt": 0.001,
        "log": 0.001,
        "quadratic": 0.002,
        "inverse": 0.002,
        "power": 0.0015,
    }
    family, score, parameter = max(candidates, key=lambda item: item[1] - penalties[item[0]])
    return ShapeFit(
        family,
        float(score),
        parameter,
        monotonicity,
        power_alpha,
        power_r2,
        power_alpha_low,
        power_alpha_high,
    )


def format_transform(feature: str, family: str, parameter: float | None) -> str:
    if family == "linear":
        return feature
    if family == "quadratic":
        return f"{feature}^2"
    if family == "sqrt":
        return f"sqrt({feature})"
    if family == "log":
        return f"log({feature})"
    if family == "inverse":
        return f"1/{feature}"
    if family == "power" and parameter is not None:
        return f"{feature}^{parameter:.3g}"
    return f"g({feature})"
