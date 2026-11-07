"""First- and second-order accumulated local effects.

The implementation follows the finite-difference construction of Apley and
Zhu.  Effects are calculated only from observations inside occupied empirical
quantile cells, avoiding the unconditional Cartesian grids used by partial
dependence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Predictor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FirstOrderALE:
    centers: np.ndarray
    effects: np.ndarray
    counts: np.ndarray
    edges: np.ndarray


@dataclass(frozen=True)
class SecondOrderALE:
    centers_x: np.ndarray
    centers_y: np.ndarray
    effects: np.ndarray
    counts: np.ndarray
    edges_x: np.ndarray
    edges_y: np.ndarray
    strength: float
    rank1_fraction: float


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    quantiles = np.linspace(0.0, 1.0, min(n_bins, len(values)) + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        value = float(values[0])
        width = max(abs(value), 1.0) * 1e-9
        edges = np.array([value - width, value + width])
    return edges


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, len(edges) - 2)


def first_order_ale(
    predict: Predictor,
    X: np.ndarray,
    feature: int,
    n_bins: int = 10,
) -> FirstOrderALE:
    """Estimate a centered first-order ALE curve."""

    X = np.asarray(X, dtype=float)
    edges = _quantile_edges(X[:, feature], n_bins)
    bins = _bin_index(X[:, feature], edges)
    n_actual = len(edges) - 1

    lower = X.copy()
    upper = X.copy()
    lower[:, feature] = edges[bins]
    upper[:, feature] = edges[bins + 1]
    differences = np.asarray(predict(upper), dtype=float) - np.asarray(predict(lower), dtype=float)

    counts = np.bincount(bins, minlength=n_actual).astype(float)
    sums = np.bincount(bins, weights=differences, minlength=n_actual)
    local = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    effects = np.cumsum(local) - 0.5 * local
    if counts.sum() > 0:
        effects -= np.average(effects, weights=counts)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return FirstOrderALE(centers, effects, counts, edges)


def _weighted_double_center(surface: np.ndarray, weights: np.ndarray) -> np.ndarray:
    centered = np.asarray(surface, dtype=float).copy()
    weights = np.asarray(weights, dtype=float)
    # Alternating projections provide weighted row/column centering for sparse
    # empirical grids without evaluating unobserved cells.
    for _ in range(12):
        row_weight = weights.sum(axis=1)
        row_mean = np.divide(
            (centered * weights).sum(axis=1),
            row_weight,
            out=np.zeros_like(row_weight),
            where=row_weight > 0,
        )
        centered -= row_mean[:, None]
        col_weight = weights.sum(axis=0)
        col_mean = np.divide(
            (centered * weights).sum(axis=0),
            col_weight,
            out=np.zeros_like(col_weight),
            where=col_weight > 0,
        )
        centered -= col_mean[None, :]
    total = weights.sum()
    if total > 0:
        centered -= float((centered * weights).sum() / total)
    return centered


def second_order_ale(
    predict: Predictor,
    X: np.ndarray,
    feature_x: int,
    feature_y: int,
    n_bins: int = 8,
) -> SecondOrderALE:
    """Estimate a centered two-feature ALE interaction surface.

    ``strength`` is the RMS interaction effect relative to the standard
    deviation of model predictions.  ``rank1_fraction`` measures how closely
    the interaction surface resembles a product of two one-dimensional terms.
    """

    X = np.asarray(X, dtype=float)
    edges_x = _quantile_edges(X[:, feature_x], n_bins)
    edges_y = _quantile_edges(X[:, feature_y], n_bins)
    bx = _bin_index(X[:, feature_x], edges_x)
    by = _bin_index(X[:, feature_y], edges_y)
    nx, ny = len(edges_x) - 1, len(edges_y) - 1

    x00 = X.copy()
    x10 = X.copy()
    x01 = X.copy()
    x11 = X.copy()
    x00[:, feature_x], x00[:, feature_y] = edges_x[bx], edges_y[by]
    x10[:, feature_x], x10[:, feature_y] = edges_x[bx + 1], edges_y[by]
    x01[:, feature_x], x01[:, feature_y] = edges_x[bx], edges_y[by + 1]
    x11[:, feature_x], x11[:, feature_y] = edges_x[bx + 1], edges_y[by + 1]
    delta = (
        np.asarray(predict(x11), dtype=float)
        - np.asarray(predict(x10), dtype=float)
        - np.asarray(predict(x01), dtype=float)
        + np.asarray(predict(x00), dtype=float)
    )

    flat = bx * ny + by
    counts = np.bincount(flat, minlength=nx * ny).reshape(nx, ny).astype(float)
    sums = np.bincount(flat, weights=delta, minlength=nx * ny).reshape(nx, ny)
    local = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    accumulated = np.cumsum(np.cumsum(local, axis=0), axis=1)
    interaction = _weighted_double_center(accumulated, counts)

    observed = interaction[bx, by]
    pred_scale = max(float(np.std(predict(X))), 1e-12)
    strength = float(np.sqrt(np.mean(observed**2)) / pred_scale)

    weighted = interaction * np.sqrt(counts)
    singular = np.linalg.svd(weighted, compute_uv=False)
    energy = float(np.sum(singular**2))
    rank1 = float(singular[0] ** 2 / energy) if energy > 1e-20 else 0.0
    return SecondOrderALE(
        centers_x=(edges_x[:-1] + edges_x[1:]) / 2.0,
        centers_y=(edges_y[:-1] + edges_y[1:]) / 2.0,
        effects=interaction,
        counts=counts,
        edges_x=edges_x,
        edges_y=edges_y,
        strength=strength,
        rank1_fraction=rank1,
    )
