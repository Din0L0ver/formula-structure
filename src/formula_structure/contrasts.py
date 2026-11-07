"""Frozen-coordinate contrasts for testing additive and multiplicative modules.

The key observation is that, for an additive component ``f(x) + r(z)``, two
rows with the same ``z`` satisfy

``y(x_2, z) - y(x_1, z) = f(x_2) - f(x_1)``.

The implementation below collects those contrasts across repeated frozen
contexts.  It reports how reproducible the same transition is and reconstructs
an anchored one-dimensional effect from all contrast equations at once.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FrozenContrastResult:
    feature: str
    frozen_features: tuple[str, ...]
    scale_by: str | None
    n_contexts: int
    n_contrasts: int
    n_repeated_contrasts: int
    repeated_fraction: float
    separability_score: float
    median_relative_dispersion: float
    contrasts: pd.DataFrame
    effect_curve: pd.DataFrame

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.contrasts.to_csv(output / "contrasts.csv", index=False)
        self.effect_curve.to_csv(output / "effect_curve.csv", index=False)


def _context_keys(
    frame: pd.DataFrame,
    columns: Iterable[str],
    decimals: int | None,
) -> pd.DataFrame:
    keys = frame[list(columns)].copy()
    if decimals is not None:
        for column in keys:
            if pd.api.types.is_numeric_dtype(keys[column]):
                keys[column] = keys[column].round(decimals)
    return keys


class FrozenContrastAnalyzer:
    """Estimate effects from rows where all other coordinates are frozen.

    Parameters
    ----------
    context_decimals:
        Optional rounding used only to form frozen-context keys.  ``None``
        requires exact matches and is the safest choice for engineered grids.
    adjacent_only:
        Compare adjacent feature levels inside every context.  This avoids a
        quadratic number of redundant contrast equations.
    """

    def __init__(self, *, context_decimals: int | None = None, adjacent_only: bool = True) -> None:
        self.context_decimals = context_decimals
        self.adjacent_only = bool(adjacent_only)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        feature: str,
        *,
        frozen_features: Iterable[str] | None = None,
        scale_by: str | None = None,
    ) -> FrozenContrastResult:
        if feature not in X:
            raise KeyError(f"Unknown feature: {feature}")
        frozen = tuple(
            column
            for column in (X.columns if frozen_features is None else frozen_features)
            if column != feature
        )
        missing = [column for column in frozen if column not in X]
        if missing:
            raise KeyError(f"Unknown frozen features: {missing}")
        if scale_by is not None and scale_by not in frozen:
            raise ValueError("scale_by must be one of the frozen features")

        work = X[[feature, *frozen]].copy()
        work["__target__"] = np.asarray(y, dtype=float)
        work = work.replace([np.inf, -np.inf], np.nan).dropna()
        keys = _context_keys(work, frozen, self.context_decimals)
        if frozen:
            work["__context__"] = pd.MultiIndex.from_frame(keys).map(repr)
            grouped = work.groupby("__context__", sort=False)
        else:
            work["__context__"] = "all"
            grouped = work.groupby("__context__", sort=False)

        rows: list[dict[str, float | str]] = []
        used_contexts = 0
        for context, part in grouped:
            collapsed = (
                part.groupby(feature, as_index=False)
                .agg(target=("__target__", "mean"), count=("__target__", "size"))
                .sort_values(feature)
            )
            if len(collapsed) < 2:
                continue
            used_contexts += 1
            pairs: list[tuple[int, int]]
            if self.adjacent_only:
                pairs = [(index, index + 1) for index in range(len(collapsed) - 1)]
            else:
                pairs = [
                    (left, right)
                    for left in range(len(collapsed) - 1)
                    for right in range(left + 1, len(collapsed))
                ]
            scale = 1.0
            if scale_by is not None:
                scale = float(part[scale_by].mean())
                if not np.isfinite(scale) or abs(scale) <= 1e-12:
                    continue
            for left, right in pairs:
                x_left = float(collapsed.iloc[left][feature])
                x_right = float(collapsed.iloc[right][feature])
                delta_y = float(collapsed.iloc[right]["target"] - collapsed.iloc[left]["target"])
                normalized = delta_y / scale
                rows.append(
                    {
                        "context": str(context),
                        "x_left": x_left,
                        "x_right": x_right,
                        "delta_x": x_right - x_left,
                        "delta_y": delta_y,
                        "scale": scale,
                        "normalized_delta": normalized,
                        "weight": float(
                            min(collapsed.iloc[left]["count"], collapsed.iloc[right]["count"])
                        ),
                    }
                )

        contrasts = pd.DataFrame(rows)
        empty_curve = pd.DataFrame(columns=[feature, "effect", "support"])
        if contrasts.empty:
            return FrozenContrastResult(
                feature,
                frozen,
                scale_by,
                used_contexts,
                0,
                0,
                0.0,
                float("nan"),
                float("nan"),
                contrasts,
                empty_curve,
            )

        transition_columns = ["x_left", "x_right"]
        transition_stats = contrasts.groupby(transition_columns)["normalized_delta"].agg(
            transition_mean="mean", transition_std="std", transition_count="size"
        )
        contrasts = contrasts.join(transition_stats, on=transition_columns)
        repeated = contrasts["transition_count"] >= 2
        repeated_rows = contrasts.loc[repeated]
        if len(repeated_rows) >= 2 and repeated_rows["normalized_delta"].var() > 1e-20:
            residual = (
                repeated_rows["normalized_delta"] - repeated_rows["transition_mean"]
            ).to_numpy(dtype=float)
            centered = (
                repeated_rows["normalized_delta"] - repeated_rows["normalized_delta"].mean()
            ).to_numpy(dtype=float)
            score = 1.0 - float(np.sum(residual**2) / np.sum(centered**2))
        elif len(repeated_rows) >= 2:
            score = 1.0
        else:
            score = float("nan")
        relative_dispersion = (
            transition_stats["transition_std"] / (transition_stats["transition_mean"].abs() + 1e-12)
        ).replace([np.inf, -np.inf], np.nan)
        finite_dispersion = relative_dispersion.dropna()

        levels = np.unique(
            np.concatenate([contrasts["x_left"].to_numpy(), contrasts["x_right"].to_numpy()])
        )
        level_index = {value: index for index, value in enumerate(levels)}
        design = np.zeros((len(contrasts) + 1, len(levels)), dtype=float)
        target = np.zeros(len(contrasts) + 1, dtype=float)
        weights = np.ones(len(contrasts) + 1, dtype=float)
        for row_index, row in enumerate(contrasts.itertuples(index=False)):
            design[row_index, level_index[row.x_left]] = -1.0
            design[row_index, level_index[row.x_right]] = 1.0
            target[row_index] = float(row.normalized_delta)
            weights[row_index] = max(float(row.weight), 1.0)
        # Anchor the average effect at zero.  This is stronger than one
        # contrast equation but does not affect any reconstructed difference.
        design[-1, :] = 1.0 / len(levels)
        weights[-1] = max(float(weights[:-1].sum()), 1.0)
        root_weight = np.sqrt(weights)
        effect, *_ = np.linalg.lstsq(
            design * root_weight[:, None], target * root_weight, rcond=None
        )
        support = (
            pd.concat(
                [
                    contrasts.groupby("x_left")["weight"].sum(),
                    contrasts.groupby("x_right")["weight"].sum(),
                ],
                axis=1,
            )
            .fillna(0.0)
            .sum(axis=1)
            .reindex(levels, fill_value=0.0)
        )
        curve = pd.DataFrame({feature: levels, "effect": effect, "support": support.to_numpy()})

        return FrozenContrastResult(
            feature=feature,
            frozen_features=frozen,
            scale_by=scale_by,
            n_contexts=used_contexts,
            n_contrasts=len(contrasts),
            n_repeated_contrasts=int(repeated.sum()),
            repeated_fraction=float(repeated.mean()),
            separability_score=float(score),
            median_relative_dispersion=(
                float(finite_dispersion.median()) if len(finite_dispersion) else float("nan")
            ),
            contrasts=contrasts,
            effect_curve=curve,
        )

    def scan(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        """Compare additive contrasts with simple pair-scaled contrasts."""

        rows: list[dict[str, object]] = []
        for feature in X.columns:
            additive = self.fit(X, y, feature)
            best_scale: str | None = None
            best_scaled_score = float("nan")
            for scale_by in X.columns:
                if scale_by == feature:
                    continue
                scaled = self.fit(X, y, feature, scale_by=scale_by)
                if np.isnan(scaled.separability_score):
                    continue
                if np.isnan(best_scaled_score) or scaled.separability_score > best_scaled_score:
                    best_scaled_score = scaled.separability_score
                    best_scale = str(scale_by)
            rows.append(
                {
                    "feature": str(feature),
                    "n_contexts": additive.n_contexts,
                    "n_contrasts": additive.n_contrasts,
                    "repeated_fraction": additive.repeated_fraction,
                    "additive_score": additive.separability_score,
                    "best_scale_feature": best_scale,
                    "best_scaled_score": best_scaled_score,
                    "scaled_score_gain": (
                        best_scaled_score - additive.separability_score
                        if np.isfinite(best_scaled_score)
                        and np.isfinite(additive.separability_score)
                        else float("nan")
                    ),
                }
            )
        return pd.DataFrame(rows)
