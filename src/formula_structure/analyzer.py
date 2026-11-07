"""High-level structural analysis for engineering formula discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from .ale import FirstOrderALE, SecondOrderALE, first_order_ale, second_order_ale
from .splines import SplineRegressor
from .transforms import fit_shape, format_transform


@dataclass
class AnalysisResult:
    summary: dict[str, Any]
    features: pd.DataFrame
    interactions: pd.DataFrame
    ale_curves: dict[str, FirstOrderALE]
    interaction_surfaces: dict[str, SecondOrderALE]
    templates: list[str]

    def save(self, output_dir: str | Path) -> None:
        """Save machine-readable tables and a concise Markdown report."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(self.summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.features.to_csv(output / "feature_effects.csv", index=False)
        self.interactions.to_csv(output / "interactions.csv", index=False)

        curves: list[dict[str, Any]] = []
        for feature, curve in self.ale_curves.items():
            curves.extend(
                {
                    "feature": feature,
                    "x": float(x),
                    "effect": float(effect),
                    "count": int(count),
                }
                for x, effect, count in zip(curve.centers, curve.effects, curve.counts)
            )
        pd.DataFrame(curves).to_csv(output / "ale_curves.csv", index=False)

        lines = [
            "# Structural analysis",
            "",
            f"Surrogate out-of-fold R²: **{self.summary['surrogate_oof_r2']:.3f}**.",
            f"Surrogate out-of-fold MAE: **{self.summary['surrogate_oof_mae']:.3f}**.",
            f"Additive spline MAE: **{self.summary['additive_cv_mae']:.3f}**.",
            f"Structural confidence: **{self.summary['confidence']}**.",
            "",
            "## Feature effects",
            "",
            self.features.to_markdown(index=False),
            "",
            "## Pair interactions",
            "",
            self.interactions.to_markdown(index=False),
            "",
            "## Suggested formula templates",
            "",
        ]
        lines.extend(f"- `{template}`" for template in self.templates)
        lines.extend(
            [
                "",
                "## Interpretation guardrails",
                "",
                "ALE effects describe the fitted surrogate inside occupied data regions. ",
                "A detected interaction is reported only when a tensor-product term also improves ",
                "held-out error consistently across folds. Suggested transforms are hypotheses ",
                "for symbolic regression, not physical laws.",
            ]
        )
        (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


class StructureAnalyzer:
    """Discover likely main-effect shapes and pairwise formula structure.

    The analyzer combines model-agnostic ALE diagnostics with cross-validated
    additive and tensor-product spline models.  The former stays near the
    observed joint distribution; the latter guards against declaring noisy
    visual patterns to be real interactions.
    """

    def __init__(
        self,
        *,
        n_bins: int = 10,
        n_splits: int = 5,
        min_interaction_gain: float = 0.01,
        min_interaction_strength: float = 0.015,
        multiplicative_rank_threshold: float = 0.85,
        random_state: int = 42,
    ) -> None:
        self.n_bins = int(n_bins)
        self.n_splits = int(n_splits)
        self.min_interaction_gain = float(min_interaction_gain)
        self.min_interaction_strength = float(min_interaction_strength)
        self.multiplicative_rank_threshold = float(multiplicative_rank_threshold)
        self.random_state = int(random_state)

    def _surrogate(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=350,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=1.0,
            random_state=self.random_state,
        )

    def _splits(
        self,
        X: np.ndarray,
        groups: np.ndarray | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if groups is not None:
            unique = np.unique(groups)
            count = min(self.n_splits, len(unique))
            if count >= 2:
                return list(
                    GroupKFold(
                        n_splits=count,
                        shuffle=True,
                        random_state=self.random_state,
                    ).split(X, groups=groups)
                )
        count = min(self.n_splits, max(2, len(X) // 20))
        return list(KFold(n_splits=count, shuffle=True, random_state=self.random_state).split(X))

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        groups: pd.Series | np.ndarray | None = None,
    ) -> AnalysisResult:
        if isinstance(X, pd.DataFrame):
            feature_names = [str(column) for column in X.columns]
            values = X.to_numpy(dtype=float)
        else:
            values = np.asarray(X, dtype=float)
            if values.ndim != 2:
                raise ValueError("X must be a two-dimensional matrix")
            feature_names = [f"x{index}" for index in range(values.shape[1])]
        target = np.asarray(y, dtype=float)
        group_values = None if groups is None else np.asarray(groups)

        valid = np.isfinite(values).all(axis=1) & np.isfinite(target)
        values, target = values[valid], target[valid]
        if group_values is not None:
            group_values = group_values[valid]
        if len(values) < 40:
            raise ValueError("At least 40 finite rows are required")
        if values.shape[1] < 1:
            raise ValueError("At least one feature is required")

        splits = self._splits(values, group_values)
        oof = np.full(len(target), np.nan)
        fold_models: list[tuple[np.ndarray, HistGradientBoostingRegressor]] = []
        for train, test in splits:
            model = self._surrogate().fit(values[train], target[train])
            oof[test] = model.predict(values[test])
            fold_models.append((test, model))

        surrogate_mae = float(mean_absolute_error(target, oof))
        surrogate_r2 = float(r2_score(target, oof))
        target_scale = max(float(np.std(target)), 1e-12)
        final_model = self._surrogate().fit(values, target)

        additive_oof = np.full(len(target), np.nan)
        fold_additive_mae: list[float] = []
        for train, test in splits:
            model = SplineRegressor().fit(values[train], target[train])
            prediction = model.predict(values[test])
            additive_oof[test] = prediction
            fold_additive_mae.append(float(mean_absolute_error(target[test], prediction)))
        additive_mae = float(mean_absolute_error(target, additive_oof))
        smooth_main_model = SplineRegressor().fit(values, target)

        log_additive_mae = np.nan
        if np.all(target > 0):
            log_oof = np.full(len(target), np.nan)
            for train, test in splits:
                model = SplineRegressor().fit(values[train], np.log(target[train]))
                log_oof[test] = np.exp(model.predict(values[test]))
            log_additive_mae = float(mean_absolute_error(target, log_oof))

        feature_rows: list[dict[str, Any]] = []
        ale_curves: dict[str, FirstOrderALE] = {}
        for index, name in enumerate(feature_names):
            # Smooth additive projection is deliberately used for first-order
            # shape estimation. Tree surrogates are excellent interaction
            # detectors, but their step functions bias derivative-based power
            # estimates toward square-root-like shapes.
            curve = first_order_ale(smooth_main_model.predict, values, index, self.n_bins)
            ale_curves[name] = curve
            shape = fit_shape(curve.centers, curve.effects, curve.counts)

            permutation_losses: list[float] = []
            rng = np.random.default_rng(self.random_state + index)
            for test, model in fold_models:
                baseline = mean_absolute_error(target[test], model.predict(values[test]))
                for _ in range(3):
                    modified = values[test].copy()
                    modified[:, index] = rng.permutation(modified[:, index])
                    permutation_losses.append(
                        float(mean_absolute_error(target[test], model.predict(modified)) - baseline)
                    )
            ale_strength = float(
                np.sqrt(np.average(curve.effects**2, weights=curve.counts)) / target_scale
            )
            permutation_increase = float(np.mean(permutation_losses))
            feature_rows.append(
                {
                    "feature": name,
                    "ale_strength": ale_strength,
                    "permutation_mae_increase": permutation_increase,
                    "permutation_strength": permutation_increase / target_scale,
                    "active": bool(ale_strength >= 0.01 and permutation_increase > 0.0),
                    "shape": shape.family,
                    "shape_parameter": shape.parameter,
                    "shape_r2": shape.score,
                    "power_alpha_estimate": shape.power_alpha,
                    "power_alpha_low": shape.power_alpha_low,
                    "power_alpha_high": shape.power_alpha_high,
                    "power_r2": shape.power_r2,
                    "monotonicity": shape.monotonicity,
                }
            )

        interaction_rows: list[dict[str, Any]] = []
        surfaces: dict[str, SecondOrderALE] = {}
        for left, right in combinations(range(values.shape[1]), 2):
            pair_oof = np.full(len(target), np.nan)
            improvements: list[bool] = []
            for fold_index, (train, test) in enumerate(splits):
                model = SplineRegressor(pair=(left, right)).fit(values[train], target[train])
                prediction = model.predict(values[test])
                pair_oof[test] = prediction
                pair_mae = float(mean_absolute_error(target[test], prediction))
                improvements.append(pair_mae < fold_additive_mae[fold_index])
            pair_mae = float(mean_absolute_error(target, pair_oof))
            gain = max(0.0, (additive_mae - pair_mae) / max(additive_mae, 1e-12))

            surface = second_order_ale(
                final_model.predict,
                values,
                left,
                right,
                n_bins=max(4, self.n_bins - 2),
            )
            key = f"{feature_names[left]}__{feature_names[right]}"
            surfaces[key] = surface
            stability = float(np.mean(improvements))
            marginal_detected = bool(
                gain >= self.min_interaction_gain
                and surface.strength >= self.min_interaction_strength
                and stability >= 0.6
            )
            interaction_rows.append(
                {
                    "feature_1": feature_names[left],
                    "feature_2": feature_names[right],
                    "additive_cv_mae": additive_mae,
                    "pair_cv_mae": pair_mae,
                    "cv_gain": gain,
                    "fold_stability": stability,
                    "ale_strength": surface.strength,
                    "rank1_fraction": surface.rank1_fraction,
                    "marginal_detected": marginal_detected,
                }
            )

        features = (
            pd.DataFrame(feature_rows)
            .sort_values(["ale_strength", "permutation_mae_increase"], ascending=False)
            .reset_index(drop=True)
        )
        interactions = pd.DataFrame(interaction_rows)
        if len(interactions):
            interactions = interactions.sort_values(
                ["marginal_detected", "cv_gain", "ale_strength"], ascending=False
            ).reset_index(drop=True)

            # Forward conditional selection prevents a correlated proxy pair
            # from being reported merely because it partially reproduces a
            # stronger interaction already selected.
            selected_pairs: list[tuple[int, int]] = []
            current_mae = additive_mae
            current_fold_mae = list(fold_additive_mae)
            conditional_gain: dict[tuple[str, str], float] = {}
            conditional_stability: dict[tuple[str, str], float] = {}
            selected_keys: set[tuple[str, str]] = set()
            name_to_index = {name: index for index, name in enumerate(feature_names)}
            for _, row in interactions.iterrows():
                names = (str(row["feature_1"]), str(row["feature_2"]))
                if not bool(row["marginal_detected"]):
                    conditional_gain[names] = 0.0
                    conditional_stability[names] = 0.0
                    continue
                pair = (name_to_index[names[0]], name_to_index[names[1]])
                candidate_oof = np.full(len(target), np.nan)
                fold_candidate_mae: list[float] = []
                for train, test in splits:
                    model = SplineRegressor(pairs=selected_pairs + [pair]).fit(
                        values[train], target[train]
                    )
                    prediction = model.predict(values[test])
                    candidate_oof[test] = prediction
                    fold_candidate_mae.append(float(mean_absolute_error(target[test], prediction)))
                candidate_mae = float(mean_absolute_error(target, candidate_oof))
                gain_value = max(0.0, (current_mae - candidate_mae) / max(current_mae, 1e-12))
                stability_value = float(
                    np.mean([new < old for new, old in zip(fold_candidate_mae, current_fold_mae)])
                )
                conditional_gain[names] = gain_value
                conditional_stability[names] = stability_value
                if gain_value >= self.min_interaction_gain and stability_value >= 0.6:
                    selected_pairs.append(pair)
                    selected_keys.add(names)
                    current_mae = candidate_mae
                    current_fold_mae = fold_candidate_mae

            interactions["conditional_cv_gain"] = [
                conditional_gain[(str(row.feature_1), str(row.feature_2))]
                for row in interactions.itertuples()
            ]
            interactions["conditional_fold_stability"] = [
                conditional_stability[(str(row.feature_1), str(row.feature_2))]
                for row in interactions.itertuples()
            ]
            interactions["detected"] = [
                (str(row.feature_1), str(row.feature_2)) in selected_keys
                for row in interactions.itertuples()
            ]
            interactions["kind"] = [
                (
                    "multiplicative-like"
                    if bool(row.detected)
                    and float(row.rank1_fraction) >= self.multiplicative_rank_threshold
                    else "nonseparable"
                    if bool(row.detected)
                    else "none"
                )
                for row in interactions.itertuples()
            ]
            interactions = interactions.sort_values(
                ["detected", "conditional_cv_gain", "cv_gain"], ascending=False
            ).reset_index(drop=True)

        transform_by_feature = {
            row["feature"]: format_transform(
                str(row["feature"]), str(row["shape"]), row["shape_parameter"]
            )
            for row in feature_rows
        }
        strong = interactions[interactions["detected"]] if len(interactions) else interactions
        active_features = [
            name
            for name in feature_names
            if bool(features.set_index("feature").loc[name, "active"])
        ]
        if not active_features:
            active_features = [str(features.iloc[0]["feature"])]
        templates: list[str] = []
        if len(strong) == 0:
            templates.append(" + ".join(transform_by_feature[name] for name in active_features))
        else:
            used: set[str] = set()
            terms: list[str] = []
            for _, row in strong.iterrows():
                left, right = str(row["feature_1"]), str(row["feature_2"])
                if left in used or right in used:
                    continue
                used.update((left, right))
                if row["kind"] == "multiplicative-like":
                    terms.append(
                        f"({transform_by_feature[left]}) * ({transform_by_feature[right]})"
                    )
                else:
                    terms.append(f"g({left}, {right})")
            terms.extend(transform_by_feature[name] for name in active_features if name not in used)
            templates.append(" + ".join(terms))
        templates.append("c0 + " + templates[0])

        confidence = "high" if surrogate_r2 >= 0.8 else "medium" if surrogate_r2 >= 0.5 else "low"
        summary = {
            "n_rows": int(len(values)),
            "n_features": int(values.shape[1]),
            "surrogate_oof_r2": surrogate_r2,
            "surrogate_oof_mae": surrogate_mae,
            "additive_cv_mae": additive_mae,
            "log_additive_cv_mae": None if np.isnan(log_additive_mae) else log_additive_mae,
            "interaction_advantage": max(
                0.0,
                (additive_mae - surrogate_mae) / max(additive_mae, 1e-12),
            ),
            "confidence": confidence,
            "detected_interactions": (
                int(interactions["detected"].sum()) if len(interactions) else 0
            ),
        }
        return AnalysisResult(summary, features, interactions, ale_curves, surfaces, templates)
