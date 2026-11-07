"""Interpretable discovery of piecewise engineering formula regimes.

The implementation is a compact model tree: axis-aligned gating rules split
the data, while every terminal node contains a local ridge-stabilized formula
rather than a constant.  The number of regimes is selected by grouped
cross-validation with a configurable tolerance preference for simpler trees.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold


class _FormulaBasis:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.scales_: np.ndarray | None = None
        self.column_names_: list[str] = []

    def fit(self, X: np.ndarray) -> _FormulaBasis:
        scales = np.nanmedian(np.abs(X), axis=0)
        scales[~np.isfinite(scales) | (scales < 1e-12)] = 1.0
        self.scales_ = scales
        self.column_names_ = ["1"]
        self.column_names_.extend(self.names)
        self.column_names_.extend(f"sqrt_abs({name})" for name in self.names)
        self.column_names_.extend(
            f"{self.names[left]}*{self.names[right]}"
            for left in range(len(self.names))
            for right in range(left + 1, len(self.names))
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scales_ is None:
            raise RuntimeError("Formula basis is not fitted")
        scaled = np.asarray(X, dtype=float) / self.scales_
        parts: list[np.ndarray] = [np.ones((len(scaled), 1)), scaled, np.sqrt(np.abs(scaled))]
        products = [
            (scaled[:, left] * scaled[:, right])[:, None]
            for left in range(scaled.shape[1])
            for right in range(left + 1, scaled.shape[1])
        ]
        parts.extend(products)
        return np.column_stack(parts)


@dataclass
class _LeafFit:
    beta: np.ndarray
    sse: float
    mae: float


@dataclass
class _Node:
    indices: np.ndarray
    fit: _LeafFit
    depth: int
    path: tuple[str, ...]
    split_feature: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class _GreedyModelTree:
    def __init__(
        self,
        formula_names: list[str],
        split_names: list[str],
        *,
        max_regimes: int,
        min_samples_leaf: int,
        n_thresholds: int,
        ridge: float,
        min_relative_gain: float,
    ) -> None:
        self.formula_names = formula_names
        self.split_names = split_names
        self.max_regimes = max_regimes
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.ridge = ridge
        self.min_relative_gain = min_relative_gain
        self.basis = _FormulaBasis(formula_names)
        self.design_: np.ndarray | None = None
        self.y_: np.ndarray | None = None
        self.split_X_: np.ndarray | None = None
        self.stage_roots_: dict[int, _Node] = {}

    def _fit_leaf(self, indices: np.ndarray) -> _LeafFit:
        assert self.design_ is not None and self.y_ is not None
        design = self.design_[indices]
        target = self.y_[indices]
        penalty = np.eye(design.shape[1]) * self.ridge
        penalty[0, 0] = 0.0
        beta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ target
        residual = design @ beta - target
        return _LeafFit(
            beta=beta,
            sse=float(np.sum(residual**2)),
            mae=float(np.mean(np.abs(residual))),
        )

    def _leaves(self, node: _Node) -> list[_Node]:
        if node.is_leaf:
            return [node]
        assert node.left is not None and node.right is not None
        return self._leaves(node.left) + self._leaves(node.right)

    def _thresholds(self, leaf: _Node, feature: int) -> np.ndarray:
        assert self.split_X_ is not None and self.design_ is not None and self.y_ is not None
        indices = leaf.indices
        values = self.split_X_[indices, feature]
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        n = len(sorted_values)
        if n < 2:
            return np.array([], dtype=float)
        valid_positions = np.arange(self.min_samples_leaf, n - self.min_samples_leaf + 1)
        valid_positions = valid_positions[
            sorted_values[valid_positions - 1] < sorted_values[valid_positions]
        ]
        if len(valid_positions) == 0:
            return np.array([], dtype=float)
        if len(valid_positions) <= self.n_thresholds:
            chosen = valid_positions
        else:
            # Parameter-instability score process, following the core idea of
            # model-based recursive partitioning. It proposes thresholds where
            # the local formula coefficients, not merely the target mean, move.
            design = self.design_[indices][order]
            residual = self.y_[indices][order] - design @ leaf.fit.beta
            scores = design * residual[:, None]
            cumulative = np.cumsum(scores, axis=0)
            total = cumulative[-1]
            position = valid_positions.astype(float)
            centered = cumulative[valid_positions - 1] - (position / n)[:, None] * total
            statistic = np.sum(centered**2, axis=1) / np.maximum(position * (n - position), 1.0)
            focused_count = max(1, self.n_thresholds * 3 // 4)
            focused = valid_positions[np.argsort(statistic)[-focused_count:]]
            broad_count = self.n_thresholds - len(focused)
            broad = (
                valid_positions[
                    np.linspace(0, len(valid_positions) - 1, broad_count).round().astype(int)
                ]
                if broad_count > 0
                else np.array([], dtype=int)
            )
            chosen = np.unique(np.concatenate([focused, broad]))
        return np.array(
            [(sorted_values[position - 1] + sorted_values[position]) / 2.0 for position in chosen],
            dtype=float,
        )

    def fit(self, X: np.ndarray, split_X: np.ndarray, y: np.ndarray) -> _GreedyModelTree:
        X = np.asarray(X, dtype=float)
        self.split_X_ = np.asarray(split_X, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        self.basis.fit(X)
        self.design_ = self.basis.transform(X)
        all_indices = np.arange(len(X))
        root = _Node(all_indices, self._fit_leaf(all_indices), 0, ())
        self.stage_roots_ = {1: copy.deepcopy(root)}

        while len(self._leaves(root)) < self.max_regimes:
            best: (
                tuple[
                    float,
                    _Node,
                    int,
                    float,
                    np.ndarray,
                    np.ndarray,
                    _LeafFit,
                    _LeafFit,
                ]
                | None
            ) = None
            for leaf in self._leaves(root):
                if len(leaf.indices) < 2 * self.min_samples_leaf:
                    continue
                for feature in range(self.split_X_.shape[1]):
                    values = self.split_X_[leaf.indices, feature]
                    for threshold in self._thresholds(leaf, feature):
                        left_indices = leaf.indices[values <= threshold]
                        right_indices = leaf.indices[values > threshold]
                        if (
                            len(left_indices) < self.min_samples_leaf
                            or len(right_indices) < self.min_samples_leaf
                        ):
                            continue
                        left_fit = self._fit_leaf(left_indices)
                        right_fit = self._fit_leaf(right_indices)
                        improvement = leaf.fit.sse - left_fit.sse - right_fit.sse
                        relative = improvement / max(leaf.fit.sse, 1e-12)
                        if relative < self.min_relative_gain:
                            continue
                        candidate = (
                            improvement,
                            leaf,
                            feature,
                            float(threshold),
                            left_indices,
                            right_indices,
                            left_fit,
                            right_fit,
                        )
                        if best is None or candidate[0] > best[0]:
                            best = candidate
            if best is None:
                break
            _, leaf, feature, threshold, left_idx, right_idx, left_fit, right_fit = best
            leaf.split_feature = feature
            leaf.threshold = threshold
            name = self.split_names[feature]
            leaf.left = _Node(
                left_idx,
                left_fit,
                leaf.depth + 1,
                leaf.path + (f"{name} <= {threshold:.6g}",),
            )
            leaf.right = _Node(
                right_idx,
                right_fit,
                leaf.depth + 1,
                leaf.path + (f"{name} > {threshold:.6g}",),
            )
            count = len(self._leaves(root))
            self.stage_roots_[count] = copy.deepcopy(root)
        return self

    def available_stage(self, requested: int) -> int:
        return max(stage for stage in self.stage_roots_ if stage <= requested)

    def _leaf_for_row(self, root: _Node, row: np.ndarray) -> _Node:
        node = root
        while not node.is_leaf:
            assert node.split_feature is not None and node.threshold is not None
            if row[node.split_feature] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node

    def predict(self, X: np.ndarray, split_X: np.ndarray, regimes: int) -> np.ndarray:
        stage = self.available_stage(regimes)
        root = self.stage_roots_[stage]
        design = self.basis.transform(np.asarray(X, dtype=float))
        split_X = np.asarray(split_X, dtype=float)
        prediction = np.empty(len(X), dtype=float)
        for index, row in enumerate(split_X):
            prediction[index] = float(design[index] @ self._leaf_for_row(root, row).fit.beta)
        return prediction

    def assignments(self, split_X: np.ndarray, regimes: int) -> np.ndarray:
        root = self.stage_roots_[self.available_stage(regimes)]
        leaves = self._leaves(root)
        lookup = {id(leaf): index + 1 for index, leaf in enumerate(leaves)}
        return np.array([lookup[id(self._leaf_for_row(root, row))] for row in split_X], dtype=int)

    def rules(self, regimes: int) -> pd.DataFrame:
        root = self.stage_roots_[self.available_stage(regimes)]
        rows: list[dict[str, Any]] = []
        assert self.basis.scales_ is not None
        scaled_names = [f"z{index}" for index in range(len(self.formula_names))]
        display_terms = ["1", *scaled_names]
        display_terms.extend(f"sqrt_abs({name})" for name in scaled_names)
        display_terms.extend(
            f"{scaled_names[left]}*{scaled_names[right]}"
            for left in range(len(scaled_names))
            for right in range(left + 1, len(scaled_names))
        )
        scaling_text = "; ".join(
            f"{scaled}=({name})/{scale:.8g}"
            for scaled, name, scale in zip(scaled_names, self.formula_names, self.basis.scales_)
        )
        for leaf_id, leaf in enumerate(self._leaves(root), start=1):
            coefficients = {
                name: float(value) for name, value in zip(self.basis.column_names_, leaf.fit.beta)
            }
            rows.append(
                {
                    "regime": leaf_id,
                    "rule": " AND ".join(leaf.path) if leaf.path else "all rows",
                    "n_train": int(len(leaf.indices)),
                    "train_mae": leaf.fit.mae,
                    "formula_scaled": "W = "
                    + " + ".join(
                        f"({coefficient:.10g})*{term}"
                        for coefficient, term in zip(leaf.fit.beta, display_terms)
                        if abs(float(coefficient)) > 1e-10
                    )
                    + f"; {scaling_text}",
                    "coefficients": json.dumps(coefficients, ensure_ascii=False),
                }
            )
        return pd.DataFrame(rows)


@dataclass
class RegimeDiscoveryResult:
    selected_regimes: int
    cv_scores: pd.DataFrame
    rules: pd.DataFrame
    assignments: np.ndarray
    model: _GreedyModelTree
    formula_columns: list[str]
    split_columns: list[str]

    def predict(self, X: pd.DataFrame, split_features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(
            X[self.formula_columns].to_numpy(dtype=float),
            split_features[self.split_columns].to_numpy(dtype=float),
            self.selected_regimes,
        )

    def assign(self, split_features: pd.DataFrame) -> np.ndarray:
        return self.model.assignments(
            split_features[self.split_columns].to_numpy(dtype=float),
            self.selected_regimes,
        )

    def save(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.cv_scores.to_csv(output / "regime_cv.csv", index=False)
        self.rules.to_csv(output / "regime_rules.csv", index=False)
        pd.DataFrame({"discovered_regime": self.assignments}).to_csv(
            output / "assignments.csv", index=False
        )
        lines = [
            "# Discovered formula regimes",
            "",
            f"Selected number of regimes: **{self.selected_regimes}**.",
            "",
            "## Cross-validation",
            "",
            self.cv_scores.to_markdown(index=False),
            "",
            "## Gating rules",
            "",
            self.rules.drop(columns=["coefficients", "formula_scaled"]).to_markdown(index=False),
            "",
            (
                "Each terminal node contains its own local formula. "
                "The split rules are a gating model, not causal claims."
            ),
        ]
        (output / "REGIMES.md").write_text("\n".join(lines), encoding="utf-8")


class RegimeDiscoverer:
    """Discover an interpretable gating tree for several local formulas."""

    def __init__(
        self,
        *,
        max_regimes: int = 6,
        min_samples_leaf: int = 80,
        n_thresholds: int = 32,
        n_splits: int = 4,
        ridge: float = 1e-4,
        min_relative_gain: float = 0.01,
        selection_tolerance: float = 0.10,
        random_state: int = 42,
    ) -> None:
        self.max_regimes = int(max_regimes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.n_thresholds = int(n_thresholds)
        self.n_splits = int(n_splits)
        self.ridge = float(ridge)
        self.min_relative_gain = float(min_relative_gain)
        self.selection_tolerance = float(selection_tolerance)
        self.random_state = int(random_state)

    def _splits(self, X: np.ndarray, groups: np.ndarray | None):
        if groups is not None and len(np.unique(groups)) >= self.n_splits:
            return list(
                GroupKFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                ).split(X, groups=groups)
            )
        return list(
            KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            ).split(X)
        )

    def _new_tree(
        self,
        formula_names: list[str],
        split_names: list[str],
        min_leaf: int,
    ) -> _GreedyModelTree:
        return _GreedyModelTree(
            formula_names,
            split_names,
            max_regimes=self.max_regimes,
            min_samples_leaf=min_leaf,
            n_thresholds=self.n_thresholds,
            ridge=self.ridge,
            min_relative_gain=self.min_relative_gain,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        split_features: pd.DataFrame,
        groups: pd.Series | np.ndarray | None = None,
    ) -> RegimeDiscoveryResult:
        formula_names = [str(column) for column in X.columns]
        split_names = [str(column) for column in split_features.columns]
        formula_X = X.to_numpy(dtype=float)
        gate_X = split_features.to_numpy(dtype=float)
        target = np.asarray(y, dtype=float)
        group_values = None if groups is None else np.asarray(groups)
        valid = (
            np.isfinite(formula_X).all(axis=1)
            & np.isfinite(gate_X).all(axis=1)
            & np.isfinite(target)
        )
        formula_X, gate_X, target = formula_X[valid], gate_X[valid], target[valid]
        if group_values is not None:
            group_values = group_values[valid]
        if len(target) < 4 * self.min_samples_leaf:
            raise ValueError("Not enough rows for stable regime discovery")

        records: list[dict[str, float | int]] = []
        fold_values: dict[int, list[float]] = {k: [] for k in range(1, self.max_regimes + 1)}
        for train, test in self._splits(formula_X, group_values):
            fold_min = max(20, min(self.min_samples_leaf, len(train) // (2 * self.max_regimes)))
            tree = self._new_tree(formula_names, split_names, fold_min).fit(
                formula_X[train], gate_X[train], target[train]
            )
            for regimes in range(1, self.max_regimes + 1):
                prediction = tree.predict(formula_X[test], gate_X[test], regimes)
                fold_values[regimes].append(float(mean_absolute_error(target[test], prediction)))
        for regimes, values in fold_values.items():
            records.append(
                {
                    "regimes": regimes,
                    "cv_mae": float(np.mean(values)),
                    "cv_mae_se": (
                        float(np.std(values, ddof=1) / np.sqrt(len(values)))
                        if len(values) > 1
                        else 0.0
                    ),
                }
            )
        cv = pd.DataFrame(records)
        best = cv.loc[cv["cv_mae"].idxmin()]
        threshold = float(best["cv_mae"] * (1.0 + self.selection_tolerance))
        eligible = cv[cv["cv_mae"] <= threshold]
        selected = int(eligible["regimes"].min())

        final_tree = self._new_tree(formula_names, split_names, self.min_samples_leaf).fit(
            formula_X, gate_X, target
        )
        selected = final_tree.available_stage(selected)
        assignments = final_tree.assignments(gate_X, selected)
        return RegimeDiscoveryResult(
            selected_regimes=selected,
            cv_scores=cv,
            rules=final_tree.rules(selected),
            assignments=assignments,
            model=final_tree,
            formula_columns=formula_names,
            split_columns=split_names,
        )
