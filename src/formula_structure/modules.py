"""Interpretable formula modules with iterative conditional selection."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import GroupKFold, KFold

from .analyzer import AnalysisResult


@dataclass(frozen=True)
class FormulaModule:
    """One candidate term ``phi(X)`` in a linear-in-parameters formula."""

    name: str
    features: tuple[str, ...]
    evaluator: Callable[[pd.DataFrame], np.ndarray]

    def evaluate(self, X: pd.DataFrame) -> np.ndarray:
        values = np.asarray(self.evaluator(X), dtype=float).reshape(-1)
        if len(values) != len(X):
            raise ValueError(f"Module {self.name!r} returned the wrong number of rows")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Module {self.name!r} returned non-finite values")
        return values


def transform_values(values: np.ndarray, family: str, parameter: float | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if family == "linear":
        return values
    if family == "quadratic":
        return values**2
    if family == "sqrt":
        return np.sqrt(np.maximum(values, 0.0))
    if family == "log":
        return np.log(np.maximum(values, 1e-12))
    if family == "inverse":
        return 1.0 / np.maximum(np.abs(values), 1e-12)
    if family == "power" and parameter is not None:
        return np.maximum(np.abs(values), 1e-12) ** float(parameter)
    return values


def _transform_name(feature: str, family: str, parameter: float | None) -> str:
    if family == "linear":
        return feature
    if family == "sqrt":
        return f"sqrt({feature})"
    if family == "quadratic":
        return f"{feature}^2"
    if family == "log":
        return f"log({feature})"
    if family == "inverse":
        return f"1/{feature}"
    if family == "power" and parameter is not None:
        return f"{feature}^{parameter:.4g}"
    return feature


def guided_modules(result: AnalysisResult, *, max_order: int = 3) -> list[FormulaModule]:
    """Convert structural findings into modules without flattening ``g(x,y)``.

    Only interactions classified as multiplicative-like become product terms.
    Nonseparable blocks remain absent instead of being incorrectly replaced by
    a raw product.  Confirmed pairs may be extended hierarchically to triples.
    """

    feature_rows = result.features.set_index("feature")
    main: dict[str, FormulaModule] = {}
    modules: list[FormulaModule] = []
    for feature, row in feature_rows.iterrows():
        if not bool(row.get("active", True)):
            continue
        family = str(row["shape"])
        parameter = row.get("shape_parameter")
        parameter = None if pd.isna(parameter) else float(parameter)
        module = FormulaModule(
            _transform_name(str(feature), family, parameter),
            (str(feature),),
            lambda frame, feature=str(feature), family=family, parameter=parameter: (
                transform_values(frame[feature].to_numpy(dtype=float), family, parameter)
            ),
        )
        main[str(feature)] = module
        modules.append(module)

    pair_modules: list[FormulaModule] = []
    if len(result.interactions):
        for row in result.interactions.itertuples(index=False):
            if not bool(row.detected) or str(row.kind) != "multiplicative-like":
                continue
            left, right = str(row.feature_1), str(row.feature_2)
            if left not in main or right not in main:
                continue
            left_module, right_module = main[left], main[right]
            pair = FormulaModule(
                f"({left_module.name})*({right_module.name})",
                (left, right),
                lambda frame, left_module=left_module, right_module=right_module: (
                    left_module.evaluate(frame) * right_module.evaluate(frame)
                ),
            )
            pair_modules.append(pair)
            modules.append(pair)

    if max_order >= 3:
        for pair in pair_modules:
            for feature, module in main.items():
                if feature in pair.features:
                    continue
                modules.append(
                    FormulaModule(
                        f"({pair.name})*({module.name})",
                        (*pair.features, feature),
                        lambda frame, pair=pair, module=module: (
                            pair.evaluate(frame) * module.evaluate(frame)
                        ),
                    )
                )
    return modules


def initial_exponents_from_analysis(
    result: AnalysisResult,
    features: Iterable[str] | None = None,
) -> dict[str, float]:
    """Return one interpretable exponent seed per requested feature."""

    rows = result.features.set_index("feature")
    requested = list(rows.index.astype(str)) if features is None else [str(x) for x in features]
    named = {"linear": 1.0, "sqrt": 0.5, "quadratic": 2.0, "inverse": -1.0}
    exponents: dict[str, float] = {}
    for feature in requested:
        if feature not in rows.index:
            raise KeyError(f"Feature {feature!r} is absent from the analysis result")
        row = rows.loc[feature]
        if str(row["shape"]) == "power" and pd.notna(row["shape_parameter"]):
            exponents[feature] = float(row["shape_parameter"])
        else:
            exponents[feature] = named.get(str(row["shape"]), 1.0)
    return exponents


def module_graph_from_selection(
    result: AnalysisResult,
    selected_modules: Iterable[FormulaModule],
) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Extract a compact node/edge/triple graph from selected formula modules.

    The ordering follows the structural feature ranking.  Triples are
    hierarchical candidates produced from a detected pair and a third active
    feature; they are not evidence of a separately estimated third-order ALE
    effect.
    """

    selected = list(selected_modules)
    feature_order = list(result.features["feature"].astype(str))
    used = {feature for module in selected for feature in module.features}
    if not used:
        used = set(feature_order[: min(3, len(feature_order))])
    features = [feature for feature in feature_order if feature in used]
    pairs: list[tuple[str, str]] = []
    triples: list[tuple[str, str, str]] = []
    for module in selected:
        module_features = tuple(str(feature) for feature in module.features)
        if len(module_features) == 2 and module_features not in pairs:
            pairs.append(module_features)
        elif len(module_features) == 3:
            if module_features not in triples:
                triples.append(module_features)
            parent_pair = module_features[:2]
            if parent_pair not in pairs:
                pairs.append(parent_pair)
    return features, pairs, triples


def expanded_graph_modules(
    features: Iterable[str],
    *,
    pairs: Iterable[tuple[str, str]] = (),
    triples: Iterable[tuple[str, str, str]] = (),
    seeded_exponents: dict[str, float] | None = None,
) -> list[FormulaModule]:
    """Build a compact multi-shape dictionary on a discovered graph.

    A single transform per variable is too restrictive for engineering
    expressions such as ``(b + sqrt(b)) * d * h``.  This function keeps the
    graph fixed, but gives every node a small automatic dictionary containing
    ``x``, ``sqrt(x)``, ``x²``, and the exponent estimated by the structural
    analyzer.  Pair blocks use every distinct transform combination.  Triple
    blocks remain tractable: they contain the all-linear product plus variants
    where one coordinate at a time uses a nonlinear transform.

    The returned modules are still candidates.  ``IterativeModuleRegressor``
    must select them by out-of-fold improvement before a final formula is fit.
    """

    ordered_features = tuple(dict.fromkeys(str(feature) for feature in features))
    exponent_seeds = dict(seeded_exponents or {})
    transforms: dict[str, list[FormulaModule]] = {}
    modules: list[FormulaModule] = []

    for feature in ordered_features:
        specifications: list[tuple[str, float]] = [
            (feature, 1.0),
            (f"sqrt({feature})", 0.5),
            (f"{feature}^2", 2.0),
        ]
        seeded = float(exponent_seeds.get(feature, 1.0))
        if all(abs(seeded - exponent) > 0.08 for _, exponent in specifications):
            specifications.append((f"{feature}^{seeded:.4g}", seeded))

        feature_modules: list[FormulaModule] = []
        for name, exponent in specifications:
            module = FormulaModule(
                name,
                (feature,),
                lambda frame, feature=feature, exponent=exponent: (
                    np.maximum(np.abs(frame[feature].to_numpy(dtype=float)), 1e-12) ** exponent
                ),
            )
            feature_modules.append(module)
            modules.append(module)
        transforms[feature] = feature_modules

    normalized_pairs: list[tuple[str, str]] = []
    for left, right in pairs:
        pair = (str(left), str(right))
        if pair[0] not in transforms or pair[1] not in transforms or pair[0] == pair[1]:
            continue
        if pair not in normalized_pairs and pair[::-1] not in normalized_pairs:
            normalized_pairs.append(pair)
        for left_module in transforms[pair[0]]:
            for right_module in transforms[pair[1]]:
                modules.append(
                    FormulaModule(
                        f"({left_module.name})*({right_module.name})",
                        pair,
                        lambda frame, left_module=left_module, right_module=right_module: (
                            left_module.evaluate(frame) * right_module.evaluate(frame)
                        ),
                    )
                )

    for first, second, third in triples:
        triple = (str(first), str(second), str(third))
        if len(set(triple)) != 3 or any(feature not in transforms for feature in triple):
            continue
        primary = [transforms[feature][0] for feature in triple]
        variants: list[tuple[FormulaModule, FormulaModule, FormulaModule]] = [tuple(primary)]
        for position, feature in enumerate(triple):
            for alternative in transforms[feature][1:]:
                candidate = list(primary)
                candidate[position] = alternative
                variants.append(tuple(candidate))
        for first_module, second_module, third_module in variants:

            def evaluate_triple(
                frame: pd.DataFrame,
                first_module: FormulaModule = first_module,
                second_module: FormulaModule = second_module,
                third_module: FormulaModule = third_module,
            ) -> np.ndarray:
                return (
                    first_module.evaluate(frame)
                    * second_module.evaluate(frame)
                    * third_module.evaluate(frame)
                )

            modules.append(
                FormulaModule(
                    f"({first_module.name})*({second_module.name})*({third_module.name})",
                    triple,
                    evaluate_triple,
                )
            )

    # A graph can mention the same algebraic module through more than one
    # hierarchical path.  Keep insertion order while removing duplicates.
    return list({module.name: module for module in modules}.values())


def _mape(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - y) / np.maximum(np.abs(y), 1e-12)))


class IterativeModuleRegressor:
    """Select formula modules conditionally, then estimate their coefficients.

    Selection is leakage-safe: every candidate must improve out-of-fold MAPE
    relative to the modules already selected.  The final coefficients may be
    fitted by ordinary least squares or tuned directly for MAPE with Optuna.
    """

    def __init__(
        self,
        modules: Iterable[FormulaModule],
        *,
        max_terms: int = 4,
        min_relative_gain: float = 0.005,
        n_splits: int = 4,
        fit_objective: str = "mse",
        coefficient_method: str = "ols",
        optuna_trials: int = 250,
        random_state: int = 42,
    ) -> None:
        self.modules = tuple(modules)
        self.max_terms = int(max_terms)
        self.min_relative_gain = float(min_relative_gain)
        self.n_splits = int(n_splits)
        self.fit_objective = str(fit_objective)
        self.coefficient_method = coefficient_method
        self.optuna_trials = int(optuna_trials)
        self.random_state = int(random_state)
        self.selected_modules_: list[FormulaModule] = []
        self.selection_history_: list[dict[str, float | str | int]] = []
        self.oof_residuals_: np.ndarray | None = None
        self.selected_oof_mape_: float | None = None
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None

    def _splits(self, size: int, groups: np.ndarray | None) -> list[tuple[np.ndarray, np.ndarray]]:
        if groups is not None and len(np.unique(groups)) >= 2:
            count = min(self.n_splits, len(np.unique(groups)))
            return list(GroupKFold(count).split(np.arange(size), groups=groups))
        count = min(self.n_splits, size)
        if count < 2:
            raise ValueError("At least two rows are required")
        return list(
            KFold(count, shuffle=True, random_state=self.random_state).split(np.arange(size))
        )

    @staticmethod
    def _matrix(X: pd.DataFrame, modules: Iterable[FormulaModule]) -> np.ndarray:
        values = [module.evaluate(X) for module in modules]
        return np.column_stack(values) if values else np.empty((len(X), 0), dtype=float)

    def _fit_linear(self, design: np.ndarray, target: np.ndarray) -> LinearRegression:
        model = LinearRegression()
        if self.fit_objective == "mse":
            return model.fit(design, target)
        if self.fit_objective == "relative":
            weights = 1.0 / np.maximum(np.abs(target), 1e-12) ** 2
            return model.fit(design, target, sample_weight=weights)
        raise ValueError("fit_objective must be 'mse' or 'relative'")

    def _oof_score(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        modules: list[FormulaModule],
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        return _mape(y, self._oof_prediction(X, y, modules, splits))

    def _oof_prediction(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        modules: list[FormulaModule],
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        design = self._matrix(X, modules)
        prediction = np.full(len(y), np.nan)
        for train, test in splits:
            if design.shape[1] == 0:
                if self.fit_objective == "relative":
                    weights = 1.0 / np.maximum(np.abs(y[train]), 1e-12) ** 2
                    prediction[test] = float(np.average(y[train], weights=weights))
                else:
                    prediction[test] = float(np.mean(y[train]))
            else:
                model = self._fit_linear(design[train], y[train])
                prediction[test] = model.predict(design[test])
        return prediction

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        groups: pd.Series | np.ndarray | None = None,
    ) -> IterativeModuleRegressor:
        target = np.asarray(y, dtype=float)
        group_values = None if groups is None else np.asarray(groups)
        splits = self._splits(len(X), group_values)
        selected: list[FormulaModule] = []
        remaining = list(self.modules)
        current_score = self._oof_score(X, target, selected, splits)
        self.selection_history_ = []

        while remaining and len(selected) < self.max_terms:
            scored = [
                (self._oof_score(X, target, [*selected, module], splits), module)
                for module in remaining
            ]
            candidate_score, candidate = min(scored, key=lambda item: item[0])
            gain = (current_score - candidate_score) / max(current_score, 1e-12)
            self.selection_history_.append(
                {
                    "step": len(selected) + 1,
                    "candidate": candidate.name,
                    "previous_oof_mape": current_score,
                    "candidate_oof_mape": candidate_score,
                    "relative_gain": gain,
                }
            )
            if gain < self.min_relative_gain:
                break
            selected.append(candidate)
            remaining.remove(candidate)
            current_score = candidate_score

        self.selected_modules_ = selected
        selected_oof = self._oof_prediction(X, target, selected, splits)
        self.oof_residuals_ = target - selected_oof
        self.selected_oof_mape_ = _mape(target, selected_oof)
        design = self._matrix(X, selected)
        if design.shape[1] == 0:
            self.intercept_ = float(np.mean(target))
            self.coef_ = np.empty(0)
            return self
        model = self._fit_linear(design, target)
        intercept = float(model.intercept_)
        coefficients = np.asarray(model.coef_, dtype=float)

        if self.coefficient_method == "relative_lad":
            # For positive engineering targets, weighted median regression
            # solves sum(|prediction-target|/|target|) exactly.  This is the
            # convex coefficient subproblem underlying MAPE, so there is no
            # reason to spend stochastic Optuna trials on fixed modules.
            weights = 1.0 / np.maximum(np.abs(target), 1e-12)
            lad = QuantileRegressor(
                quantile=0.5,
                alpha=0.0,
                fit_intercept=True,
                solver="highs",
            ).fit(design, target, sample_weight=weights)
            intercept = float(lad.intercept_)
            coefficients = np.asarray(lad.coef_, dtype=float)
        elif self.coefficient_method == "optuna_mape":
            try:
                import optuna
            except ImportError as error:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Optuna is required for coefficient_method='optuna_mape'"
                ) from error
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            y_scale = max(float(np.std(target)), float(np.mean(np.abs(target))), 1.0)
            feature_scale = np.maximum(np.std(design, axis=0), 1e-12)
            coefficient_scale = y_scale / feature_scale
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
            )
            initial = {"intercept": intercept}
            initial.update(
                {f"coef_{index}": float(value) for index, value in enumerate(coefficients)}
            )
            study.enqueue_trial(initial)

            def objective(trial: optuna.Trial) -> float:
                candidate_intercept = trial.suggest_float(
                    "intercept", intercept - 2.0 * y_scale, intercept + 2.0 * y_scale
                )
                candidate_coefficients = np.array(
                    [
                        trial.suggest_float(
                            f"coef_{index}",
                            float(value - 5.0 * coefficient_scale[index]),
                            float(value + 5.0 * coefficient_scale[index]),
                        )
                        for index, value in enumerate(coefficients)
                    ]
                )
                return _mape(target, candidate_intercept + design @ candidate_coefficients)

            study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)
            intercept = float(study.best_params["intercept"])
            coefficients = np.array(
                [study.best_params[f"coef_{index}"] for index in range(len(coefficients))]
            )
        elif self.coefficient_method != "ols":
            raise ValueError("coefficient_method must be 'ols', 'relative_lad', or 'optuna_mape'")

        self.intercept_ = intercept
        self.coef_ = coefficients
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("IterativeModuleRegressor is not fitted")
        return self.intercept_ + self._matrix(X, self.selected_modules_) @ self.coef_

    @property
    def formula(self) -> str:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("IterativeModuleRegressor is not fitted")
        terms = [f"{self.intercept_:.8g}"]
        terms.extend(
            f"({coefficient:.8g})*({module.name})"
            for coefficient, module in zip(self.coef_, self.selected_modules_)
        )
        return "y = " + " + ".join(terms)


class OptunaPowerFormulaRegressor:
    """Tune powers/shifts in a discovered module graph and then its weights.

    The formula remains linear in coefficients but nonlinear in the shape
    parameters.  For example a detected pair ``F,D`` and a separate ``G``
    produces candidate terms ``F^alpha``, ``D^beta``, ``G^gamma``, and
    ``F^alpha*D^beta``.  Optuna searches the exponents and small positive
    shifts; weighted least squares solves the coefficients inside every trial.
    A second Optuna stage may tune those coefficients directly for MAPE.
    """

    def __init__(
        self,
        features: Iterable[str],
        *,
        pairs: Iterable[tuple[str, str]] = (),
        triples: Iterable[tuple[str, str, str]] = (),
        initial_exponents: dict[str, float] | None = None,
        nonlinear_trials: int = 300,
        coefficient_trials: int = 180,
        random_state: int = 42,
    ) -> None:
        self.features = tuple(dict.fromkeys(str(feature) for feature in features))
        self.pairs = tuple(tuple(map(str, pair)) for pair in pairs)
        self.triples = tuple(tuple(map(str, triple)) for triple in triples)
        self.initial_exponents = dict(initial_exponents or {})
        self.nonlinear_trials = int(nonlinear_trials)
        self.coefficient_trials = int(coefficient_trials)
        self.random_state = int(random_state)
        self.scales_: dict[str, float] = {}
        self.exponents_: dict[str, float] = {}
        self.shifts_: dict[str, float] = {}
        self.term_names_: list[str] = []
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None

    def _basis(
        self,
        X: pd.DataFrame,
        exponents: dict[str, float],
        shifts: dict[str, float],
    ) -> tuple[np.ndarray, list[str]]:
        transformed: dict[str, np.ndarray] = {}
        labels: dict[str, str] = {}
        for feature in self.features:
            values = X[feature].to_numpy(dtype=float)
            shifted = np.maximum(values + shifts[feature], 1e-12) / self.scales_[feature]
            transformed[feature] = shifted ** exponents[feature]
            labels[feature] = (
                f"(({feature}+{shifts[feature]:.5g})/{self.scales_[feature]:.5g})"
                f"^{exponents[feature]:.5g}"
            )
        values = [transformed[feature] for feature in self.features]
        names = [labels[feature] for feature in self.features]
        for left, right in self.pairs:
            values.append(transformed[left] * transformed[right])
            names.append(f"({labels[left]})*({labels[right]})")
        for first, second, third in self.triples:
            values.append(transformed[first] * transformed[second] * transformed[third])
            names.append(f"({labels[first]})*({labels[second]})*({labels[third]})")
        return np.column_stack(values), names

    @staticmethod
    def _relative_fit(design: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
        weights = 1.0 / np.maximum(np.abs(target), 1e-12) ** 2
        model = LinearRegression().fit(design, target, sample_weight=weights)
        return float(model.intercept_), np.asarray(model.coef_, dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> OptunaPowerFormulaRegressor:
        if not self.features:
            raise ValueError("At least one active feature is required")
        try:
            import optuna
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("Optuna is required for OptunaPowerFormulaRegressor") from error
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        target = np.asarray(y, dtype=float)
        for feature in self.features:
            values = X[feature].to_numpy(dtype=float)
            positive = values[values > 0]
            self.scales_[feature] = max(float(np.median(np.abs(values))), 1e-12)
            minimum = float(np.min(positive)) if len(positive) else 1.0
            # Engineering formulas in this project use small offsets such as
            # P+0.79.  The cap avoids a large unidentifiable shift/power ridge.
            self.shifts_[feature] = max(1.0, min(5.0, 0.25 * minimum))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        initial: dict[str, float] = {}
        for feature in self.features:
            initial[f"alpha::{feature}"] = float(self.initial_exponents.get(feature, 1.0))
            initial[f"shift::{feature}"] = 0.0
        study.enqueue_trial(initial)

        def nonlinear_objective(trial: optuna.Trial) -> float:
            exponents = {
                feature: trial.suggest_float(f"alpha::{feature}", -2.0, 3.0)
                for feature in self.features
            }
            shifts = {
                feature: trial.suggest_float(f"shift::{feature}", 0.0, self.shifts_[feature])
                for feature in self.features
            }
            design, _ = self._basis(X, exponents, shifts)
            if not np.all(np.isfinite(design)):
                return float("inf")
            intercept, coefficients = self._relative_fit(design, target)
            return _mape(target, intercept + design @ coefficients)

        study.optimize(nonlinear_objective, n_trials=self.nonlinear_trials, show_progress_bar=False)
        self.exponents_ = {
            feature: float(study.best_params[f"alpha::{feature}"]) for feature in self.features
        }
        self.shifts_ = {
            feature: float(study.best_params[f"shift::{feature}"]) for feature in self.features
        }
        design, self.term_names_ = self._basis(X, self.exponents_, self.shifts_)
        intercept, coefficients = self._relative_fit(design, target)

        if self.coefficient_trials > 0:
            y_scale = max(float(np.std(target)), float(np.mean(np.abs(target))), 1.0)
            feature_scale = np.maximum(np.std(design, axis=0), 1e-12)
            coefficient_scale = y_scale / feature_scale
            coefficient_study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state + 1),
            )
            initial_coefficients = {"intercept": intercept}
            initial_coefficients.update(
                {f"coef_{index}": float(value) for index, value in enumerate(coefficients)}
            )
            coefficient_study.enqueue_trial(initial_coefficients)

            def coefficient_objective(trial: optuna.Trial) -> float:
                candidate_intercept = trial.suggest_float(
                    "intercept", intercept - y_scale, intercept + y_scale
                )
                candidate_coefficients = np.array(
                    [
                        trial.suggest_float(
                            f"coef_{index}",
                            float(value - 3.0 * coefficient_scale[index]),
                            float(value + 3.0 * coefficient_scale[index]),
                        )
                        for index, value in enumerate(coefficients)
                    ]
                )
                return _mape(target, candidate_intercept + design @ candidate_coefficients)

            coefficient_study.optimize(
                coefficient_objective,
                n_trials=self.coefficient_trials,
                show_progress_bar=False,
            )
            intercept = float(coefficient_study.best_params["intercept"])
            coefficients = np.array(
                [
                    coefficient_study.best_params[f"coef_{index}"]
                    for index in range(len(coefficients))
                ]
            )

        self.intercept_ = intercept
        self.coef_ = coefficients
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("OptunaPowerFormulaRegressor is not fitted")
        design, _ = self._basis(X, self.exponents_, self.shifts_)
        return self.intercept_ + design @ self.coef_

    @property
    def formula(self) -> str:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("OptunaPowerFormulaRegressor is not fitted")
        terms = [f"{self.intercept_:.8g}"]
        terms.extend(
            f"({coefficient:.8g})*({name})"
            for coefficient, name in zip(self.coef_, self.term_names_)
        )
        return "y = " + " + ".join(terms)
