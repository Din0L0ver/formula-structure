"""Cross-validated additive and pair-interaction spline surrogates."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import SplineTransformer, StandardScaler


class SplineRegressor:
    """Additive spline model with an optional tensor-product interaction."""

    def __init__(
        self,
        pair: tuple[int, int] | None = None,
        pairs: Iterable[tuple[int, int]] | None = None,
        n_knots: int = 5,
        degree: int = 3,
    ) -> None:
        if pair is not None and pairs is not None:
            raise ValueError("Use either pair or pairs, not both")
        self.pairs = tuple(pairs) if pairs is not None else (() if pair is None else (pair,))
        self.n_knots = n_knots
        self.degree = degree
        self.transforms_: list[SplineTransformer] = []
        self.scaler_: StandardScaler | None = None
        self.model_: RidgeCV | None = None

    def _fit_basis(self, X: np.ndarray) -> list[np.ndarray]:
        bases: list[np.ndarray] = []
        self.transforms_ = []
        for column in range(X.shape[1]):
            unique = np.unique(X[:, column])
            knots = max(2, min(self.n_knots, len(unique)))
            degree = max(1, min(self.degree, knots - 1))
            transform = SplineTransformer(
                n_knots=knots,
                degree=degree,
                include_bias=False,
                extrapolation="linear",
            )
            bases.append(transform.fit_transform(X[:, [column]]))
            self.transforms_.append(transform)
        return bases

    def _design(self, bases: list[np.ndarray]) -> np.ndarray:
        parts = list(bases)
        for pair in self.pairs:
            left, right = (bases[pair[0]], bases[pair[1]])
            interaction = np.einsum("ni,nj->nij", left, right).reshape(len(left), -1)
            parts.append(interaction)
        return np.column_stack(parts)

    def fit(self, X: np.ndarray, y: np.ndarray) -> SplineRegressor:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        design = self._design(self._fit_basis(X))
        self.scaler_ = StandardScaler()
        scaled = self.scaler_.fit_transform(design)
        self.model_ = RidgeCV(alphas=np.logspace(-4, 4, 13))
        self.model_.fit(scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.model_ is None:
            raise RuntimeError("SplineRegressor is not fitted")
        X = np.asarray(X, dtype=float)
        bases = [
            transform.transform(X[:, [column]]) for column, transform in enumerate(self.transforms_)
        ]
        return self.model_.predict(self.scaler_.transform(self._design(bases)))
