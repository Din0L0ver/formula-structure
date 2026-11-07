"""Synthetic datasets for examples, tests, and privacy-safe demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticEngineeringData:
    """A fully synthetic piecewise engineering-style regression problem.

    ``groups`` identifies repeated measurements of the same generated setting.
    It can be passed to grouped cross-validation to keep repetitions together.
    ``regime`` is provided only for evaluating regime recovery; discovery
    algorithms should not receive it as an input feature.
    """

    X: pd.DataFrame
    y: pd.Series
    regime: pd.Series
    groups: pd.Series


def make_synthetic_engineering_data(
    n_samples: int = 1600,
    *,
    repeats: int = 2,
    noise: float = 2.0,
    rounding_step: float = 5.0,
    random_state: int = 42,
) -> SyntheticEngineeringData:
    """Generate a positive, rounded target with four hidden formula regimes.

    The column names and coefficients are intentionally generic and generated
    independently of any private engineering dataset.  The hidden function is
    nonlinear in ``power`` and contains a regime-dependent ``flow * head``
    interaction, making the dataset useful for demonstrating the full library.
    """

    if n_samples < 40:
        raise ValueError("n_samples must be at least 40")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    if rounding_step < 0:
        raise ValueError("rounding_step must be non-negative")

    rng = np.random.default_rng(random_state)
    n_settings = int(np.ceil(n_samples / repeats))
    setting = pd.DataFrame(
        {
            "flow": rng.uniform(10.0, 300.0, n_settings),
            "head": rng.uniform(5.0, 120.0, n_settings),
            "power": rng.uniform(1.0, 80.0, n_settings),
            "mode": rng.integers(0, 2, n_settings).astype(float),
        }
    )
    frame = setting.loc[setting.index.repeat(repeats)].reset_index(drop=True).iloc[:n_samples]
    groups = np.repeat(np.arange(n_settings), repeats)[:n_samples]

    mode = frame["mode"].to_numpy(dtype=float)
    head = frame["head"].to_numpy(dtype=float)
    regime = np.select(
        [
            (mode == 0.0) & (head <= 55.0),
            (mode == 0.0) & (head > 55.0),
            (mode == 1.0) & (head <= 35.0),
            (mode == 1.0) & (head > 35.0),
        ],
        [1, 2, 3, 4],
    ).astype(int)

    intercept = np.array([0.0, 150.0, 450.0, 220.0, 900.0])
    flow_coefficient = np.array([0.0, 1.2, 2.0, 0.8, 1.5])
    head_coefficient = np.array([0.0, 0.8, 1.1, 0.4, 1.4])
    interaction_coefficient = np.array([0.0, 0.015, 0.025, 0.0, 0.018])

    flow = frame["flow"].to_numpy(dtype=float)
    power = frame["power"].to_numpy(dtype=float)
    target = (
        intercept[regime]
        + flow_coefficient[regime] * flow
        + head_coefficient[regime] * head
        + interaction_coefficient[regime] * flow * head
        + 9.0 * np.power(power, 0.7)
        + rng.normal(0.0, noise, n_samples)
    )
    if rounding_step > 0:
        target = np.round(target / rounding_step) * rounding_step

    return SyntheticEngineeringData(
        X=frame.reset_index(drop=True),
        y=pd.Series(target, name="target"),
        regime=pd.Series(regime, name="true_regime"),
        groups=pd.Series(groups, name="group"),
    )
