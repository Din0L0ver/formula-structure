# formula-structure

Research-oriented Python library for recovering
compact, interpretable formulas from tabular observations of a hidden
calculator or simulation.

It is designed for problems where one global expression may be insufficient:
the relationship can switch between regimes, contain nonlinear feature
effects, use multiplicative modules, and expose a rounded or noisy target.

> Status: research prototype. The library produces testable structural
> hypotheses, not proofs of physical laws.

## What it does

- discovers interpretable regime trees whose leaves contain local formulas;
- estimates first- and second-order accumulated local effects (ALE);
- tests whether pair interactions improve grouped out-of-fold error;
- distinguishes approximately multiplicative interactions using a rank-one
  diagnostic;
- converts structural findings into a constrained graph of formula modules;
- supports a compact power graph with Optuna-tuned exponents and shifts;
- supports an expanded multi-shape graph with exact weighted-LAD coefficient
  fitting for MAPE;
- analyses repeated or frozen-coordinate experiments with contrast graphs.

## Installation

Create a virtual environment and install the local checkout:

```bash
python -m pip install -e .
```

Install optional optimization and development dependencies:

```bash
python -m pip install -e ".[optimization,plots,dev]"
```

## Quick start: discover piecewise regimes

```python
from formula_structure import RegimeDiscoverer, make_synthetic_engineering_data

data = make_synthetic_engineering_data(random_state=7)

regimes = RegimeDiscoverer(
    max_regimes=5,
    min_samples_leaf=50,
    n_splits=3,
    random_state=7,
).fit(
    data.X[["flow", "head", "power"]],
    data.y,
    split_features=data.X[["mode", "head"]],
    groups=data.groups,
)

print(regimes.selected_regimes)
print(regimes.rules[["regime", "rule", "n_train", "train_mae"]])
```

Every terminal node is a separate local regression formula. The tree is a
gating model: its split rules describe where formula coefficients change; they
are not causal claims.

## Analyse one regime

```python
from formula_structure import StructureAnalyzer

labels = regimes.assign(data.X[["mode", "head"]])
mask = labels == 1

analysis = StructureAnalyzer(
    n_bins=8,
    n_splits=3,
    random_state=7,
).fit(
    data.X.loc[mask, ["flow", "head", "power"]],
    data.y.loc[mask],
    groups=data.groups.loc[mask],
)

print(analysis.features)
print(analysis.interactions)
print(analysis.templates)
```


## Method overview

```text
tabular observations
        |
        v
regime model tree
        |
        v
local surrogate + ALE + spline interaction tests
        |
        v
dependency graph and candidate modules
        |
        +--> power graph: continuous exponents/shifts + Optuna
        |
        +--> expanded graph: x, sqrt(x), x^2, x^alpha + weighted LAD
```

For fixed modules, minimizing MAPE over coefficients is a weighted least
absolute deviations problem:

```text
min_beta sum_i |y_i - Phi_i beta| / |y_i|
       = sum_i w_i |y_i - Phi_i beta|,  w_i = 1 / |y_i|.
```

The expanded graph solves this convex coefficient subproblem directly. The
word “exact” applies to the coefficients for the selected modules, not to the
global structure search.

See [Methodology and scientific basis](docs/METHODOLOGY.md) for the detailed
decision rules, limitations, and related research.


## Repository layout

```text
src/formula_structure/   library code
docs/                    methodology and Russian-language overview
```

## Limitations

- ALE describes the fitted surrogate inside observed regions; it is not a
  causal effect.
- Different formulas can be observationally indistinguishable on a narrow
  input range.
- Interaction and regime searches are greedy and do not guarantee a globally
  optimal symbolic expression.
- A high rank-one interaction score is compatible with a product of functions,
  but does not prove a specific product such as `x * z`.

## Language

A Russian overview is available in [docs/README_RU.md](docs/README_RU.md).

