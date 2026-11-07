# Methodology and scientific basis

`formula-structure` is a small research library for narrowing the hypothesis
space before symbolic regression.  It does not claim to discover a physical
law directly.  It produces testable structural hypotheses and attaches
out-of-sample evidence to them.

## What it reports

- first-order accumulated local effect curves;
- predictive strength and a low-complexity shape hypothesis for every feature;
- a free power-exponent estimate and a practical-equivalence interval;
- pairwise second-order ALE interaction strength;
- cross-validated gain from adding a tensor-product pair term;
- stability of that gain across folds;
- whether an interaction surface is approximately rank one and therefore
  compatible with a product of one-dimensional terms;
- compact templates that can be passed to PySR, Optuna, or a constrained
  least-squares model.

## Example

```python
from formula_structure import StructureAnalyzer

result = StructureAnalyzer(random_state=42).fit(
    X[["Q", "H", "P"]],
    y,
    groups=pump_identity,
)

print(result.features)
print(result.interactions)
print(result.templates)
result.save("structure_report")
```

## Automatic regime discovery

`RegimeDiscoverer` searches for an interpretable gating tree whose terminal
nodes contain separate local formulas:

```python
from formula_structure import RegimeDiscoverer

regimes = RegimeDiscoverer(max_regimes=7).fit(
    X[["Q", "H", "P"]],
    y,
    split_features=X[["speed", "H"]],
    groups=pump_identity,
)

print(regimes.selected_regimes)
print(regimes.rules)
regimes.save("regime_report")
```

Candidate thresholds are proposed from cumulative parameter-instability
scores.  Every split therefore asks whether the coefficients of a local
formula change across a gating variable.  Grouped cross-validation chooses
the smallest number of local formulas within a configurable tolerance of the
best predictive result.

## Decision rule

An interaction is not accepted from a picture alone.  It must:

1. have a non-negligible second-order ALE amplitude;
2. improve grouped cross-validation relative to an additive spline model;
3. improve a majority of folds;
4. remain useful after stronger interactions have already been included.

This conditional forward step reduces proxy interactions caused by correlated
features.

## Scientific basis

- D. W. Apley and J. Zhu, *Visualizing the Effects of Predictor Variables in
  Black Box Supervised Learning Models*, JRSS B 82(4), 2020.
  https://doi.org/10.1111/rssb.12377
- J. H. Friedman and B. E. Popescu, *Predictive Learning via Rule Ensembles*,
  Annals of Applied Statistics 2(3), 2008.
  https://doi.org/10.1214/07-AOAS148
- S.-M. Udrescu and M. Tegmark, *AI Feynman: A Physics-Inspired Method for
  Symbolic Regression*, Science Advances 6(16), 2020.
  https://doi.org/10.1126/sciadv.aay2631
- S.-M. Udrescu et al., *AI Feynman 2.0: Pareto-Optimal Symbolic Regression
  Exploiting Graph Modularity*, NeurIPS 2020.
  https://proceedings.neurips.cc/paper/2020/hash/33a854e247155d590883b93bca53848a-Abstract.html
- G. Hooker, *Generalized Functional ANOVA Diagnostics for High-Dimensional
  Functions of Dependent Variables*, JCGS 16(3), 2007.
  https://doi.org/10.1198/106186007X237892
- A. Zeileis, T. Hothorn, and K. Hornik, *Model-Based Recursive
  Partitioning*, JCGS 17(2), 2008.
  https://doi.org/10.1198/106186008X319331
- J. R. Quinlan, *Learning with Continuous Classes*, AI'92, 1992.
  https://sci2s.ugr.es/keel/pdf/algorithm/congreso/1992-Quinlan-AI.pdf
- R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, *Adaptive
  Mixtures of Local Experts*, Neural Computation 3(1), 1991.
  https://doi.org/10.1162/neco.1991.3.1.79

## Limitations

- The conclusions describe a fitted surrogate, not causality.
- A high rank-one score is compatible with multiplication but is not a proof.
- Exponent estimates can be weakly identified on a narrow feature interval;
  use the reported equivalence interval rather than only the point estimate.
- Low surrogate cross-validation quality makes every downstream structural
  conclusion provisional.
- Grouped or otherwise leakage-safe folds are essential for repeated
  engineering catalog rows.

## Frozen contrasts and iterative formula modules

`FrozenContrastAnalyzer` implements the repeated-coordinate experiment used in
manual engineering analysis. For rows with identical context `z`, it records
`y(x2,z)-y(x1,z)`. The same transition is compared across all available
contexts, producing a separability score, coverage diagnostics, and an
anchored effect curve reconstructed from the graph of differences.

Passing `scale_by="z"` tests whether dividing the contrast by the frozen value
of `z` makes it reproducible. This is direct evidence for a multiplicative
module. If transitions are not repeated, the score is reported as unavailable
instead of extrapolating a structural claim.

`IterativeModuleRegressor` performs conditional forward selection: each term
must improve out-of-fold MAPE after all previously accepted terms. It then
fits a formula linear in module coefficients. `OptunaPowerFormulaRegressor`
keeps the discovered module graph but lets Optuna tune powers and small shifts,
followed by the coefficients. A detected pair plus a separate feature can
therefore produce a compact family such as

`c + a*F^alpha + b*F^alpha*D^beta + d*G^gamma`.

For higher-dimensional equipment, one transform per variable is often too
restrictive. `expanded_graph_modules` keeps the detected interaction graph but
allows a small dictionary at every node: `x`, `sqrt(x)`, `x^2`, and the power
estimated by `StructureAnalyzer`. Pair modules combine all distinct node
transforms. Triple modules use the all-linear product and variants in which one
coordinate at a time is nonlinear. This is sufficient to express terms such
as `d*h*(b + sqrt(b))` without opening an unrestricted symbolic search.

The expanded dictionary is pruned by `IterativeModuleRegressor` using
out-of-fold MAPE. Setting `fit_objective="relative"` makes every fold estimate
coefficients by relative weighted least squares. For fixed selected modules,
`coefficient_method="relative_lad"` solves the remaining MAPE coefficient
problem by weighted median regression. Optuna remains useful for the genuinely
nonlinear part of the problem—continuous exponents and shifts—while the convex
linear subproblem is solved directly.

