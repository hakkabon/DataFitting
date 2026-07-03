# DataFitting — Methodology

This document describes the statistical methods implemented by each type in
the package, in enough detail to understand what the code computes and why.
Source files are noted in parentheses.

## 1. Polynomial least-squares regression (`Regression.swift`)

`Regression` fits a degree-`d` polynomial

```
ŷ(x) = β₀ + β₁x + β₂x² + ... + β_d x^d
```

to a set of `m` samples `(x_i, y_i)` by ordinary least squares — minimizing
the sum of squared residuals `Σ (y_i − ŷ(x_i))²`.

**Design matrix.** The samples are packed into an `m × (d+1)` design matrix

```
    | 1  x₀  x₀²  ...  x₀^d |
X = | 1  x₁  x₁²  ...  x₁^d |
    | ...                    |
    | 1  x_{m-1} ... x_{m-1}^d |
```

with response vector `Y = [y₀, y₁, ..., y_{m-1}]`.

**Normal equations.** The least-squares coefficient vector `β` is the
solution to the normal equations:

```
β = (XᵀX)⁻¹ Xᵀ Y
```

This is exactly what `init(points:degree:)` computes, using
[`Dimensional`](https://github.com/hakkabon/Dimensional) for the transpose,
multiply, and inverse operations. This is the same textbook approach as
[ordinary least squares / linear regression](http://en.wikipedia.org/wiki/Linear_regression),
generalized to multiple polynomial terms rather than just an intercept and
slope.

**Requirements.** You need `m > d` (strictly more samples than the number
of free coefficients `d+1` minus 1... concretely, `degree < points.count`),
or `XᵀX` is singular and the inverse doesn't exist. Because the inverse is
computed via a cofactor expansion (see `Dimensional`), this approach is
appropriate for small-to-moderate degree polynomials; it is not a
numerically robust general-purpose least-squares solver (no QR or SVD
decomposition), so very high-degree fits or ill-conditioned data (e.g.
tightly clustered or duplicated x-values) can lose precision.

**Prediction.** `predict(point:)` evaluates `β₀ + β₁x + β₂x² + ... + β_d x^d`
at an arbitrary `x`, including outside the sampled range (extrapolation),
which is mathematically well-defined but statistically unreliable the
farther you go from the observed data.

## 2. LOESS — locally weighted scatterplot smoothing (`Loess.swift`)

`Loess` implements Cleveland's LOESS algorithm: instead of one global
polynomial, it fits a *new*, local, weighted linear regression around every
individual point, using only nearby samples, and uses that local fit's value
at that point as the smoothed estimate.

**Bandwidth window.** `bandwidth` (0, 1] is the fraction of *all* samples to
include in each local window. For `n` samples, the window size is
`round(bandwidth * n)` points — the `bandwidthInPoints` nearest neighbors
(in x) of each point being smoothed.

**Tricube weighting.** Within a local window, points are weighted by the
tricube function of their (normalized) distance from the point being
estimated:

```
w(u) = (1 − |u|³)³   for |u| < 1,   0 otherwise
```

where `u` is the distance to the farthest point in the window, scaled to
`[-1, 1]`. Nearby points get weight close to 1; points at the edge of the
window get weight close to 0 — this is what makes the fit "local" and
smooth rather than a hard cutoff.

**Weighted linear fit.** At each `x₀`, a weighted least-squares line
`ŷ = α + βx` is fit using the tricube weights (combined multiplicatively
with the current robustness weights, see below), via the closed-form
weighted simple linear regression formulas:

```
β = (E[xy] − E[x]E[y]) / (E[x²] − E[x]²)
α = E[y] − β E[x]
```

where `E[·]` denotes the weighted mean over the local window. This is the
univariate case of [weighted least squares](http://en.wikipedia.org/wiki/Weighted_least_squares).

**Robustness iterations.** Outliers can distort a local linear fit, so
LOESS re-weights points by how large their residual was and re-fits:

1. Fit as above with all robustness weights = 1.
2. Compute residuals `|y_i − ŷ_i|`, take the median residual `m`.
3. Recompute robustness weights using a bisquare function:
   `w_i = (1 − (residual_i / (6m))²)²` (clamped to 0 for `residual_i ≥ 6m`).
4. Refit using `robustness weight × tricube weight`.

`robustnessIters` (default 2) controls how many times this repeat-and-reweight
cycle runs; more iterations make the fit more resistant to outliers at the
cost of more computation. This matches the robust LOESS procedure described
in Cleveland's original 1979 paper ("Robust Locally Weighted Regression and
Smoothing Scatterplots") and is functionally equivalent to the
[Apache Commons Math `LoessInterpolator`](https://commons.apache.org/proper/commons-math/)
implementation this code is structurally close to.

**Output.** `yhat[i]` is the smoothed value at `x[i]` (the *sorted* input
x-values, exposed as `loess.x`) — LOESS produces a value at each input
sample, not at arbitrary query points.

## 3. Nadaraya–Watson kernel smoothing (`KernelSmoothing.swift`, `Kernels.swift`)

`KernelSmoothing` implements the classic
[Nadaraya–Watson kernel regression estimator](https://en.wikipedia.org/wiki/Kernel_regression):
a weighted average of the observed `y` values, where the weight of each
observation decays with its distance from the query point according to a
kernel function.

```
        Σ_i K((x_i − x₀) / h) · y_i
ŷ(x₀) = ---------------------------
        Σ_i K((x_i − x₀) / h)
```

where `h` is the bandwidth and `K` is one of the kernel functions in
`Kernels.swift`.

**Kernel functions** (`KernelFunction` protocol):

| Kernel | Formula (for `|x| ≤ 1` unless noted) | Support |
|---|---|---|
| `Gauss` | `exp(−0.5x²)` | Infinite (effectively truncated by `radius`) |
| `Gaussian` | Same shape, `radius` derived from a target probability mass (`density`) via the inverse normal CDF | Infinite |
| `Boxcar` | `0.5` | `[-1, 1]` |
| `Epanechnikov` | `0.75(1 − x²)` | `[-1, 1]` |
| `Tricube` | `(70/81)(1 − |x|³)³` | `[-1, 1]` |

Each kernel also exposes a `radius`: the maximum distance (in the same units
as the data's x-range) beyond which its contribution is treated as zero, used
purely as a performance cutoff so `smooth(...)` doesn't sum over the whole
dataset for every query point. `radius` is derived from `bandwidth` and the
data's `range` (max − min of x), so it scales with your data.

**Evaluation points.** By default, `KernelSmoothing` evaluates the fit at `N`
points spaced evenly across the observed x-range (`xpoints`); you can instead
pass an explicit set of `points` to evaluate at.

**Bandwidth selection.** `silverman(x:)` implements
[Silverman's rule of thumb](https://en.wikipedia.org/wiki/Kernel_density_estimation#Bandwidth_selection):

```
h = (4σ⁵ / 3n)^(1/5)
```

as a reasonable default bandwidth for Gaussian-like data, where `σ` is the
sample standard deviation and `n` the sample count. (This helper exists on
the type but isn't currently wired up to the initializer — you can call it
yourself to pick a `bandwidth` before constructing a kernel.)

**Sliding window.** `smooth(x:y:points:)` exploits the fact that both the
sorted sample data and the kernel's finite radius let it maintain a moving
`low` index rather than rescanning from the start for every query point —
an O(n + m) sweep rather than O(n·m) for `n` samples and `m` query points
(modulo the per-point inner scan up to the kernel radius).

## 4. Normal quantile function (`Quantile.swift`)

`QuantileFunction.x(probability:)` computes the inverse of the standard
normal cumulative distribution function (the normal quantile / probit
function): given a probability `p`, it returns `x` such that
`P(Z ≤ x) = p` for `Z ~ N(0, 1)`.

It uses [Peter Acklam's rational (minimax) approximation](https://web.archive.org/web/2015/http://home.online.no/~pjacklam/notes/invnorm/),
accurate to within about `1.15 × 10⁻⁹` in relative error, split into three
regions (lower tail, central region, upper tail) each with its own rational
polynomial approximation. This is what `Gaussian.range`'s `willSet` uses to
translate a target probability mass (`density`) into an effective kernel
radius.

## 5. Array statistics (`Array+Extension.swift`)

Simple `FloatingPoint` array helpers used throughout the package:

- `sum()`, `avg()` — straightforward reduction.
- `stdev(unbiased:)` — population standard deviation (`unbiased: false`,
  divides by `n`) or sample standard deviation (`unbiased: true`, divides by
  `n − 1`, [Bessel's correction](https://en.wikipedia.org/wiki/Bessel%27s_correction)).
- `stats(unbiased:)` — single-pass mean, variance, and standard deviation,
  using the identity `Var(X) = E[X²] − E[X]²`. Note this identity is less
  numerically stable than a two-pass (or Welford's online) algorithm for
  data with a large mean relative to its variance; prefer `stdev(unbiased:)`
  and a separate mean calculation if you need higher precision on such data.

## References

- Cleveland, W. S. (1979). "Robust Locally Weighted Regression and Smoothing
  Scatterplots." *Journal of the American Statistical Association*.
- Nadaraya, E. A. (1964). "On Estimating Regression." *Theory of Probability
  and Its Applications*. / Watson, G. S. (1964). "Smooth Regression
  Analysis." *Sankhyā*.
- Silverman, B. W. (1986). *Density Estimation for Statistics and Data
  Analysis*.
- Acklam, P. J. "An algorithm for computing the inverse normal cumulative
  distribution function."
