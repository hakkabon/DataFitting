# DataFitting

A small Swift package for fitting curves to scattered `(x, y)` data. It provides
three independent fitting strategies plus a couple of statistics helpers they
share:

| Type | What it does | Global or local? | Output |
|---|---|---|---|
| `Regression` | Ordinary least-squares polynomial regression | Global (one formula for the whole domain) | Closed-form coefficients, evaluate anywhere |
| `Loess` | Local regression (Cleveland's LOESS) | Local, robust to outliers | A smoothed value at each input `x` |
| `KernelSmoothing` | Nadaraya–Watson kernel regression | Local, weighted average | A smoothed value at each requested `x` |

All three consume `CGPoint` samples. See [METHODOLOGY.md](METHODOLOGY.md) for
the statistical background and the exact formulas each type implements.

## Installation

Add the package to `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/hakkabon/DataFitting.git", from: "1.0.1"),
]
```

`DataFitting` depends on [`Dimensional`](https://github.com/hakkabon/Dimensional)
for matrix/vector algebra, used by `Regression` to solve the normal equations.

## Quick start

### Polynomial regression

```swift
import DataFitting

let points = [
    CGPoint(x: 0, y: 1.1),
    CGPoint(x: 1, y: 2.9),
    CGPoint(x: 2, y: 5.2),
    CGPoint(x: 3, y: 6.8),
]

let fit = Regression(points: points, degree: 1)   // straight line
let y = fit.predict(point: 4.0)                   // extrapolate to x = 4
```

Increase `degree` to fit quadratics, cubics, etc. You need at least
`degree + 1` sample points.

### LOESS local regression

```swift
let loess = Loess(data: points, bandwidth: 0.3, iterations: 2)
// loess.x    -- sorted input x-values
// loess.yhat -- smoothed y-value at each loess.x[i]
```

### Nadaraya–Watson kernel smoothing

```swift
let smoother = KernelSmoothing(data: points, kernel: Gauss(bandwidth: 0.15), N: 100)
// smoother.xpoints -- 100 evenly spaced evaluation points across the x-range
// smoother.yhat    -- smoothed y-value at each xpoints[i]
```

Available kernels: `Gauss`, `Gaussian` (density-parameterized), `Boxcar`,
`Epanechnikov`, `Tricube`.

### Array statistics

```swift
let values: [Double] = [2, 4, 4, 4, 5, 5, 7, 9]
values.avg()                    // arithmetic mean
values.stdev()                  // population standard deviation
values.stdev(unbiased: true)    // sample standard deviation (n-1)
values.stats(unbiased: true)    // (mean, variance, stdev) in one pass
```

## Choosing between the three fitters

- Use **`Regression`** when you believe the underlying relationship really is
  a low-degree polynomial and want a compact, extrapolatable formula.
- Use **`Loess`** when the relationship is unknown/non-polynomial, your data
  has outliers, and you mainly want a smoothed curve over the observed range
  (LOESS is not meant for extrapolation).
- Use **`KernelSmoothing`** when you want a simple, well-understood weighted
  local average and are prepared to choose or tune a kernel bandwidth
  yourself (no robustness iterations, unlike LOESS).

## Known limitations

- All three types take `CGPoint` (so `Double`-only data must be converted).
- `Regression`, `Loess`, and `KernelSmoothing` all sort input data internally
  and assume the `x` values are the independent variable; duplicate `x`
  values are allowed but not deduplicated.
- `KernelSmoothing` and `Loess` are single-threaded and evaluate an O(n) (or
  O(n·window)) scan per output point — fine for the interactive/plotting use
  case this package targets, not tuned for very large datasets.
- Neither `Loess` nor `KernelSmoothing` extrapolates meaningfully outside the
  observed `x`-range; `Regression` will extrapolate, but polynomial
  extrapolation is unreliable past the sample range, especially for
  higher-degree fits.

See [METHODOLOGY.md](METHODOLOGY.md) for the math and references.
