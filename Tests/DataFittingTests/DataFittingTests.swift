import Testing
import CoreGraphics
@testable import DataFitting

@Suite("Array statistics")
struct ArrayStatisticsTests {

    @Test("sum and average of a simple series")
    func sumAndAverage() {
        let values: [Double] = [1, 2, 3, 4, 5]
        #expect(values.sum() == 15)
        #expect(values.avg() == 3)
    }

    @Test("population vs. sample standard deviation")
    func stdevPopulationVsSample() {
        // Known example: [2, 4, 4, 4, 5, 5, 7, 9]
        // population variance = 4, population stdev = 2
        // sample stdev (n-1) ≈ 2.13809
        let values: [Double] = [2, 4, 4, 4, 5, 5, 7, 9]
        #expect(abs(values.stdev(unbiased: false) - 2.0) < 1e-9)
        #expect(abs(values.stdev(unbiased: true) - 2.13809314646499) < 1e-9)
    }
}

@Suite("Polynomial regression")
struct RegressionTests {

    @Test("recovers exact coefficients for a noiseless line y = 2x + 1")
    func linearFitIsExact() {
        let points = (0...10).map { CGPoint(x: CGFloat($0), y: CGFloat(2 * $0 + 1)) }
        let regression = Regression(points: points, degree: 1)
        for x in stride(from: 0.0, through: 10.0, by: 1.0) {
            let predicted = regression.predict(point: x)
            #expect(abs(predicted - (2 * x + 1)) < 1e-6)
        }
    }

    @Test("recovers exact coefficients for a noiseless parabola y = x^2 - 3x + 2")
    func quadraticFitIsExact() {
        let points = (-5...5).map { x -> CGPoint in
            let d = Double(x)
            return CGPoint(x: CGFloat(d), y: CGFloat(d * d - 3 * d + 2))
        }
        let regression = Regression(points: points, degree: 2)
        for x in stride(from: -5.0, through: 5.0, by: 1.0) {
            let predicted = regression.predict(point: x)
            let expected = x * x - 3 * x + 2
            #expect(abs(predicted - expected) < 1e-6)
        }
    }
}

@Suite("Kernel functions")
struct KernelTests {

    @Test("Gauss kernel radius depends only on current bandwidth/range, not history")
    func gaussRadiusIsIdempotent() {
        var kernel = Gauss(bandwidth: 0.5)
        kernel.range = 10
        let radiusAfterFirstSet = kernel.radius
        // Re-assigning the same bandwidth must reproduce the same radius,
        // not compound on the previous radius.
        kernel.bandwidth = 0.5
        #expect(abs(kernel.radius - radiusAfterFirstSet) < 1e-9)
    }

    @Test("Epanechnikov and Tricube kernels are publicly constructible")
    func nonDefaultKernelsAreConstructible() {
        let epanechnikov = Epanechnikov(bandwidth: 0.4, range: 1, radius: 1)
        let tricube = Tricube(bandwidth: 0.4, range: 1, radius: 1)
        #expect(epanechnikov.eval(0) == 0.75)
        #expect(abs(tricube.eval(0) - 70.0 / 81.0) < 1e-9)
    }

    @Test("kernels evaluate to zero outside the unit interval")
    func compactSupportOutsideUnitInterval() {
        let epanechnikov = Epanechnikov(bandwidth: 0.5, range: 1, radius: 1)
        let tricube = Tricube(bandwidth: 0.5, range: 1, radius: 1)
        let boxcar = Boxcar(bandwidth: 0.5)
        #expect(epanechnikov.eval(1.5) == 0)
        #expect(tricube.eval(1.5) == 0)
        #expect(boxcar.eval(1.5) == 0)
    }
}

@Suite("Nadaraya-Watson kernel smoothing")
struct KernelSmoothingTests {

    @Test("smoothing a noiseless line reproduces the line away from the boundary")
    func smoothsLinearData() {
        // y = 3x, sampled densely so boundary effects at the ends don't dominate.
        let points = (0...40).map { CGPoint(x: CGFloat($0), y: CGFloat(3 * $0)) }
        let smoother = KernelSmoothing(data: points, kernel: Gauss(bandwidth: 0.15), N: 41)
        // Check interior points only; kernel smoothers are biased near boundaries.
        for i in 10..<30 {
            let x = smoother.xpoints[i]
            let expected = 3 * x
            #expect(abs(smoother.yhat[i] - expected) < 1.0)
        }
    }
}

@Suite("Loess local regression")
struct LoessTests {

    @Test("smooths noiseless linear data back to the line")
    func smoothsLinearData() {
        let points = (0...30).map { CGPoint(x: CGFloat($0), y: CGFloat(2 * $0 + 5)) }
        let loess = Loess(data: points, bandwidth: 0.3, iterations: 2)
        for i in 5..<25 {
            let expected = 2 * loess.x[i] + 5
            #expect(abs(loess.yhat[i] - expected) < 1.0)
        }
    }
}
