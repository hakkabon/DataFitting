//
//  GaussKernel.swift
//  Data Fitting
//
//  Created by Ulf Akerstedt-Inoue on 2019/05/05.
//  Copyright © 2019 hakkabon software. All rights reserved.
//

import Foundation
import CoreGraphics

public class KernelSmoothing {
    
    var kernel: KernelFunction = Gauss()
    public var xpoints: [Double] = [Double]()
    public var yhat: [Double] = [Double]()

    init() {
    }

    /// Allocates a new Nadaraya-Watson kernel smoother using a kernel of choice.
    ///
    /// - Parameters:
    ///     - data: The input (x,y) values.
    ///     - kernel: The kernel to be used for smoothing.
    ///     - N: The number of points at which to evaluate the fit.
    ///     - points: The points at which to evaluate the smoothed fit. If missing,
    ///               N points are chosen uniformly to cover range of x.
    public init(data: [CGPoint], kernel: KernelFunction, N: Int = 100, points: [Double] = [Double]()) {
        precondition(data.count > 2, "Need more sample points, got \(data.count) sample points.")

        self.kernel = kernel

        // Sort (x,y) sample data ascendingly
        let sorted = data.enumerated().sorted(by: { $0.element.x < $1.element.x })
        let x = sorted.map( { Double($0.element.x) } )
        let y = sorted.map( { Double($0.element.y) } )

        // Calulate range of predictors.
        let range = abs(x[x.count-1] - x[0])

        // Calculate actual radius of kernel.
        self.kernel.range = range

        // Setup points at which to evaluate the smoothed fit.
        let N = points.count == 0 ? max(N, data.count) : N 
        let delta = range / Double(N - 1)
        let ekvidistant = points.count == 0
        xpoints = Array(repeating: 0.0, count: N)
        for i in 0 ..< N {
            if ekvidistant {
                xpoints[i] = x[0] + Double(i) * delta
            } else {
                xpoints[i] = points[i]
            }
        }
        
        // Reserve space for predicted response.
        yhat = Array(repeating: 0.0, count: N)

        // Perform the actual kernel smoothing.
        smooth(x: x, y: y, points: xpoints)
    }

    // Calculate optimal bandwidth according to Silverman's rule of thumb for
    // kernel density estimation under Gaussian data
    func silverman(x: [Double]) -> Double {
        let num = 4 * pow(x.stdev(), 5)
        let ratio = num / (3.0 * Double(x.count))
        return pow(ratio, 0.2)
    }

    // Computes the Nadraya-Watson type of smoothing at given 'points'.
    func smooth(x: [Double], y: [Double], points: [Double]) {
        guard x.count > 0 else { return }

        var low = 0
        while low < x.count && x[low] < points[0] - self.kernel.radius { low += 1 }

        for (j, x0) in points.enumerated() {
            var numerator: Double = 0.0
            var denominator: Double = 0.0
            for i in low ..< x.count {
                if x[i] < x0 - self.kernel.radius { low = i }
                if x[i] > x0 + self.kernel.radius { break }
                let weight = kernel.eval( abs( x[i] - x0 ) / kernel.bandwidth )
                numerator += weight * y[i]
                denominator += weight
            }
            yhat[j] = denominator > 0 ? numerator / denominator : 0
        }
    }
}
