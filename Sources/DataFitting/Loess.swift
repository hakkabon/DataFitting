//
//  Loess.swift
//  Data Fitting
//
//  Created by Ulf Akerstedt-Inoue on 2019/05/16.
//  Copyright © 2019 hakkabon software. All rights reserved.
//

import Foundation
import CoreGraphics

/// Compute a least-squares linear fit weighted by the product of robustness weights
/// and the tricube weight function.
/// See http://en.wikipedia.org/wiki/Linear_regression
/// (section "Univariate linear case")
/// and http://en.wikipedia.org/wiki/Weighted_least_squares
/// (section "Weighted least squares")
public class Loess {

    typealias Interval = (left: Int, right: Int)
    
    var bandwidth: Double = 0.3
    var robustnessIters: Int = 2
    var accuracy: Double = 1e-8
    public var x: [Double] = [Double]()
    public var y: [Double] = [Double]()
    public var yhat: [Double] = [Double]()

    public init() {
    }

    public init(data: [CGPoint], bandwidth: Double = 0.3, iterations: Int = 2, accuracy: Double = 1e-12) {
        self.bandwidth = bandwidth
        self.robustnessIters = iterations
        self .accuracy = accuracy

        // Sort (x,y) sample data ascendingly
        let sorted = data.enumerated().sorted(by: { $0.element.x < $1.element.x })
        x = sorted.map( { Double($0.element.x) } )
        y = sorted.map( { Double($0.element.y) } )
        
        // Reserve space for n weights.
        let n = data.count
        let weights = Array(repeating: 1.0, count: n)

        // We need at least three sample points for smoothing.
        yhat = x.count > 2 ? smooth(x: x, y: y, weights: weights) : y
    }
    
    // Computes the loess type smoothing (Cleveland) at given (x,y) points.
    func smooth(x xval: [Double], y yval: [Double], weights: [Double]) -> [Double] {
        let n = xval.count

        precondition(Int(round(bandwidth * Double(n))) > 2, "Too few number of points for given bandwidth.")

        let bandwidthInPoints = Int(round(bandwidth * Double(n)))
        var yhat = Array(repeating: 0.0, count: n)
        var residuals = Array(repeating: 0.0, count: n)
        var robustnessWeights = Array(repeating: 1.0, count: n)
        
        // Do an initial fit and 'robustnessIters' robustness iterations.
        // This is equivalent to doing 'robustnessIters+1' robustness iterations
        // starting with all robustness weights set to 1.
        for iter in 0 ... robustnessIters {
            var bandwidthInterval = (0, bandwidthInPoints - 1)
            
            // At each x, compute a local weighted linear regression
            for (i, x0) in xval.enumerated() {
                
                // Find out the interval of source points on which a regression is to be made.
                if i > 0 {
                    bandwidthInterval = update(interval: bandwidthInterval, x: xval, weights: weights, index: i)
                }

                let ileft = bandwidthInterval.0
                let iright = bandwidthInterval.1
                
                // Compute the point of the bandwidth interval that is farthest from x.
                let edge = xval[i] - xval[ileft] > xval[iright] - xval[i] ? ileft : iright
                
                // Compute a least-squares linear fit weighted by the product of robustness weights
                // and the tricube weight function.
                var sumWeights: Double = 0
                var sumX: Double = 0
                var sumXSquared: Double = 0
                var sumY: Double = 0
                var sumXY: Double = 0
                let denom = abs(1.0 / (xval[edge] - x0))
                for k in ileft ... iright {
                    let xk = xval[k]
                    let yk = yval[k]
                    let dist = k < i ? x0 - xk : xk - x0
                    let w = tricube(x: dist * denom) * robustnessWeights[k] * weights[k]
                    let xkw = xk * w;
                    sumWeights += w
                    sumX += xkw
                    sumXSquared += xk * xkw
                    sumY += yk * w
                    sumXY += yk * xkw
                }
                
                let meanX = sumX / sumWeights
                let meanY = sumY / sumWeights
                let meanXY = sumXY / sumWeights
                let meanXSquared = sumXSquared / sumWeights
                
                let beta = sqrt(abs(meanXSquared - meanX * meanX)) < accuracy ? 0 :
                    (meanXY - meanX * meanY) / (meanXSquared - meanX * meanX)
                
                let alpha = meanY - beta * meanX
                yhat[i] = beta * x0 + alpha
                residuals[i] = abs(yval[i] - yhat[i])
            }
            
            // No need to recompute the robustness weights at the last
            // iteration, they won't be needed anymore
            guard iter < robustnessIters else { break }
            
            // Recompute the robustness weights.
            
            // Find the median residual.
            // An arraycopy and a sort are completely tractable here,
            // because the preceding loop is a lot more expensive
            let sortedResiduals = residuals.sorted(by: { $0 < $1 })
            let medianResidual = sortedResiduals[n / 2]
            if abs(medianResidual) < accuracy { break }
            
            for i in 0 ..< n {
                let arg = residuals[i] / (6 * medianResidual)
                if arg >= 1 {
                    robustnessWeights[i] = 0
                } else {
                    let w = 1 - arg * arg
                    robustnessWeights[i] = w * w
                }
            }
        }
        return yhat
    }
    
    func update(interval: Interval, x: [Double], weights: [Double], index i: Int) -> Interval {
    
        func nextNonzero(index: Int, w: [Double]) -> Int {
            for i in index+1 ..< w.count {
                if w[i] > 0 {
                    return i
                }
            }
            return w.count - 1
        }

        // The right edge should be adjusted if the next point to the right is closer to
        // xval[i] than the leftmost point of the current interval
        let nextRight = nextNonzero(index: interval.right, w: weights)
        if nextRight < x.count && x[nextRight] - x[i] < x[i] - x[interval.left] {
            let nextLeft = nextNonzero(index: interval.left, w: weights)
            return (nextLeft, nextRight)
        }
        return interval
    }
    
    func tricube(x: Double) -> Double {
        let absX = abs(x)
        if absX >= 1 {
            return 0
        }
        let tmp = 1.0 - absX * absX * absX
        return tmp * tmp * tmp
    }
}
