//
//  Kernels.swift
//  Data Fitting
//
//  Created by Ulf Akerstedt-Inoue on 2019/05/14.
//  Copyright © 2019 hakkabon software. All rights reserved.
//

import Foundation

public protocol KernelFunction {
    var bandwidth: Double { get set }   // in % of x-range
    var radius: Double { get }          // max distance about x[0]: | x[0] - x[i] | < 2 * radius
    var range: Double { get set }       // max(x) - min(x) / N-1
    func eval(_ x: Double) -> Double
}

public struct Gaussian: KernelFunction {
    let sqrtpi: Double = 0.3989422804014326779399460599343818684758586311649346
    public var bandwidth: Double = 0.5
    public var radius: Double = 1
    var sigma: Double = 0.5
    var density: Double = 0.5
    public var range: Double = 1 {
        willSet(newValue) {
            // Calculate Radius by using density as 2-tailed probability in Inverse Normal CDF
            let tail = (1.0 - density) / 2.0
            // bandwidth is in units of half inter-quartile range.
            let quantile = QuantileFunction().x(probability: tail) * newValue
            self.radius = max(quantile, bandwidth * 0.5)
        }
    }
    public init(density: Double = 0.5) {
        self.density = density
    }
    public func eval(_ x: Double) -> Double {
        return sqrtpi * (1.0 / sigma) * exp(-0.5 * (pow(x, 2.0) / pow(sigma, 2.0)))
    }
}
public struct Gauss: KernelFunction {
    public var bandwidth: Double = 0.5 {
        willSet(newValue) {
            self.radius = range * 4 * newValue * 0.3706506
        }
    }
    public var range: Double = 1 {
        willSet(newValue) {
            self.radius = newValue * 4 * bandwidth * 0.3706506
        }
    }
    public var radius: Double = 1
    public init(bandwidth: Double = 0.5) {
        self.bandwidth = bandwidth
    }
    public func eval(_ x: Double) -> Double { return exp(-0.5 * x * x) }
}
public struct Boxcar: KernelFunction {
    public var bandwidth: Double = 0.5 {
        willSet(newValue) {
            self.radius = range * newValue * 0.5
        }
    }
    public var range: Double = 1 {
        willSet(newValue) {
            self.radius = newValue * bandwidth * 0.5
        }
    }
    public var radius: Double = 1
    public init(bandwidth: Double = 0.5) {
        self.bandwidth = bandwidth
    }
    public func eval(_ x: Double) -> Double { return abs(x) <= 1 ? 0.5 : 0 }
}
public struct Epanechnikov: KernelFunction {
    public var bandwidth: Double
    public var range: Double
    public var radius: Double
    public init(bandwidth: Double = 0.5, range: Double = 1, radius: Double = 1) {
        self.bandwidth = bandwidth
        self.range = range
        self.radius = radius
    }
    public func eval(_ x: Double) -> Double { return abs(x) <= 1 ? 0.75 * ( 1 - x*x ) : 0 }
}
public struct Tricube: KernelFunction {
    public var bandwidth: Double
    public var range: Double
    public var radius: Double
    public init(bandwidth: Double = 0.5, range: Double = 1, radius: Double = 1) {
        self.bandwidth = bandwidth
        self.range = range
        self.radius = radius
    }
    public func eval(_ x: Double) -> Double {
        return abs(x) <= 1 ? ( 70/81 ) * tricubed( x ) : 0
    }
    func tricubed(_ x: Double) -> Double {
        let x_abs = 1 - pow(abs(x), 3)
        return pow(x_abs, 3)
    }
}
