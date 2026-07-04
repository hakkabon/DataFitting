//
//  Array+Extension.swift
//  Data Fitting
//
//  Created by Ulf Akerstedt-Inoue on 2019/05/07.
//  Copyright © 2019 hakkabon software. All rights reserved.
//

import Foundation

extension Array where Element: FloatingPoint {
    
    /// Returns the sum of all elements in the array
    public func sum() -> Element {
        return self.reduce(0, +)
    }
    
    /// Returns the average of all elements in the array
    public func avg() -> Element {
        return self.isEmpty ? 0 : self.sum() / Element(self.count)
    }
    
    /// Returns the variance of the Array (entire population) or unbiased sample
    /// population.
    /// Differentiate between population variance and sample variance.
    public func stdev(unbiased: Bool = false) -> Element {
        precondition(!unbiased || self.count > 1, "Unbiased sample stdev requires at least 2 elements.")
        let mean = self.avg()
        let variance = self.reduce(0, { $0 + ($1-mean)*($1-mean) })
        let N = unbiased ? Element(self.count) - 1 : Element(self.count)
        return sqrt( variance / N )
    }

    /// Returns the mean, variance and std deviation (entire population) or unbiased
    /// sample population.
    /// Differentiate between population variance and sample variance.
    public func stats(unbiased: Bool = false) -> (Element,Element,Element) {
        precondition(!unbiased || self.count > 1, "Unbiased sample stats require at least 2 elements.")
        let factor = unbiased ? Element(self.count) / Element(self.count-1) : 1
        typealias Accumulator = (Element,Element)
        let accumulator: Accumulator = (0,0)
        var s = self.reduce(into: accumulator) {
            $0.0 += $1
            $0.1 += $1 * $1
        }
        s.0 /= Element(self.count)
        s.1 /= Element(self.count)
        s.1 -= s.0 * s.0
        s.1 = unbiased ? s.1 * factor : s.1
        return (s.0, s.1, sqrt(s.1 > 0 ? s.1 : 0))
    }
}
