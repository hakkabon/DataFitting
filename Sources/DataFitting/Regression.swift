//
//  Regression.swift
//  Data Fitting
//
//  Created by Ulf Akerstedt-Inoue on 2019/05/05.
//  Copyright © 2019 hakkabon software. All rights reserved.
//

import CoreGraphics
import Dimensional

public class Regression {

    var ß: [Double] = [Double]()
    var degree: Int = 0

    init() {
    }
    
    public init(points: [CGPoint], degree: Int = 1) {
        precondition(degree > 0, "Degree should be higher than zero")
        precondition(degree < points.count, "Need more sample points, got \(points.count) sample points.")

        self.degree = degree
        
        //      |1  x01 x02 ... x0n|      |y0|
        //      |1  x11 x12 ... x1n|      |y1|
        //  X = |1  x21 x22 ... x2n|  Y = |y2|  ß = (X'X)^-1 * X'Y'
        //      |1  x31 x32 ... x3n|      |y3|  ß vector m elements
        //      |1  x41 x42 ... x4n|      |y4|
        //      |1  xm1 xm2 ... xmn|      |ym|
        
        // dimension is rather awkward (cols, rows)
        var X: Matrix<Double> = Matrix(repeating: 0, dimensions: (degree+1, points.count))
        X.columns[0] = Array(repeating: 1.0, count: points.count)
        X.columns[1] = points.map { Double($0.x) }
        if degree > 1 {
            for j in 2 ... degree {
                let x = points.map { pow( Double($0.x), Double(j) ) }
                X.columns[j] = x
            }
        }
        let Y = points.map( { Double($0.y) } )

        // calculate regression coefficients ß = (X'X)^-1 X'Y'
        ß = (X.transposed * X).inverse * X.transposed * Y
    }

    public func predict(point: Double) -> Double {
        var x = Array(repeating: Double(point), count: ß.count)
        x[0] = 1

        // Check for higher degree polynomials.
        if degree > 1 {
            for i in 2 ... degree {
                x[i] = pow( x[i], Double(i) )
            }
        }
        let prediction = x.dot(ß)
        return prediction
    }
}
