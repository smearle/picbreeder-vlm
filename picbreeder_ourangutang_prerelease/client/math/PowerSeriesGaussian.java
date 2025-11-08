/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.math;

/**
 * Represents a Gaussian based on a minimax approach.
 * <p>
 * Thanks to Mark Colbert for the coefficients.
 * 
 * @author Adam Campbell 
 */
public class PowerSeriesGaussian implements Function {

	/**
	 * Computes the gaussian function on <code>x</code>.
	 * Uses the minimax optimizations on power series
	 * 
	 * @param x The input value
	 * @return approximately <code>exp(-x<sup>2</sup>)</code>
	 */
	public double valueAt(double x) {
		double x2=x*x, x4=x2*x2;
		return (1.9959394200356138 + 0.3518336357159904*x2 +
			    0.3698673724861037*x4)/
			    (1 + 0.6413839518669078*x2 +
			      0.34905871539144967*x4); 
	}
}
