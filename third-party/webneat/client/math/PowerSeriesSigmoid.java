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
 * Represents a sigmoid function based on a minimax approach.
 * <p>
 * Thanks to Mark Colbert for the coefficients.
 * 
 * @author Adam Campbell 
 */

public class PowerSeriesSigmoid implements Function {

	/**
	 * Computes the sigmoid function on <code>x</code>.
	 * Uses the minimax optimizations on power series
	 * 
	 * @param x The input value
	 * @return Approximately <code>1 / (1 + exp(-x))</code>
	 */
	public double valueAt(double x) {
		double x2=x*x, x3=x2*x, x4=x2*x2;
		return (0.49994708426510076 + 0.2302973194230301*x +
			    0.043393195796798435*x2 +
			    0.003938714087413967*x3 +
			    0.0001438866363740516*x4)/
			  (1 - 0.039735448171325446*x +
			    0.10631646361027865*x2 -
			    0.003501462610127195*x3 +
			    0.0005097284928370708*x4) ; 
	}
}
