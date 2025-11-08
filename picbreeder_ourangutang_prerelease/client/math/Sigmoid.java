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
 * Represents an executable sigmoid function.
 * 
 * @author Nick Beato 
 */
public final class Sigmoid implements Function {
	/**
	 * Computes the sigmoid function on <code>x</code>.
	 * 
	 * @param x The input value
	 * @return <code>1 / (1 + exp(-x))</code>
	 */
	public final double valueAt(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}
}
