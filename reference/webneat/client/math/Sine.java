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
 * An invokable version of the sine function.
 * 
 * @author Nick
 */

public final class Sine implements Function {
	/**
	 * Computes the value at <code>Math.sin(x)</code>.
	 * 
	 * @param x The input value
	 * @return <code>sin(x)</code>
	 */
	public double valueAt(double x) {
		return Math.sin(x);
	}
}
