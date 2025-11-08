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
 * An invokable version of the cosine function.
 * 
 * @author Nick
 */
public final class Cosine implements Function {
	/**
	 * Computes the value at <code>Math.cos(x)</code>.
	 * 
	 * @param x The input value
	 * @return <code>cos(x)</code>
	 */
	public double valueAt(double x) {
		return Math.cos(x);
	}
}
