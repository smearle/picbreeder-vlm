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
 * Represents an identity function. This should only be used
 * in the input layer, but it may be used elsewhere (it is a
 * differentiable function).
 * 
 * @author Nick
 *
 */

public final class Identity implements Function {
	/**
	 * Returns the argument x.
	 * 
	 * @param x The input value
	 * @return x
	 */
	public double valueAt(double x) {
		return x;
	}
}
