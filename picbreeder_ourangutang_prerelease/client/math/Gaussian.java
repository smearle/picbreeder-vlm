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
 * Represents an executable Gaussian function (normal distribution).
 * 
 * @author Nick Beato 
 */
public class Gaussian implements Function {
	/**
	 * Computes the gaussian function on <code>x</code>.
	 * 
	 * @param x The input value
	 * @return <code>exp(-x<sup>2</sup>)</code>
	 */
	public double valueAt(double x) {
		return Math.exp(-x * x);
	}
}
