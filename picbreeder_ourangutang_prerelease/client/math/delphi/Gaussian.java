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

package client.math.delphi;

import client.math.Function;

/**
 * The DelphiNEAT Gaussian function.
 * 
 * @author Adam Campbell
 */
public class Gaussian implements Function {
	/**
	 * Computes the gaussian function on <code>x</code>.
	 * 
	 * @param x The input value
	 * @return <code>exp(-2.5x<sup>2</sup>)</code>
	 */
	public double valueAt(double x) {
		x *= 2.5;
		return Math.exp(-x*x);
	}
}
