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
 * This is a utility class to convert a unipolar function
 * to a bipolar. It does not gaurantee that the input
 * function is unipolar, that is the programmer's responsibility.
 * 
 * @author Nick
 */

public final class UnipolarToBipolar implements Function {
	private final Function function;
	
	/**
	 * Constructs a bipolar functions who's output is dependant
	 * on the given unipolar function.
	 * 
	 * @param function The unipolar function
	 */
	public UnipolarToBipolar(Function function) {
		this.function = function;
	}
	
	public double valueAt(double x) {
		return function.valueAt(x) * 2.0 - 1.0;
	}
}
