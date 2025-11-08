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
 * A SampledBoundedFunction approximates a given function by
 * sampling it at a specific frequency. The look up table is then
 * used to speed up calculation of f(x).
 * <p>
 * This code assumes that the input function "levels" off at
 * some point in the range (-100, 100). For example, sigmoid and
 * gaussian both do this. If your function doesn't do this, use
 * the SampledUnboundedFunction.
 * 
 * @author Nick
 *
 */
public final class SampledBoundedFunction implements Function {
	private final double [] table;
	private final int offset;
	private final double resolution;
	private final double low;
	private final double high;
	
	/**
	 * Creates a sampled version of the given function using the
	 * desired frequency.
	 * 
	 * @param function The function
	 * @param frequency The frequency
	 */
	public SampledBoundedFunction(Function function, double frequency) {
		low = findPoint(function, 0, -100, frequency) - frequency;
		high = findPoint(function, 0, 100, frequency) + frequency;
		
		resolution = 1.0 / frequency;
		// 0 = low + offset * frequency --> offset = -low / frequency --> offset = -low * resolution
		offset = (int)(-low * resolution);
		table = new double [(int)((high - low) * resolution) + 1];
		
		for(int i = 0; i < table.length; i++)
			table[i] = function.valueAt((i - offset) * frequency);
	}
	
	public double valueAt(double x) {
		if(x <= low) return table[0];
		else if(x >= high) return table[table.length - 1];
		else return table[offset + (int)(x * resolution + 0.5)];
	}
	
	/**
	 * Finds the extreme domain point (where the function flattens) using a binary
	 * search.  The code assumes that the user gaurantees the input function makes sense.
	 * 
	 * @param f The function to search
	 * @param center The center of the function (any point on the slope)
	 * @param boundary The extreme of the function (any point on the flattened portion)
	 * @param tolerance The tolerance of the value at the flattening to the real value
	 * @return The x coordinate of where the flattening starts, accurate to within the tolerance
	 */
	private static double findPoint(Function f, double center, double boundary, double tolerance) {
		double m;
		double extreme = f.valueAt(boundary);
		/*
		 * The binary search is controlled by a for loop to avoid double precision.
		 * This should be accurate on a domain 0,100 to the nearest 2^iterations.  Since I
		 * hardcoded 1000, we are looking at at least thousandths.
		 */
		for(int i = 0; i < 1000; i++) {
			m = (center + boundary) / 2.0;
			if(Math.abs(f.valueAt(m) - extreme) < tolerance)
				boundary = m;
			else
				center = m;
		}

		return boundary;
	}
}
