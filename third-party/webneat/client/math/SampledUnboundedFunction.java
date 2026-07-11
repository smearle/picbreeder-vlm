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

import java.util.*;

/**
 * A SampledUnboundedFunction is a look up table based version of the
 * given input function. Since the input function does not "level off"
 * (if it does, used the bounded function), the inputs are rounded
 * and the results are hashed.
 * 
 * @author Nick
 */

public final class SampledUnboundedFunction implements Function {
	private Map <Double, Double> samples;
	private final Function function;
	private final double resolution;
	
	/**
	 * Creates a sampled version of the given function, using the given
	 * frequency as an indication of the desired tolerance.
	 * 
	 * @param function The function
	 * @param frequency The frequency
	 */
	public SampledUnboundedFunction(Function function, double frequency) {		
		this.function = function;
		resolution = 1.0 / frequency;
		samples = new HashMap <Double, Double> ();
	}

	public double valueAt(double x) {
		final double scaled = x * resolution;
		final double truncated = Math.floor(scaled);
		return interpolate(sampleAt(truncated), sampleAt(truncated + 1.0), (scaled - truncated) / resolution);
	}

	private double sampleAt(double x) {
		if(!samples.containsKey(x))
			samples.put(x, function.valueAt(x / resolution));
		
		return samples.get(x);
	}
	
	private double interpolate(double x, double y, double t) {
		return x * (1.0 - t) + y * t;
	}
}
