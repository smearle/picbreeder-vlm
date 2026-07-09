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

package client.utilities;

/**
 * An extended Random class that provides a nextBoolean method
 * that takes probability into account.
 * 
 * @author Nick
 */

public final class Random extends java.util.Random {
	private final static Random singleton = new Random();

	public static Random instance() {        
		return singleton;
    }
	
	private Random() {
		super(System.currentTimeMillis());
	}
	
	/**
	 * Returns a random boolean with the given probability of being
	 * <code>true</code>.
	 * 
	 * @param probability The probability of a <code>true</code> occurance.
	 * @return <code>true</code> with the specified probability, <code>false</code> otherwise.
	 */
	public boolean nextBoolean(double probability) {
		return nextDouble() < probability;
	}
}
