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

package client.renderers;

import client.Singleton;

/**
 * Singleton object for RenderingAlgorithm class.
 * <p>
 * All singleton objects are implemented as independant classes to
 * allow objects to implement multiple singletons without the multiple
 * class inheritence issue.
 * 
 * @author Nick
 */

public class RenderingAlgorithmInstance extends Singleton {
	private static RenderingAlgorithm singleton = new client.renderers.algorithms.Background();
	
	static {
		new RenderingAlgorithmInstance();
	}
	
	/**
	 * Gets the RenderingAlgorithm singleton.
	 * 
	 * @return The singleton
	 */
	public static RenderingAlgorithm get() {
		return singleton;
	}
	
	/**
	 * Sets the RenderingAlgorithm singleton.
	 * 
	 * @param instance The singleton
	 */
	public static void set(RenderingAlgorithm instance) {
		singleton = instance;
	}
}
