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

package client;

/**
 * Singleton object for ParameterTable class.
 * <p>
 * All singleton objects are implemented as independant classes to
 * allow objects to implement multiple singletons without the multiple
 * class inheritence issue.
 * 
 * @author Nick
 *
 */
public class ParameterTableInstance {
	private static ParameterTable singleton = null;
	
	/**
	 * Sets the ParameterTable singleton.
	 * 
	 * @param instance The ParameterTable singleton
	 */
	public static void set(ParameterTable instance) {
		singleton = instance;
	}

	/**
	 * Gets the ParameterTable singleton.
	 * 
	 * @return The ParameterTable singleton
	 */
	public static ParameterTable get() {
		return singleton;
	}
}
