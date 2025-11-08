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

package client.cppn;

/**
 * Singleton object for CPPNFactory class.
 * <p>
 * All singleton objects are implemented as independant classes to
 * allow objects to implement multiple singletons without the multiple
 * class inheritence issue.
 * 
 * @author Nick Beato
 */

public class CPPNFactoryInstance {
	private static CPPNFactory singleton = new client.cppn.impl.DefaultCPPNFactory();

	/**
	 * Sets the CPPNFactory singleton.
	 * 
	 * @param instance The CPPNFactory singleton
	 */
	public static void set(CPPNFactory instance) {
		singleton = instance;
	}

	/**
	 * Gets the CPPNFactory singleton.
	 * 
	 * @return The CPPNFactory singleton
	 */
	public static CPPNFactory get() {
		return singleton;
	}
}
