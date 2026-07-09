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

package client.evolution;

import client.Singleton;

/**
 * Singleton object for GeneticFactory class.
 * <p>
 * All singleton objects are implemented as independant classes to
 * allow objects to implement multiple singletons without the multiple
 * class inheritence issue.
 * 
 * @author Nick
 */

public class GeneticFactoryInstance extends Singleton {
	private static GeneticFactory singleton = null;
	
	static {
		new GeneticFactoryInstance();
	}
	
	public void beginSession() {
		// must reset the factory to clear the history
		// and different generator parameters
		singleton = new client.evolution.impl.DefaultGeneticFactory();
	}
	
	public void endSession() {
		singleton = null;
	}
	
	/**
	 * Sets the GeneticFactory singleton.
	 * 
	 * @param instance The GeneticFactory singleton
	 */
	public static void set(GeneticFactory instance) {
		singleton = instance;
	}
	
	/**
	 * Gets the GeneticFactory singleton.
	 * 
	 * @return The singleton
	 */
	public static GeneticFactory get() {
		return singleton;
	}
}
