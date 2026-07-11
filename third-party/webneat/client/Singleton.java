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

import java.util.Set;

/**
 * Singleton object that acts as a base for all singletons in the system.
 * <p>
 * All singleton objects are implemented as independant classes to
 * allow objects to implement multiple singletons without the multiple
 * class inheritence issue.
 * 
 * @author Nick
 */

public abstract class Singleton {
	private static final Set <Singleton> knownSingletons = new java.util.HashSet <Singleton> ();
	
	/**
	 * Allows objects to iterate over all singletons in the system. This is
	 * used primarily by the client.gui.SessionHandler to initialize the
	 * browser sessions.
	 * 
	 * @return An iterable list of singletons
	 */
	public static Iterable <Singleton> getSingletons() {
		return knownSingletons;
	}
	
	/**
	 * Any singleton must invoke this constructor
	 */
	protected Singleton() {
		if(!this.getClass().getName().endsWith("Instance"))
			throw new RuntimeException("Singleton classes must end with Instance!");
		
		knownSingletons.add(this);
		beginSession();
	}
	
	/**
	 * Tells the single that it should initialize for a new session.
	 * The default behavior is to do nothing.
	 */
	public void beginSession() {
		
	}
	
	/**
	 * Tells the singleton that it should uninitialize itself, indicating the applet
	 * is closing, even if it is still in memory. The default behavior is
	 * to do nothing.
	 */
	public void endSession() {
		
	}
}
