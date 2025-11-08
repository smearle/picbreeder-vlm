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

/**
 * Singleton object for DatabaseInstance class.
 * <p>
 * All singleton objects are implemented as independant classes to
 * allow objects to implement multiple singletons without the multiple
 * class inheritence issue.
 * 
 * @author Nick
 */

public class DatabaseInstance {
	private static Database singleton = null;
	
	/**
	 * Initializes the database when a new session starts.
	 */
	public static void beginSession() {
		singleton = new client.evolution.impl.StorageDatabase();
	}
	
	/**
	 * Resets the database when a session has completed.
	 *
	 */
	public static void endSession() {
		singleton = null;
	}
	
	static {
		beginSession();
	}


	/**
	 * Gets the DatabaseInstance singleton.
	 * 
	 * @return The DatabaseInstance singleton
	 */
	public static Database get() {
		return singleton;
	}
}
