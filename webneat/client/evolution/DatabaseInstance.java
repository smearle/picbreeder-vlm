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
 * Singleton object for DatabaseInstance class.
 * <p>
 * All singleton objects are implemented as independant classes to
 * allow objects to implement multiple singletons without the multiple
 * class inheritence issue.
 * 
 * @author Nick
 */

public class DatabaseInstance extends Singleton {
	private static Database singleton = null;
	
	static {
		new DatabaseInstance();
	}
	
	public void beginSession() {
		singleton = new client.evolution.impl.StorageDatabase();
	}
	
	public void endSession() {
		singleton = null;
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
