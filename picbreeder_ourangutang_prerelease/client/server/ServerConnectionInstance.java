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

package client.server;

/**
 * Singleton object for ServerConnection class.
 * <p>
 * All singleton objects are implemented as independant classes to
 * allow objects to implement multiple singletons without the multiple
 * class inheritence issue.
 * 
 * @author Nick
 *
 */
public class ServerConnectionInstance {
	private static ServerConnection singleton = null;
	
	/**
	 * Sets the ServerConnection singleton.
	 * 
	 * @param instance The ServerConnection singleton
	 */
	public static void set(ServerConnection instance) {
		singleton = instance;
	}

	/**
	 * Gets the ServerConnection singleton.
	 * 
	 * @return The ServerConnection singleton
	 */
	public static ServerConnection get() {
		return singleton;
	}
}
