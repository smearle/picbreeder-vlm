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
 * Singleton object for ImageDatabase class.
 * <p>
 * All singleton objects are implemented as independant classes to
 * allow objects to implement multiple singletons without the multiple
 * class inheritence issue.
 * 
 * @author Nick
 */

public class ImageDatabaseInstance {
	private static ImageDatabase singleton = null;
	
	/**
	 * Sets the ImageDatabase singleton.
	 * 
	 * @param instance The ImageDatabase singleton
	 */
	public static void set(ImageDatabase instance) {
		singleton = instance;
	}

	/**
	 * Gets the ImageDatabase singleton.
	 * 
	 * @return The ImageDatabase singleton
	 */
	public static ImageDatabase get() {
		return singleton;
	}
}
