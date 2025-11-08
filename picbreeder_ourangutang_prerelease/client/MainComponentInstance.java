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

import java.awt.Component;

/**
 * Instance class for the main awt component. This can be used
 * for things like message boxes, errors, etc.  It may also be
 * used to get the applet.
 * 
 * @author Nick
 */

public class MainComponentInstance {
	private static Component singleton = null;
	
	public static void set(Component instance) {
		singleton = instance;
	}
	
	public static Component get() {
		return singleton;
	}
}
