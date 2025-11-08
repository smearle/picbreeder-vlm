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

package client.gui;

public interface SessionEnd {
	/**
	 * Indicates that the user quit the current session.
	 */
	public void quit();
	
	/**
	 * Indicates that the user wants to publish the current session.
	 */
	public void publish();
	
	/**
	 * Indicates the the user needs to register on the website.
	 */
	public void register();
	
	public void authenticate(String username, char []password);
}
