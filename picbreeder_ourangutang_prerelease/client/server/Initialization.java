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
 * The Initialization provides a means for the ServerConnection to
 * initialize itself to the applet or application's parameters.
 * 
 * @author Nick
 */

public interface Initialization {
	/**
	 * Retreives a given parameter during initialization.
	 * 
	 * @param parameterName The parameter to retreive
	 * @return The value of the parameter
	 */
	public String getParameter(String parameterName);
}
