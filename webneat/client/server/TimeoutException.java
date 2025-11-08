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
 * A TimeoutException occurs if the server does not respond to the client.
 * If possible, this exception should be handled gracefully, with the option
 * of trying again later.
 * 
 * @author Nick
 */

public final class TimeoutException extends ServerException {
	public TimeoutException() {
		super("A connection to the server could not be established. Please try again later.");
	}
}
