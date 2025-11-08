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
 * A FatalException indicates an unrecoverable server error. This
 * should report an error and terminate the program.
 * 
 * @author Nick
 */

public final class FatalException extends ServerException {
	public FatalException() {
		super();
	}
	
	public FatalException(String reason) {
		super(reason);
	}
}
