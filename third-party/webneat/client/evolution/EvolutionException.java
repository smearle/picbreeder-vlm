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
 * Indicates that an exception has occured during evolution.
 * 
 * @author Nick
 */
public class EvolutionException extends Exception {
	/**
	 * Creates an EvolutionException, indicating the
	 * reason the exception occured.
	 * 
	 * @param reason The reason this exception occured
	 */
	public EvolutionException(String reason) {
		super(reason);
	}
}
