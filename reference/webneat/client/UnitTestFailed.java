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
 * Base exception class for unit testing. Any class that
 * wants to have a unit test run should declare a method
 * <code>public static void unitTest() throws UnitTestFailed</code>.
 * Using reflection, we can run a unit test on all objects to
 * verify things are behaving.
 *  
 * @author Nick
 *
 */

public final class UnitTestFailed extends Exception {
	/**
	 * Creates an exception for the type of the given
	 * object.
	 * 
	 * @param type An object of the offending type
	 */
	public UnitTestFailed(Object type) {
		this(type.getClass());
	}
	
	/**
	 * Creates an exception for the specified type.
	 * 
	 * @param type The class that caused the failure
	 */
	public UnitTestFailed(Class type) {
		super(type.getName());
	}
}
