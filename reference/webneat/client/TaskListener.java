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
 * A task listener lets threads communicate.  It provides two methods.
 * See the javadoc below for information on them.
 * 
 * @author Nick
 *
 */
public interface TaskListener {
	/**
	 * Notifies the listening object that the task is about to start
	 * processing.  If the listener needs to provide data or keep
	 * track of the threads, this is the correct place to do it.
	 */
	public void notifyTaskStarting();
	
	/**
	 * Notifies the listening object that the task has already completed
	 * and that any resultant data may be used safely.
	 */
	public void notifyTaskFinished();
}
