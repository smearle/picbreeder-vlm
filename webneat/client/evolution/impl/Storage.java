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

package client.evolution.impl;

import client.Transferable;
import client.evolution.Generation;

public interface Storage extends Transferable, Iterable <Generation> {
	public String getName();
	/**
	 * Returns the number of generations stored in this container.
	 * 
	 * @return The number of generations in this storage object
	 */
	public int size();
	
	/**
	 * Gets the minimum generation number in this storage object.
	 * 
	 * @return The minimum generation number
	 */
	public int getMinimum();
	
	/**
	 * Gets the maximum generation number in this storage object.
	 * 
	 * @return The maximum generation number
	 */
	public int getMaximum();
	
	/**
	 * Returns the generation at the given age.  If the generation is
	 * not in this container, the program will throw an exception (TBD).
	 * 
	 * @param number The generation number
	 * @return The generation
	 */
	public Generation getGeneration(int number);
	
	/**
	 * Adds a generation into this container.
	 * 
	 * @param generation The generation
	 */
	public void addGeneration(Generation generation);
	
	/**
	 * Removes a generation from this container.
	 * 
	 * @param generation The generation
	 */
	public void removeGeneration(Generation generation);
	
	/**
	 * Checks if this container is empty.
	 * 
	 * @return <code>true</code> if empty, <code>false</code> otherwise.
	 */
	public boolean isEmpty();
}
