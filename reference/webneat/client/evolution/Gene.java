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

import client.*;

/**
 * A Gene represents part of the genotype that stores historical marking
 * information.
 *
 * @author Nick Beato
 */

public interface Gene extends Transferable, Comparable <Gene> {
	/**
	 * Gets the historical marking of this gene.
	 * 
	 * @return The historical marking
	 */
	public Marking getMarking();
	
	/**
	 * Tests if this gene uses the specified marking.
	 *
	 *@param marking The marking to check
	 * @return <code>true</code> if the gene uses the marking, <code>false</code> otherwise
	 */
	public boolean matches(Marking marking);
	
	/**
	 * Assigns a random value to this gene.  This method is primarily called
	 * when starting evolution from scratch to initialize a default network. 
	 */
	public void randomize();
}
