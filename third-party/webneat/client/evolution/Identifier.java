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
 * An Identifier keeps track of the branch and id number of a genome in a series.
 * It differs from a Marking in that a marking is per gene, where the Identifier
 * is per genome.
 * 
 * @author Nick Beato
 */

public interface Identifier extends Transferable, Comparable <Identifier> {
	/**
	 * Gets the branch-specific id of this marking.  The ids are created by
	 * client during evolution and only stored by the server.
	 * 
	 * @return The branch-specific id of this marking
	 */
	public long getId();

	/**
	 * Gets the evolutionary branch of this marking as a String. This id
	 * is managed by the server, not by the client.
	 * 
	 * @return The evolution branch of this marking
	 */
	public String getBranch();
	
	/**
	 * Checks whether or not this identifier was created during the current
	 * session.
	 * 
	 * @return <code>true</code> if the identifier is new, <code>false</code> otherwise.
	 */
	public boolean usesCurrentBranch();
	
	/**
	 * Checks if the identifier is valid.  An identifier is only invalid if a genome
	 * is created via {@link GeneticFactory#createInvalidGenome()} until it is loaded
	 * from XML.
	 * 
	 * @return <code>true</code> if the identifier is valid, <code>false</code> otherwise.
	 */
	public boolean isValid();
}
