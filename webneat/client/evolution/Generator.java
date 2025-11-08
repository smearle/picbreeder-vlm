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

import java.util.*;

/**
 * A Generator is used to create Genomes.  The key feature of a generator is
 * that it may take Genomes as input to its process, allowing it to perform
 * mutations, crossovers, etc.  All of the population generation will be done
 * via the Generators, allowing us to easily control how the population gets
 * created.
 * 
 * @author Nick Beato 
 */
public interface Generator {
	/**
	 * Creates a new genome from the specified parents using this
	 * generator's method(s).  The produced genome will be a deep copy
	 * of the parents' genes.  In other words, modifying the genes of
	 * the produced genome will not effect the parents.
	 * 
	 * 
	 * @param parents The parents of the produced offspring 
	 * @return The newly constructed Genome
	 */

	public Genome generate(Collection<Genome> parents);
	
	/**
	 * Queries the generator to determine the minimum number of unique
	 * parents the generator needs to successfully produce an offspring
	 * in the {@link Generator#generate(Collection)} method. This method
	 * ensures that a generator will never be incorrectly invoked.
	 * <p>
	 * For mutations, this will always return 1. For crossovers, this will
	 * most likely return 2. It may return more than two for multiparent
	 * crossovers.
	 * 
	 * @return The minimum number of parents required by this generator
	 */
	public int minimumParents();
}


