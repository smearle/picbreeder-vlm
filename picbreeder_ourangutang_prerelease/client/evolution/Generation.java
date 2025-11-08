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

import java.util.Collection;
import client.*;

/**
 * The Generation class represents a single generation during
 * the evolution of a series. It contains a population of
 * individuals. There are accessors to get particular individuals
 * and accessors to get the canonical representative of this
 * generation (usually the highest fitness individual).
 * 
 * @author Nick Beato
 */

public interface Generation extends Transferable, Iterable <Individual> {
	/**
	 * Gets the number of individuals currently in this generation.
	 * 
	 * @return The population size
	 */
    public int getSize();
    
    /**
     * Gets the generation number (relative to the current series).
     * 
     * @return The generation number
     */
    public int getNumber();
    
    /**
     * Gets an individual from the population. The parameter <code>number</code>
     * should be in the range of <code>[0, getSize())</code>.
     * 
     * @param number The individual number
     * @return The Individual
     */
    public Individual getIndividual(int number);
    
    /**
     * Gets the canonical representative of this generation. This is an individual
     * with the highest fitness.
     * 
     * @return The individual, or <code>null</code> if none are selected.
     */
    public Individual getRepresentative();
    
    /**
     * Restores the current generation to the correct population size.
     * The parents of the previous generation are supplied to generation
     * new individuals if nescessary.
     * <p>
     * This operation is intended to "undo" the {@link Series#prune()}
     * method, which will destroy individuals that are not related to
     * individuals being saved.
     * 
     * @param parents The parents of the previous generation
     */
    public void restore(Collection <Genome> parents); // reverses pruning
    
    /**
     * Gets the individuals from this generation. This is used
     * to break the individuals into groups during renderering.
     * 
     * @return The individuals
     */
    public Collection <Individual> getIndividuals();
}


