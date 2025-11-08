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
import client.server.ServerException;

/**
 * The Series represents a sequence of generations during evolution
 * (always starting from generation one). A series may be branched
 * from another series by initializing it with a genome. From the
 * perspective of the client, the first generation is relative
 * to this series.
 */

public interface Series extends Transferable {
	/**
	 * Initializes the first generation of a series depending on how
	 * it was created. Called to initialize the system and "redo"
	 * the initial generation
	 */
	public void initializeFirstGeneration();
	
	/**
	 * Counts the number of generations in this series object.
	 * 
	 * @return The number of generations
	 */
	public int getLength();
	
	/**
	 * Gets the name of the previous branch.
	 * 
	 * @return The identifier of the previous branch or <code>null</code>
	 *  if this is a root branch.
	 */
	public String getPreviousBranch();
	
	/**
	 * Gets a specific generation from the series.
	 * 
	 * @param generationNumber The generation number
	 * @return The generation
	 * @throws ServerException When data cannot be loaded from the server
	 */
	public Generation getGeneration(int generationNumber) throws ServerException;
	
	/**
	 * Gets a specific individual from the series.
	 * 
	 * @param generationNumber The generation the of the individual
	 * @param individualNumber The array index of the individual
	 * @return The individual
	 * @throws ServerException When data cannot be loaded from the server
	 */
	public Individual getIndividualFromGeneration(int generationNumber, int individualNumber) throws ServerException;
	
	/**
	 * Sets the name of this branch, modifying all markings and identifiers
	 * created with this branch.
	 * <p>
	 * Don't call this unless you know wtf you are doing.
	 * 
	 * @param name The new branch name
	 */
	public void setCurrentBranch(String name);
	
	/**
	 * Gets the name of this branch.
	 * <p>
	 * Don't call this unless you know wtf you are doing.
	 * <p>
	 * This data will be incorrect if cached and subsequent code
	 * invokes {@link Series#setCurrentBranch(String)}.
	 * You should use an identifier or marking when dealing with
	 * data that must still be correct after such an invocation.
	 * 
	 * @return The name of the current branch (at this time)
	 */
	public String getCurrentBranch();
	
	// should this stuff be in another class?
	/**
	 * Retrieves a reference to the current generation. This is not gauranteed to
	 * be persistent during run-time, so it should not be cached.
	 * 
	 * @return The current generation
	 */
	public Generation getCurrentGeneration() throws ServerException;
	
	/**
	 * Retrieves a referent to the particular individual in the current
	 * generation.
	 * 
	 * @param individualNumber The individual number
	 * @return The individual
	 * @throws ServerException
	 */
	public Individual getIndividualFromCurrentGeneration(int individualNumber) throws ServerException;
	
	// evolutionary controls
	/**
	 * Spawns a new generation based on the individuals selected
	 * in the current generation.
	 * 
	 * @throws EvolutionException If no parents are selected
	 */
	public void spawn() throws EvolutionException;
	
	/**
	 * Increments the current generation.
	 * <p>
	 * If the current generation is the most recent, this will do nothing.
	 */
	public void goForward();
	
	/**
	 * Decrements the current generation.
	 * <p>
	 * If the current generation is the oldest, this will do nothing.
	 */
	public void goBack();
	
	/**
	 * Checks if the current generation is the oldest.
	 * 
	 * @return <code>true</code> if the current generation is not the oldest, <code>false</code> otherwise.
	 */
	public boolean canGoBack();

	/**
	 * Checks if the current generation is the most recent.
	 * 
	 * @return <code>true</code> if the current generation is not the most recent, <code>false</code> otherwise.
	 */
	public boolean canGoForward();
	
	/**
	 * Prunes the series object for saving.
	 * <p>
	 * The pruning algorithm should remove all genomes that are not an
	 * ancestor of the representative genome.
	 */
	public void prune();
	
	/**
	 * Notifies the series that it successfully saved. This is only called
	 * once the client can gaurantee that the server state matches the client
	 * state.
	 */
	public void notifySaveSuccessful();
}

