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

import java.util.Set;
import java.util.Map;
import client.Transferable;

/**
 * The Database stores all of the information that is transferred
 * between the client and the server. It maps a set of names, stored
 * in the series, to the generations grouped with those names.  All of
 * these are loaded "just in time" from the server.
 * 
 * The Database can be viewed as the "server copy" of all data. This
 * way all management of resources separated from the series.
 * 
 * @author Nick
 */
public interface Database {
	/**
	 * Adds a generation to the server. This will return the
	 * storage name to the calling method. The storage name indicates
	 * what "file" the generation is within.
	 * 
	 * @param generation The generation to add
	 * @return The name of the storage file containing the generation.
	 */
	public String addGeneration(Generation generation);
	
	/**
	 * Removes the given generation from the server. This
	 * assumes the generation exists on the server!
	 * 
	 * @param generation The generation to remove
	 */
	public void removeGeneration(Generation generation);
	
	/**
	 * Indicates that the server should have a storage file
	 * with the given name. This is used to make sure that
	 * things can be loaded.
	 * 
	 * @param name The name of the storage file
	 */
	public void ensureServerContains(String name);
	
	/**
	 * Clears the database, indicating the client wants to
	 * reinitialize it.
	 */
	public void clearServer();
	
	/**
	 * Gets the generation from the server.  The storage name
	 * is stored in the series.
	 * 
	 * @param storageName The storage location from the series
	 * @param number The generation number
	 * @return The generation
	 * @throws FatalException If the data is not available
	 * @throws TimeoutException If the server connection failed
	 */
	public Generation retrieveGeneration(String storageName, int number)
		throws client.server.FatalException, client.server.TimeoutException;
	
	/**
	 * Instructs the database to prefetch the given generation, as it
	 * may be retrieved in the near future.
	 * 
	 * @param storageName The storage location from the series
	 * @param number The generation number
	 */
	public void prefetchGeneration(String storageName, int number);
	
	/**
	 * Retreives a set of storage files that should be removed from
	 * the real server during a save operation to synchronize it with
	 * this object.
	 * 
	 * @return The set of storage files that need to be removed
	 */
	public Set <String> getRemovalList();
	
	/**
	 * Retreives a map of storage files that should be added to
	 * the real server during a save operation to synchronize it with
	 * this object.
	 * 
	 * @return The names and data of storage files to add to the server
	 */
	public Map <String, Transferable> getAdditionList();
}
