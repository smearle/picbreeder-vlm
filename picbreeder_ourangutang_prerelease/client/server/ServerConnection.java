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

package client.server;

import client.ParameterTable;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Defines the communication points between the client and server, from the
 * client's perspective.  All communication and data encryption will be
 * abstracted from the client in this interface.
 *
 * @author Nick Beato
 */

public interface ServerConnection {
	/**
	 * Initializes the server with the provided settings.  If the server
	 * requires information not provided, it is considered a debugging error.
	 * So just throw a RuntimeException.
	 * 
	 * @param settings The parameters
	 * @throws FatalException The initialization failed
	 * @throws TimeoutException The server is not responding
	 */
	public void initialize(Initialization settings)
		throws FatalException, TimeoutException;
	
	/**
	 * Retrieves the series to modify for this session.  This function
	 * will only be called if the {@link ServerConnection#hasSeries()} function
	 * succesfully returns.
	 * <p>
	 * This function implies that the user is modifying an existing
	 * series from a save operation.  The series' name is valid and may
	 * contain multiple generations.  Evolution will start at the oldest
	 * generation.
	 *
	 * @return An XML representation of the series in a stream
	 */
	public InputStream getSeries();

	/**
	 * Queries the server to see if it has a series to modify.  This should
	 * only return <code>true</code> if the series was already saved by the
	 * user and has a name.  In that event, an immediate call to {@link ServerConnection#getSeries()}
	 * will be made to establish evolution.
	 * <p>
	 * The events that cause this function to fail are:
	 * <ul>
	 * 	<li>The user is branching from an existing published series.</li>
	 *	<li>The user is creating a new root series.</li>
	 * </ul>
	 *
	 * @return <code>true</code> if the server has a series object, 
	 * 	<code>false</code> otherwise.
	 */
	public boolean hasSeries();


	/**
	 * Retrieves the seed genome to branch for this session.  This function
	 * will only be called if the {@link ServerConnection#hasGenome()} function
	 * returns <code>true</code> <b>and</b> {@link ServerConnection#hasSeries()} returns
	 * <code>false</code>.
	 * <p>
	 * This function implies that the user is branching a published
	 * genome from the website.  The series' name is invalid and must be
	 * acquired via the {@link ServerConnection#getSeriesName()} operation. 
	 * Evolution will start by creating mutations of the genome.
	 *
	 * @return An XML representation of the genome in a stream
	 */
	public InputStream getGenome();
	
	/**
	 * Retrieves the representative genome of a particular branch.
	 * This function is invoked when the {@link ServerConnection#hasSeries()} method
	 * returns <code>true</code>.  When the user needs access to
	 * generation 0, it will invoke this call to restore the pruned
	 * individuals in that generation via the branchFrom genome.
	 *
	 * @param branchName The branch to get the genome from
	 * @return An XML representation of the genome in a stream
	 * @throws FatalException The server could not process the request
	 * @throws TimeoutException The server is not responding
	 */
	public InputStream getRepresentativeGenome(String branchName)
		throws FatalException, TimeoutException;

	/**
	 * Queries the server to see if it has a genome to modify.  This should
	 * only return <code>true</code> if the genome is something that was
	 * published <b>and</b> no series exists to modify, an immediate call 
	 * to {@link ServerConnection#getGenome()} will be made to establish evolution.
	 * <p>
	 * The events that cause this function to fail are:
	 * <ul>
	 * 	<li>The user is modifying a saved series.</li>
	 *	<li>The user is creating a new root series.</li>
	 * </ul>
	 *
	 * @return <code>true</code> if the server has a genome object, 
	 * 	<code>false</code> otherwise.
	 */
	public boolean hasGenome();

	/**
	 * Queries the server to get the name of a newly branched series.
	 * For robustness, this function should only return a new name
	 * if the user is not modifying a saved series.  This function will
	 * immediately be followed by an invocation
	 * {@link ServerConnection#save(String[], String[])}.
	 *
	 * @return The name of the series (even if it matches that in the series object).
	 * @throws FatalException The server could not process the request
	 * @throws TimeoutException The server is not responding
	 */
	public String getSeriesName()
		throws FatalException, TimeoutException;

	/**
	 * Queries the server for the image format to use.  This should be
	 * the extension of the file to use, such as "jpg" or "png".
	 *
	 * @return The file format for binary image files
	 */
	// not required anymore
	//public String getFileFormat();
	
	/**
	 * Gets the OutputStream that the series will save to.
	 * This data will be flushed to the server by invoking the {@link ServerConnection#save(String[], String[])}
	 * method.
	 * 
	 * @return The OutputStream for the series
	 * @throws FatalException The stream cannot be created
	 */
	public OutputStream getSaveStreamForSeries()
		throws FatalException;
	
	/**
	 * Gets the OutputStream that the representative genome will save to.
	 * This data will be flushed to the server by invoking the {@link ServerConnection#save(String[], String[])}
	 * method.
	 * 
	 * @return The OutputStream for the representative genome
	 * @throws FatalException The stream cannot be created
	 */
	public OutputStream getSaveStreamForGenome()
		throws FatalException;
	
	/**
	 * Gets the OutputStream that the named storage will save to.
	 * This data will be flushed to the server by invoking the {@link ServerConnection#save(String[], String[])}
	 * method.
	 * 
	 * @param name The storage name
	 * @return The OutputStream for the storage
	 * @throws FatalException The stream cannot be created
	 */
	public OutputStream getSaveStreamForStorage(String name)
		throws FatalException;

	/**
	 * Saves a series to the server.
	 * <p>
	 * The server should first remove all files within the <code>removeThese</code>
	 * array.  Then, it should add the files in the <code>addThese</code> array, even
	 * if the file name appears in the <code>removeThese</code> array.  The data for
	 * the <code>i</code>th file in the <code>addThese</code> is stored by writing
	 * to the outputstream using the same identifier.
	 * <p>
	 * It is gauranteed that all streams are ready for writing when this function
	 * is invoked.
	 *
	 * @param removeThese The data files to remove
	 * @param addThese The data files to add
	 * @throws FatalException The server could not process the request
	 * @throws TimeoutException The server is not responding
	 */
	public void save(String [] removeThese, String [] addThese)
		throws FatalException, TimeoutException;
	
	public void saveAnonymously(String []addThese)
		throws FatalException, TimeoutException;

	/**
	 * Retrieves the parameters for this session.
	 *
	 * @return The parameters
	 */
	public ParameterTable getParameters();

	/**
	 * Notifies the server that the client has quit.
	 */
	public void shutDown();
	
	/**
	 * Called to indicate that the client has succesfully initialized
	 * and is evolving.  This can be used to clean up some initialization
	 * memory.
	 */
	public void clientStarted();
	
	/**
	 * Retrieves the series data for the specified file.
	 * 
	 * @param name The storage object's name
	 * @return The storage data
	 * @throws FatalException The server could not process the request
	 * @throws TimeoutException The server is not responding
	 */
	public InputStream loadStorage(String name)
		throws FatalException, TimeoutException;
	
	/**
	 * Queries if a user is currently logged into the server from the client.
	 * 
	 * @return <code>true</code> if a user is logged in, <code>false</code> otherwise.
	 */
	public boolean isUserLoggedIn();
	
	/**
	 * Logs an existing user into the system.
	 * 
	 * @param userName The user handle
	 * @param encodedPassword A hashed version of the user's password
	 * @throws LoginException There is an error logging in
	 * @see client.utilities.PasswordEncoder
	 */
	public void logInExistingUser(String userName, String encodedPassword) throws LoginException, TimeoutException;
	
	/**
	 * Checks if a user logged in during evolution and needs to be logged in on the website
	 * 
	 * @return <code>true</code> if a user logged in, <code>false</code> otherwise.
	 */
	
	public String getGUID() throws TimeoutException,FatalException;
	
	public String getPassword();
	
	public String getUsername();
	
}
