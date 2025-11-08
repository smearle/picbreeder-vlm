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

import org.ksoap2.SoapEnvelope;
import org.ksoap2.serialization.SoapSerializationEnvelope;
import org.ksoap2.serialization.SoapObject;
import org.ksoap2.transport.HttpTransportSE;
import org.kobjects.base64.Base64;
import java.util.Vector;
import java.util.TreeMap;
import java.io.*;
import java.util.zip.*;
import client.ParameterTable;

/**
 * Handles Communication between the server and the client.
 *
 * @author David D'Ambrosio
 */

public class SOAPServer implements ServerConnection{

	private int parentId;

	private int seriesId;

	private String password;

	private String userName;

	private String ENDPOINT;
	
	private final String anonPass="00f7a1560074be4209e8f1fac851fda8";
	
	private boolean loggedIn;

	private final String NAMESPACE = "http://igood";
	
	private InputStream temporarySeries=null;
	private InputStream temporaryGenome=null;
	
	// used to close the zip file and get the result byte array
	// meant to work as a struct
	private final static class Streams {
		final ByteArrayOutputStream bytes;
		final ZipOutputStream zip;
		
		public Streams() {
			bytes = new ByteArrayOutputStream();
			zip = new ZipOutputStream(bytes);
		}
	}
	
	private TreeMap <String, Streams> storage = new TreeMap <String, Streams> ();
	private Streams series=null;
	private Streams genome=null;

	/**
	 * Sets up the server object parameters.
	 * 
	 * @param settings Contains all the parameters needed by the server
	 * 
	 */
	
	public void initialize(Initialization settings) throws FatalException, TimeoutException
	{
		seriesId = Integer.parseInt(settings.getParameter("seriesId"));
		parentId = Integer.parseInt(settings.getParameter("parentId"));
		userName = settings.getParameter("username");
		password = settings.getParameter("password");
		ENDPOINT = settings.getParameter("webservices");
		
		if(userName.equals("anonymous"))
		{
			loggedIn=false;
			password=anonPass;
		}
		else
		{
			loggedIn=true;
		}
		
		if (hasGenome() && hasSeries())  //should never happen
			throw new FatalException("Bad input data");
		else if(hasGenome())
			storeGenome();
		else if(hasSeries())
			storeSeries();
		
	}
	
	public ParameterTable getParameters() {
		return new test.TestParameters();
	}

	/**
	 * Checks if continuing a previously saved series
	 * 
	 * @return True if continuing a previously saved series, false otherwise
	 */
	
	public boolean hasSeries()
	{
		return (seriesId != -1);
	}
	
	/**
	 * Checks if branching from an existing genome
	 * 
	 * @return True if branching from an existing genome, false otherwise
	 */
	
	public boolean hasGenome()
	{
		return (parentId != -1);
	}
	
	/**
	 * Gets the represenative genome from the parent series
	 * 
	 * @return String with the XML of the represenative genome from the parent series
	 */
	
	public InputStream getGenome()
	{
		return temporaryGenome;
	}
	
	private void storeGenome() throws FatalException, TimeoutException
	{
		//temporaryGenome=new ByteArrayInputStream(unzip(callSoapFunction("getRepresentativeGenome",new Object[] { userName, password, parentId}).toString()).getBytes());
		String fromServer = callSoapFunction("getRepresentativeGenome",new Object[] { userName, password, parentId}).toString();
		temporaryGenome = createInputStream(fromServer);
	}
	
	/**
	 * Gets the represenative genome from an arbitrary series
	 * 
	 * @param branchName Id of the branch to retrive the genome from
	 * 
	 * @return String with the XML of the represenative genome from the parent series
	 */
	
	public InputStream getRepresentativeGenome(String branchName) throws FatalException, TimeoutException
	{
		//return new ByteArrayInputStream(unzip(callSoapFunction("getRepresentativeGenome",new Object[] { userName, password, Integer.parseInt(branchName)}).toString()).getBytes());
		String fromServer = callSoapFunction("getRepresentativeGenome",new Object[] { userName, password, Integer.parseInt(branchName)}).toString();
		return createInputStream(fromServer);
	}
	
	/**
	 * Gets files from the sever.
	 * 
	 * @param generations The integer ids of the files to be retrived
	 * 
	 * @return An array of strings represening the files in the order that they were requested
	 */
	
	// TODO outdated but never invoked
	public String[] getGenerationFiles(int[] generations) throws FatalException, TimeoutException
	{
		//need to build vectors to hold the arrays for KSoap
		Vector<Integer> tempGens=new Vector<Integer>();
		for(int j=0;j<generations.length;j++)
			tempGens.add(generations[j]);
		
		Object response=callSoapFunction("getGenerations",new Object[] {userName, password, seriesId, tempGens} );
		
		//get a vector back, turn it into an array of strings
		Vector vect=(Vector)response;
		String[] gens=new String[vect.size()];
		for(int j=0;j<vect.size();j++)
		{
			gens[j]=unzip(vect.elementAt(j).toString());
		}
		return gens;
	}
	
	/**
	 * Gets a series file from the server
	 * 
	 * @param name The id of the file to be retrived
	 * 
	 * @return A string representing the file requested
	 */
	public InputStream loadStorage(String name) throws FatalException, TimeoutException
	{
		Vector<Integer> vec=new Vector<Integer>();
		vec.add(Integer.parseInt(name));
		
		Object response=callSoapFunction("getGenerations",new Object[] {userName, password, seriesId, vec} );
		
		//get a vector back, turn it into an array of strings
		Vector vect=(Vector)response;
		String[] gens=new String[vect.size()];
		gens[0]=vect.elementAt(0).toString();
		//return new ByteArrayInputStream(unzip(gens[0].toString()).getBytes());
		
		return createInputStream(gens[0]);
	}
	
	/**
	 * Saves the current series.
	 * 
	 * @param series XML data file that indexes files, parameters, etc.
	 * 
	 * @param removeThese Array of integers representing generation files that must be removed (ie. {1,2} removes 1.xml and 2.xml)
	 * 
	 * @param addThese Array of integers representing generation files that must be added (ie. {1,2} adds 1.xml and 2.xml)
	 * 
	 * @param newData Array of Strings, each string is the data for an xml file listed in addThese
	 * 
	 * @param genome XML data for the representinve genome of the series to be saved
	 * 
	 */
	
	public void save(String series, String genome, String [] removeThese, String [] addThese, String [] newData) throws FatalException, TimeoutException
	{
		
		Vector<Integer> tempAdds=new Vector<Integer>(addThese.length);
		Vector<Integer> tempDeletes=new Vector<Integer>(removeThese.length);
		Vector<String> tempAddData=new Vector<String>(newData.length);
		try{
		for(int adds=0;adds<addThese.length;adds++)
			tempAdds.add(Integer.parseInt(addThese[adds]));
		
		for(int dels=0;dels<removeThese.length;dels++)
			tempDeletes.add(Integer.parseInt(removeThese[dels]));
		
		for(int data=0;data<newData.length;data++)
			tempAddData.add(zip(newData[data]));
		}catch(Exception e){return;}
		
		callSoapFunction("saveSeries",new Object[] {userName, password, seriesId, zip(series),tempDeletes,tempAdds,tempAddData,zip(genome)});
	}
	
	/**
	 * Loads the main series file from the server.
	 * 
	 * @return The XML series file in a string
	 */	
	
	public InputStream getSeries() 
	{
		return temporarySeries;
	}
	
	private void storeSeries() throws FatalException, TimeoutException
	{
		//temporarySeries=new ByteArrayInputStream(unzip(callSoapFunction("getSeriesXML", new Object[] { userName, password, seriesId }).toString()).getBytes());
		String fromServer = callSoapFunction("getSeriesXML", new Object[] { userName, password, seriesId }).toString();
		temporarySeries = createInputStream(fromServer);
	}
	
	public void clientStarted() {
		if(temporaryGenome != null) {
			try {
				temporaryGenome.close();
			}
			catch(Exception e) {
			}
			temporaryGenome = null;
		}
		if(temporarySeries != null) {
			try {
				temporarySeries.close();
			}
			catch(Exception e) {
			}
			temporarySeries = null;
		}
	}
	
	/**
	 * Calls a remote function using SOAP.
	 * 
	 * @param functionName The name of the remote function to be called
	 * 
	 * @param arguments The arguments to the function being called, as an array of Objects, in the same order as the parameter list
	 * 
	 * @return Returns an Object containing the return value of the function or null if return type is void or if there is an error
	 */
	private Object callSoapFunction(String functionName, Object[] arguments) throws FatalException, TimeoutException
	{
		HttpTransportSE transport=new HttpTransportSE(ENDPOINT);
		SoapObject soapobj = new SoapObject(NAMESPACE, functionName);
		for (int args = 0; args < arguments.length; args++)
		{
			soapobj.addProperty("arg" + args, arguments[args]);
		}
		SoapSerializationEnvelope env = new SoapSerializationEnvelope(SoapEnvelope.VER11);
		env.bodyOut = soapobj;
		try{
			transport.call("http://www.w3.org/2001/12/soap-envelope", env);
			return env.getResponse();
		}
		catch(java.io.IOException io) 
		{
			throw new TimeoutException();
		}
		catch(Exception e)
		{
			throw new FatalException("Data Error");
		}
		
	}
	
	/**
	 * Gets the series ID for the current series, if the series is new this comes from the server
	 * 
	 * @return the current series ID
	 */
	
	public String getSeriesName() throws FatalException, TimeoutException
	{
		if (seriesId == -1)
		{
			seriesId = ((Integer)callSoapFunction("getNewSeriesID", new Object[] { userName, password, parentId })).intValue();
		}
		return Integer.toString(seriesId);
	}
	
	/**
	 * Zips a string into a base64 format.
	 * 
	 * @param toBeZipped The string to be zipped
	 * 
	 * @return The compressed version of the string
	 */	
	public String zip(String toBeZipped)
	{
		byte[] data=client.utilities.Zip.zip(toBeZipped);
		return Base64.encode(data);
	}
	/**
	 * Unzips a string that is in base64 format.
	 * 
	 * @param toBeUnzipped The string to be unzipped
	 * 
	 * @return The uncompressed version of the string
	 */	
	public String unzip(String toBeUnzipped)
	{
		return client.utilities.Zip.unzip(Base64.decode(toBeUnzipped));
	}
	
	public void shutDown()
	{
		return;
	}
	
	public OutputStream getSaveStreamForSeries() throws FatalException
	{
		series = createOutputStream();
		return series.zip;
	}
	
	public OutputStream getSaveStreamForGenome() throws FatalException
	{
		genome = createOutputStream();
		return genome.zip;
	}
	
	public OutputStream getSaveStreamForStorage(String name) throws FatalException
	{
		Streams streams = createOutputStream();
		storage.put(name, streams);
		return streams.zip;
	}
	
	public void save(String [] removeThese, String [] addThese)
		throws FatalException, TimeoutException
	{
		//if(!isUserLoggedIn())
		//	throw new FatalException("You can save this way if you are not logged in.");
		
		Vector<Integer> tempAdds=new Vector<Integer>(addThese.length);
		Vector<Integer> tempDeletes=new Vector<Integer>(removeThese.length);
		Vector<String> tempAddData=new Vector<String>(addThese.length);
		
		try{
			for(int adds=0;adds<addThese.length;adds++)
				tempAdds.add(Integer.parseInt(addThese[adds]));
			
			for(int dels=0;dels<removeThese.length;dels++)
				tempDeletes.add(Integer.parseInt(removeThese[dels]));
			
			for(String name : addThese)
				//tempAddData.add(zip(storage.get(name).toString()));
				tempAddData.add(createTextFromStream(storage.get(name)));
			
		}
		catch(Exception e) {
			throw new FatalException(e.getMessage());
		}
		
		storage.clear();
		
		String seriesXml = createTextFromStream(series);
		String genomeXml = createTextFromStream(genome);
		
		try {
			callSoapFunction("saveSeries",new Object[] {userName, password, seriesId, seriesXml,tempDeletes,tempAdds,tempAddData,genomeXml});
		}
		catch(Exception e) {
			throw new TimeoutException();
		}

	}

	// this function will decode the input, unzip the first file, and return a stream
	// representing the data in the file... it's messy
	private InputStream createInputStream(final String dataFromServer) throws FatalException {
		try {
			PipedInputStream inPipeStream = new PipedInputStream();
			ZipInputStream inStream = new ZipInputStream(inPipeStream);
			final OutputStream outputStream = new PipedOutputStream(inPipeStream);

			// must decompress in the background or the streams may deadlock
			new Thread(
				new Runnable() {
					public void run() {
						try {
							Base64.decode(dataFromServer, outputStream);
							Thread.sleep(1000);
						}
						catch(Exception e) {
							e.printStackTrace();
							throw new RuntimeException(e);
						}
					}
				}).start();

			// must be invoked after the thread is started or it will deadlock!
			inStream.getNextEntry();
			return inStream;
		}
		catch(Exception e) {
			throw new FatalException("Streaming data didn't work!");
		}
	}
	
	private Streams createOutputStream() throws FatalException {
		try {
			Streams s = new Streams();
			s.zip.putNextEntry(new ZipEntry("a"));
			return s;
		}
		catch(IOException e) {
			throw new FatalException("Zip file could not be created!");
		}
	}
	
	private String createTextFromStream(Streams data) throws FatalException {
		try {
			data.zip.finish();
			data.zip.flush();
			data.zip.closeEntry();
			data.zip.close(); // maybe free some memory?
			String result = Base64.encode(data.bytes.toByteArray());
			return result;
			
		}
		catch(IOException e) {
			throw new FatalException("Could not properly zip the data.");
		}
	}
	
	public boolean isUserLoggedIn() {
		return loggedIn;
	}
	
	public void logInExistingUser(String userName, String encodedPassword) throws LoginException, TimeoutException {
		
		int logInResult=-1;
		
		try {
			logInResult=(Integer)callSoapFunction("authenticate",new Object[] {userName, encodedPassword });
		}
		catch(Exception e) {
			throw new TimeoutException();
		}
		
		if(logInResult==-1)
			throw new LoginException("Wrong Username or Password.");
		else if(logInResult==0)
			throw new LoginException("You cannot login with that username.");
		else if(isUserAtUnpublishedLimit(logInResult))
		{
			throw new LoginException("Your account is at its limit of unpublished images so your image cannot be saved.");
		}
		else
		{
			this.userName=userName;
			password=encodedPassword;
			loggedIn=true;
		}
	}
	
	public void saveAnonymously(String [] addThese) throws FatalException, TimeoutException
	{
		if(isUserLoggedIn())
			throw new FatalException("You cannot save anonymously if you are logged in.");
		Vector<Integer> tempAdds=new Vector<Integer>(addThese.length);
		Vector<String> tempAddData=new Vector<String>(addThese.length);
	
		try{
			for(int adds=0;adds<addThese.length;adds++)
				tempAdds.add(Integer.parseInt(addThese[adds]));
			
			for(String name : addThese)
				tempAddData.add(createTextFromStream(storage.get(name)));
			
		}
		catch(Exception e) {
			throw new FatalException(e.getMessage());
		}
	
		storage.clear();
		
		String seriesXml = createTextFromStream(series);
		String genomeXml = createTextFromStream(genome);
		
		try {
			callSoapFunction("saveSeries",new Object[] {userName, password, seriesId, seriesXml,new Vector<Integer>(),tempAdds,tempAddData,genomeXml});
		}
		catch(Exception e) {
			throw new TimeoutException();
		}
	}
	
	public String getGUID() throws TimeoutException,FatalException
	{
		Object guid;
		
		if(isUserLoggedIn())
			throw new FatalException("You cannot save anonymously if you are logged in.");
		try {
			guid=callSoapFunction("getAnonymousEntry",new Object[] {seriesId});
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new TimeoutException();
		}
		return guid.toString();
	}
	
	public String getPassword()
	{
		return password;
	}
	
	public String getUsername()
	{
		return userName;
	}
	
	private boolean isUserAtUnpublishedLimit(int userId) throws TimeoutException
	{
		Object result;
		if(isUserLoggedIn())
			return false;
		else
		{
			try
			{
				result=callSoapFunction("atUnpublishedLimit",new Object[] {userId});
			}
			catch(Exception e)
			{
				e.printStackTrace();
				System.err.println("publish limit");
				throw new TimeoutException();
			}
			return (Boolean) result;
		}
	}
	public void publishAnonymously(String [] addThese) throws FatalException, TimeoutException
	{
		saveAnonymously(addThese);
	}
	
	
}

