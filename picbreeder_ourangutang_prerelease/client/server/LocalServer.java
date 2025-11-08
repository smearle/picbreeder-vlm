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

import client.*;
import java.util.*;
import java.util.zip.ZipFile;
import java.io.*;

import test.TestParameters;

/**
 * Intended for testing only.
 * 
 * @author Nick
 *
 */
public class LocalServer implements ServerConnection {
	private String loadPath = "";
	private String fileExt = "";
	private InputStream genome = null;
	private InputStream series = null;
	private File outputFile = null;
	private ParameterTable parameterTable = null;
	
	private Collection <OutputStream> openStreams = new java.util.LinkedList <OutputStream> ();
	
	public void initialize(Initialization params) {
		try {
			String param = params.getParameter("Path");
			if(param != null)
				loadPath = param;
			
			param = params.getParameter("Extension");
			if(param != null)
				fileExt = param;
			
			param = params.getParameter("Series");
			if(param != null)
				series = openInputStream(loadPath + param);
			
			param = params.getParameter("Genome");
			if(param != null)
				genome = openInputStream(loadPath + param);
			
			param = params.getParameter("Parameters");
			if(param != null) {
				InputStream in = openInputStream(param);
				parameterTable = new XmlParameterTable();
				client.utilities.XML.loadConfiguration((Configurable) parameterTable, in);
				in.close();
			}
			else
				parameterTable = new test.TestParameters();
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	private InputStream openInputStream(String fileName) throws Exception {
		if(fileName.endsWith(".zip")) {
			ZipFile zipfile = new ZipFile(fileName);
			return new BufferedInputStream(zipfile.getInputStream(zipfile.entries().nextElement()));
		}
		else
			return new BufferedInputStream(new FileInputStream(new java.io.File(fileName)));
	}

	
	public InputStream getSeries() {
		return series;
	}
	
	public boolean hasSeries() {
		return series != null;
	}
	
	public InputStream getGenome() {
		return genome;
	}

	public InputStream getRepresentativeGenome(String branchName) {
		return null;
	}
	
	public boolean hasGenome() {
		return genome != null;
	}
	
	public String getSeriesName() {
		//return "theMostAwesomestSeries";
		
		return "shouldBeBranch";
	}
	
	public String getFileFormat() {
		return "jpeg";
	}
	
	public ParameterTable getParameters() {
		return parameterTable;
	}
	
	public OutputStream getSaveStreamForSeries() {
		javax.swing.JFileChooser f = new javax.swing.JFileChooser();
		
		if(javax.swing.JFileChooser.APPROVE_OPTION == f.showSaveDialog(client.MainComponentInstance.get())) {
			outputFile = f.getSelectedFile().getAbsoluteFile();
		
			try {
				if(!outputFile.getName().substring(outputFile.getName().length() - 4).equals(".xml"))
					outputFile = new File(outputFile.getName() + fileExt).getAbsoluteFile();
				else
					fileExt = outputFile.getName().substring(outputFile.getName().lastIndexOf('.'));
				
				OutputStream s = new BufferedOutputStream(new FileOutputStream(outputFile));
				openStreams.add(s);
				System.out.println("Writing: " + outputFile.getName());
				return s;
			}
			catch(IOException e) {
				outputFile = null;
				e.printStackTrace();
			}
		}
		
		return null;
	}
	
	public OutputStream getSaveStreamForGenome() {
		if(outputFile == null)
			return null;
		
		String name = outputFile.getParent() + java.io.File.separator + "genome" + fileExt;

		try {
			OutputStream s = new BufferedOutputStream(new FileOutputStream(new File(name)));
			openStreams.add(s);
			System.out.println("Writing: " + name);
			return s;
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		
		return null;
	}
	
	public OutputStream getSaveStreamForStorage(String name) {
		if(outputFile == null)
			return null;
		
		name = outputFile.getParent() + java.io.File.separator + name + fileExt;

		try {
			OutputStream s = new BufferedOutputStream(new FileOutputStream(new File(name)));
			openStreams.add(s);
			System.out.println("Writing: " + name);
			return s;
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		
		return null;
	}
	
	public void save(String []removeThese, String []addThese) throws FatalException {
		if(outputFile == null)
			throw new FatalException("bad file");
		
		for(String s : removeThese)
			System.out.println("REMOVED " + s);
		 
		for(String s : addThese)
			System.out.println("ADDED " + s);
		
		for(OutputStream s : openStreams) {
			try {
				s.close();
			}
			catch(IOException e) {
			}
		}
		
		openStreams.clear();
		outputFile = null;		
	}

	public void saveAnonymously(String []addThese) throws FatalException {
		save(new String[]{}, addThese);
	}
	
	public void shutDown() {
	}
	
	public void clientStarted() {
		if(series != null) {
			try {
				series.close();
			}
			catch(Exception e) {
			}
			series = null;
		}
		
		if(genome != null) {
			try {
				genome.close();
			}
			catch(Exception e) {
			}
			genome = null;
		}
	}
	
	public InputStream loadStorage(String name) throws TimeoutException {
		try {
			return openInputStream(loadPath + name + fileExt);
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}
	
	public boolean isUserLoggedIn() {
		return false;
	}
	
	public void logInExistingUser(String user, String pass) {
	}
	
	public String getGUID() throws TimeoutException,FatalException
	{
		return "";
	}
	
	public String getPassword()
	{
		return "";
	}
	
	public String getUsername()
	{
		return "";
	}
	
}
