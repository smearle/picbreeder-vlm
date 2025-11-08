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

package client.gui;

import java.io.IOException;
import java.io.OutputStream;
import java.io.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLConnection;
import java.util.Vector;

import javax.swing.JApplet;
import javax.swing.JOptionPane;

import client.server.*;
import client.*;
import client.evolution.GeneticFactoryInstance;
import client.evolution.Genome;
import client.evolution.Series;

/**
 * This is the main applet entry point.
 * 
 * @author Nick
 */

public class Applet extends JApplet implements Initialization, ImageDatabase, SessionEnd {
	private java.awt.Component sessionPanel = null;

	static {
		SessionHandler.setClassLoader(Applet.class.getClassLoader());
	}
	
	public Applet() {
		super();
	}
	
	public void init() {
	}
	
	public void start() {
		assert(sessionPanel == null);
		
		try {
			ServerConnection server = new client.server.SOAPServer();
			server.initialize(this);
			SessionHandler.beginSession(server);
		
			ImageDatabaseInstance.set(this);
			MainComponentInstance.set(this);
			
			Series series = null;
			client.ImageDatabaseInstance.set(new FileSystemImageDatabase()); 
			
			if(client.server.ServerConnectionInstance.get().hasSeries()) {
				series = GeneticFactoryInstance.get().createInvalidSeries();
				
				java.io.InputStream xml = client.server.ServerConnectionInstance.get().getSeries();
				client.utilities.XML.load(series, xml);
			}
			else if(client.server.ServerConnectionInstance.get().hasGenome()) {
				Genome genome = GeneticFactoryInstance.get().createInvalidGenome();
	
				java.io.InputStream xml = client.server.ServerConnectionInstance.get().getGenome();
				
				client.utilities.XML.load(genome, xml);
				series = GeneticFactoryInstance.get().createBranchSeries(genome);
			}
			else
				series = GeneticFactoryInstance.get().createRootSeries();
		
			client.server.ServerConnectionInstance.get().clientStarted();
			SeriesInstance.set(series);
			
			sessionPanel = new MainPanel();
			add(sessionPanel);
		}
		catch(Exception e) {
			JOptionPane.showMessageDialog(this, e.getMessage(), e.getClass().getName(), JOptionPane.ERROR_MESSAGE);
			e.printStackTrace();
			//TODO: figure out where this should redirect
			redirect("mypanel.php");
			return;
		}
	}
	
	public void stop() {
		if(sessionPanel != null) {
			SessionHandler.endSession();
			this.remove(sessionPanel);
			sessionPanel = null;
		}
	}
	
	public void destroy() {
	}
	
	public java.awt.Image getImage(String image) {
		return getImage(getCodeBase(), image);
	}

	public void publish() {
		String seriesId = SeriesInstance.get().getCurrentBranch();
		redirect("editgenome?sid="+seriesId+"&pid=0");
	}
	
	public void quit() {
		
		if(client.server.ServerConnectionInstance.get().isUserLoggedIn())
			redirect("mypanel.php");
		else
			redirect("../index.php");
	}
	
	public void register() {
		
		try{
			String guid=client.server.ServerConnectionInstance.get().getGUID();
			redirect("register.php?GUID="+guid);
		}
		catch(Exception e){e.printStackTrace();}
		
	}
	
	private void redirect(String page) {
		try {
			getAppletContext().showDocument(new java.net.URL(getCodeBase().toString()+page),"_self");
		}
		catch(Exception e) {
		}
	}
	
	public void authenticate(String userName, char[] password)
	{ 
		try {
		  URL url;
		  URLConnection con;
		  OutputStream oStream;
		  String parametersAsString;
		  byte[] parameterAsBytes, fullParameters;
		  String aLine;
		  parametersAsString="myusername="+userName+"&mypassword=";

		  byte[]b = new byte[password.length];
			for(int i = 0; i < b.length; i++)
				b[i] = (byte) password[i];
		  
		  parameterAsBytes = parametersAsString.getBytes();
		  
		  fullParameters= new byte[parameterAsBytes.length+b.length];
		  System.arraycopy(parameterAsBytes, 0, fullParameters, 0, parameterAsBytes.length);
		  System.arraycopy(b, 0, fullParameters, parameterAsBytes.length, b.length);
		  
		  java.util.Arrays.fill(password, 0, password.length, (char)0);
			password = null;
			
		  java.util.Arrays.fill(b, 0, b.length, (byte)0);
			b = null;
		  
		  // send parameters to server
		  url = this.getCodeBase();
		  url = new URL(url+"../login/login.php?target=NONE");
		  con = url.openConnection();
		  con.setDoOutput(true);
		  con.setDoInput(true);
		  //con.setDoInput(false);
		  con.setRequestProperty("Content=length", String.valueOf(fullParameters.length));
		  oStream = con.getOutputStream();
		  oStream.write(fullParameters);
		  oStream.flush();
		  // read response from server
		  BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
		  aLine = in.readLine();
		  while (aLine != null)
		  { System.out.println(aLine);
		    aLine = in.readLine();
		  }
		  in.close(); 
		  // read response from server
		  oStream.close();
		  java.util.Arrays.fill(fullParameters, 0, fullParameters.length, (byte)0);
		  fullParameters = null;
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
}
