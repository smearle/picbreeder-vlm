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

import client.ImageDatabaseInstance;
import client.MainComponentInstance;
import client.ParameterTableInstance;
import client.server.ServerConnection;
import client.server.ServerConnectionInstance;
import client.Singleton;

/**
 * Utility class that is responsible for invoking the beginSession and
 * endSession methods on all classes.  This is required to initialize
 * the applet correctly.
 * <p>
 * This is meant to be a temporary solution... we should be removing all
 * statics from the program :(
 * 
 * @author Nick
 */

public class SessionHandler {
	private static boolean hasSession = false;
	
	public synchronized static void beginSession(ServerConnection server) throws SessionAlreadyExists {
		if(hasSession)
			throw new SessionAlreadyExists();

		try {
			ServerConnectionInstance.set(server);
			ParameterTableInstance.set(server.getParameters());

			for(Singleton s : Singleton.getSingletons()) {
				//System.out.println(s.getClass().getName() + " beginSession");
				s.beginSession();
			}
			
			hasSession = true;
		}
		catch(Exception e) {
			nullifySingletons();
			throw new RuntimeException(e);
		}
	}
	
	public synchronized static void endSession() {
		for(Singleton s : Singleton.getSingletons()) {
			//System.out.println(s.getClass().getName() + " endSession");
			s.endSession();
		}
		
		hasSession = false;
	}
	
	private static void nullifySingletons() {
		ImageDatabaseInstance.set(null);
		MainComponentInstance.set(null);
		ServerConnectionInstance.set(null);
		SeriesInstance.set(null);
		ParameterTableInstance.set(null);
	}
}
