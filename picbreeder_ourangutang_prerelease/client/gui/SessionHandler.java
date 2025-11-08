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

import java.lang.reflect.*;
import java.util.Vector;

import client.ImageDatabaseInstance;
import client.MainComponentInstance;
import client.ParameterTableInstance;
import client.server.ServerConnection;
import client.server.ServerConnectionInstance;

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

class SessionHandler {
	private static boolean hasSession = false;
	private static ClassLoader loader = ClassLoader.getSystemClassLoader();

	public static void setClassLoader(ClassLoader classLoader) {
		loader = classLoader;
	}
	
	public synchronized static void beginSession(ServerConnection server) throws SessionAlreadyExists {
		if(hasSession)
			throw new SessionAlreadyExists();

		try {
			ServerConnectionInstance.set(server);
			ParameterTableInstance.set(server.getParameters());
			invokeOnAllClasses("beginSession");
			hasSession = true;
		}
		catch(Exception e) {
			nullifySingletons();
			throw new RuntimeException(e);
		}
	}
	
	public synchronized static void endSession() {
		nullifySingletons();
		
		invokeOnAllClasses("endSession");
		hasSession = false;
	}
	
	private static void nullifySingletons() {
		ImageDatabaseInstance.set(null);
		MainComponentInstance.set(null);
		ServerConnectionInstance.set(null);
		SeriesInstance.set(null);
		ParameterTableInstance.set(null);
	}
	
	private static final Class[] NullTypes = {};
	private static final Object[] NullParameters = {};
	
	private static void invokeOnAllClasses(String method) {
		//System.out.println("attempt to invoke " + method + " on all classes.");
		try {
			Field data = ClassLoader.class.getDeclaredField("classes");
			data.setAccessible(true);
			
			Vector classes = (Vector) data.get(loader);

			//System.out.println("number of classes: " + classes.size());

			for(int i = 0; i < classes.size(); i++) {
				Class c = (Class) classes.elementAt(i);
				//System.out.println("Class name=" + c.getName());

				if(c == SessionHandler.class)
					continue;
				
				try {
					Method m = c.getDeclaredMethod(method, NullTypes);
					m.setAccessible(true);
					
					m.invoke(null, NullParameters);
					
					// DBG out for now
					//System.out.println("Class "+ c.getName() + " invoked " + m.getName());
				}
				catch(NoSuchMethodException e) {
					// class doesn't need it
				}
			}
		}
		catch(InvocationTargetException e) {
			throw new RuntimeException(e.getTargetException());
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}
}
