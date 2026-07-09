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

package client.tools;

/**
 * Runs the <code>public static void unitTest()</code>
 * method on all compiled classes, logging exceptions that occur.
 * <p>
 * Simple messages indicating progress will write to standard out.
 * All failures will be written to standard error. This is to
 * allow the user to pipe the failed unit tests to a file for
 * easier debug.
 * 
 * @author Nick
 *
 */

import java.io.*;
import java.lang.reflect.*;

public class UnitTester {	
	public static void main(String []args) throws Exception {
		// some classes get the static initializer called, causing them
		// to poll the parameters
		client.ParameterTableInstance.set(new test.TestParameters());
		
		String classPath = System.getProperty("java.class.path", ".");
		String []dirs = classPath.split(";");
		
		for(String dir : dirs)
			recurse(new File(dir), "");
	}
	
	private static void recurse(File file, String path) {
		if(file.isDirectory())
			for(File f : file.listFiles())
				recurse(f, path + f.getName() + ".");
		else if(file.getName().endsWith(".class"))
			process(path.substring(0, path.length() - 7));
	}
	
	private static void process(String className) {
		try {
			Class c = Class.forName(className);
			Method m = c.getDeclaredMethod("unitTest", ARG_TYPES);
			m.setAccessible(true);
			
			System.out.print("UnitTest " + c.getName() + "... ");
			m.invoke(null, ARGS);
			System.out.println("Success");
			
		}
		catch(InvocationTargetException e) {
			System.out.println("Failure");
			e.getTargetException().printStackTrace(System.err);
		}
		catch(NoSuchMethodException e) {
			// ok
		}
		catch(Exception e) {
			System.out.println("Failure");
			e.printStackTrace(System.err);
		}
	}
	
	private static final Object [] ARGS = {};
	private static final Class [] ARG_TYPES = {};
}
