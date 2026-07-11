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

import client.utilities.*;

/**
 * 
 * 
 */
public class Validator {
	public static void main(String []args) throws Exception {
		ArgumentParser parser = new ArgumentParser(args);
		
		String fileName = parser.findArgument("-f");
		String dtdFolder = parser.findArgument("-d");
		
		System.out.println("Validating " + fileName + " with dtds in " + dtdFolder);
		
		if(XML.validate(fileName, dtdFolder) == null) {
			System.out.println("Failure!");
			System.exit(-1);
		}
		else
			System.out.println("Success!");
	}
}
