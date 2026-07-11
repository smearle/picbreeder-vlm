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

package test;

import client.server.Initialization;

public class TestInitialization implements Initialization {
	//using the GUI, for testing purposes
	public String getParameter(String name) {
		if(name.equals("seriesId")) return "-1";
		else if(name.equals("parentId")) return "127";
		else if(name.equals("username")) return "secretj";
		else if(name.equals("password")) return "b7a3a59292b5a8206e3448e0dfe2b360";
		else if(name.equals("webservices")) return "http://10.173.214.16:8080/axis/services/WebNeat?wsdl";
		else return null;
	}
}
