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

import client.server.Initialization;

public class RendererInitialization implements Initialization {

	public int gen;
	public RendererInitialization(int genome)
	{
		gen=genome;
	}
	
	public String getParameter(String parameterName) {
		if(parameterName.equals("seriesId")) return "-1";
		else if(parameterName.equals("parentId")) return Integer.toString(gen);
		else if(parameterName.equals("username")) return "secretj";
		else if(parameterName.equals("password")) return "b7a3a59292b5a8206e3448e0dfe2b360";
		else if(parameterName.equals("webservices")) return "http://picbreeder.org:8080/axis/services/WebNeat?wsdl";
		else return null;
	}

}
