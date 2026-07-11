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

package client.evolution.impl;

import org.w3c.dom.*;
import client.Transferable;
import client.evolution.*;
import client.server.FatalException;
import client.server.TimeoutException;

class GenerationImposter implements Transferable {
	private Generation generation;
	private String storageName;
	private int number;
	private boolean matchesServer;
	
	public GenerationImposter(Generation g) {
		String name = DatabaseInstance.get().addGeneration(g);
		
		generation = g;
		storageName = name;
		number = g.getNumber();
		matchesServer = false;
	}
	
	public GenerationImposter() {
		generation = null;
		storageName = "";
		number = -1;
		matchesServer = true;
	}
	
	public void ensureLoaded() throws FatalException, TimeoutException {
		generation = DatabaseInstance.get().retrieveGeneration(storageName, number);
	}
	
	public Generation getGeneration() throws FatalException, TimeoutException {
		if(!isLoaded())
			ensureLoaded();
		
		return generation;
	}

	public void prefetch() {
		if(!isLoaded())
			DatabaseInstance.get().prefetchGeneration(storageName, number);
	}
	
	public boolean isSyncedWithServer() {
		return matchesServer;
	}
	
	public void notifySuccessfulSave() {
		matchesServer = true;
		DatabaseInstance.get().ensureServerContains(storageName);
	}
	
	public boolean isLoaded() {
		return generation != null;
	}
	
	public String getElementName() {
		return "generation";
	}
	
	public void load(Element xmlElement) {
		number = Integer.parseInt(xmlElement.getAttribute("number"));
		storageName = xmlElement.getAttribute("storage");
		matchesServer = true;
		
		DatabaseInstance.get().ensureServerContains(storageName);
	}
	
	public void store(Element xmlElement, Document xmlDocument) {
		if(number < 0)
			number = generation.getNumber();
		
		xmlElement.setAttribute("number", Integer.toString(number));
		xmlElement.setAttribute("storage", storageName);
	}
}
