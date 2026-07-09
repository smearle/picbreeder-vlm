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

import client.evolution.*;
import org.w3c.dom.*;
import java.util.ArrayList;

// assumes the generations are ordered contiguously
// ie, 12, 13, 14, 15 may be in this file

class DefaultStorage implements Storage {
	private final ArrayList <Generation> data;
	private final String name;
	
	DefaultStorage(String name) {
		data = new ArrayList <Generation> ();
		this.name = name;
	}
	
	public int getMinimum() {
		return data.get(0).getNumber();
	}
	
	public int getMaximum() {
		return data.get(data.size()-1).getNumber();
	}
	
	public String getName() {
		return name;
	}
	
	public int size() {
		return data.size();
	}
	
	public java.util.Iterator <Generation> iterator() {
		return data.iterator();
	}
	
	public void addGeneration(Generation g) {
		for(int i = 0; i < data.size(); i++)
			if(data.get(i).getNumber() == g.getNumber()) {
				data.set(i, g);
				return;
			}
		
		data.add(g);
	}
	
	public void removeGeneration(Generation g) {
		for(int i = 0; i < data.size(); i++)
			if(data.get(i).getNumber() == g.getNumber()) {
				data.set(i, null); // set to null to help the garbage collector
				data.remove(i);
				return;
			}
	}
	
	public boolean isEmpty() {
		return data.size() == 0;
	}
	
	public Generation getGeneration(int age) {
		for(Generation g : data)
			if(g.getNumber() == age)
					return g;
		return null;
	}
	
	public String getElementName() {
		return "storage";
	}
	
	public void load(Element xmlElement) {
		data.clear();
		
		NodeList nodes = xmlElement.getElementsByTagName("generation");
		
		for(int i = 0; i < nodes.getLength(); i++) {
			Generation g = GeneticFactoryInstance.get().createInvalidGeneration();
			g.load((Element) nodes.item(i));
			data.add(g);
		}
	}
	
	public void store(Element xmlElement, Document xmlDocument) {
		xmlElement.setAttribute("name", name);
		xmlElement.setAttribute("minimum", Integer.toString(getMinimum()));
		xmlElement.setAttribute("maximum", Integer.toString(getMaximum()));
		
		for(Generation g : data) {
			Element e = xmlDocument.createElement(g.getElementName());
			g.store(e, xmlDocument);
			xmlElement.appendChild(e);
		}
	}
}
