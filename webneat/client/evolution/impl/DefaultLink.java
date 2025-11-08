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

import client.evolution.AbstractGene;
import client.evolution.Link;
import client.evolution.Marking;
import client.evolution.GeneticFactoryInstance;
import client.utilities.*;
import client.*;

/**
 * 
 * @author Nick Beato
 */
class DefaultLink extends AbstractGene implements Link {
	protected Marking sourceMarking, destinationMarking;
	protected double weight;
	
	private final double MAX_WEIGHT;
	
	public DefaultLink() {
		super();
		sourceMarking = GeneticFactoryInstance.get().createInvalidMarking();
		destinationMarking = GeneticFactoryInstance.get().createInvalidMarking();
		MAX_WEIGHT = ParameterTableInstance.get().getDouble("evolution", "max link weight");
		weight = 0.0;
	}
	
	public DefaultLink(Marking marking, Marking from, Marking to) {
		super(marking);
		sourceMarking =	from;
		destinationMarking = to;
		MAX_WEIGHT = ParameterTableInstance.get().getDouble("evolution", "max link weight");
		randomize();
	}

	// clone
	DefaultLink(Link link) {
		super(link.getMarking());
		sourceMarking =	link.getSourceMarking();
		destinationMarking = link.getDestinationMarking();
		MAX_WEIGHT = ParameterTableInstance.get().getDouble("evolution", "max link weight");
		weight = link.getWeight();
	}
	
	public Marking getSourceMarking() {
		return sourceMarking;
	}
	
	public Marking getDestinationMarking() {
		return destinationMarking;
	}
	
	public double getWeight() {
		return weight;
	}
	
	public void setWeight(double weight) {
		this.weight = weight;
	}
	
	public boolean connects(Marking source, Marking destination) {
		return source.equals(sourceMarking) && destination.equals(destinationMarking);
	}
	
	public void randomize() {
		weight = (Random.instance().nextDouble() * 2.0 - 1.0) * MAX_WEIGHT;
	}
	
	public String getElementName() {
		return "link";
	}
	
	public void load(Element xmlElement) {
		super.load(xmlElement);
		sourceMarking.load((Element) xmlElement.getElementsByTagName("source").item(0));
		destinationMarking.load((Element) xmlElement.getElementsByTagName("target").item(0));
		weight = Double.parseDouble(XML.parseElementText(xmlElement, "weight"));
	}
	
	public void store(Element xmlElement, Document xmlDocument) {
		super.store(xmlElement, xmlDocument);
		
		Element source = xmlDocument.createElement("source");
		sourceMarking.store(source, xmlDocument);
		xmlElement.appendChild(source);
		
		Element destination = xmlDocument.createElement("target");
		destinationMarking.store(destination, xmlDocument);
		xmlElement.appendChild(destination);
		
		xmlElement.appendChild(XML.createElementText("weight", Double.toString(weight), xmlDocument));
	}
}
