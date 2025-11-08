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
import client.utilities.XML;
import client.*;
import client.evolution.AbstractGene;
import client.evolution.Gene;
import client.evolution.Marking;
import client.evolution.Node;
import client.math.FunctionParser;

/**
 * 
 * @author Nick Beato
 */
class DefaultNode extends AbstractGene implements Node {
	protected String activation = FunctionParser.IDENTITY;

	private String type;
	private String label;
	
	public DefaultNode() {
		super();
		type = "hidden";
		label = "";
	}
	
	public DefaultNode(Marking marking) {
		super(marking);
		type = "hidden";
		label = "";
		activation = ParameterTableInstance.get().getRandomItemFromSet("activations");
	}
	
	// clone
	DefaultNode(Node node) {
		super(node.getMarking());
		type = node.getType();
		label = node.getLabel();
		activation = node.getActivation();
	}
	
	// root of a series needs this
	DefaultNode(Marking marking, String name, String type) {
		super(marking);
		this.type = type;
		label = name;
	}
	
	public final String getActivation() {
		return activation;
	}
	
	public final void setActivation(String a) {
		if(!type.equals("in"))
			activation = a;
	}
	
	public final String getType() {
		return type;
	}
	
	public String getLabel() {
		return label;
	}
	
	public boolean hasLabel() {
		return label != null && label.length() > 0;
	}
	
	public Gene clone() {
		return new DefaultNode(this);
	}
	
	public void randomize() {
		if(type.equals("in"))
			activation = FunctionParser.IDENTITY;
		else
			activation = ParameterTableInstance.get().getRandomItemFromSet("activations");
	}
	
	public String getElementName() {
		return "node";
	}
	
	public void load(Element xmlElement) {
		super.load(xmlElement);
		type = xmlElement.getAttribute("type");
		activation = XML.parseElementText(xmlElement, "activation");
		
		// note: dtd will enforce the attributes exists
		if(xmlElement.hasAttribute("label"))
			label = xmlElement.getAttribute("label");
	}
	
	public void store(Element xmlElement, Document xmlDocument) {
		super.store(xmlElement, xmlDocument);
		xmlElement.setAttribute("type", type);
		if(hasLabel())
			xmlElement.setAttribute("label", label);
		xmlElement.appendChild(XML.createElementText("activation", activation, xmlDocument));
	}
}
