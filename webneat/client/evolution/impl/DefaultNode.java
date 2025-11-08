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

	/**
	 * The affinity of the gene.
	 */
	private String affinity;
	
	/**
	 * The bias of this node.
	 */
	private double bias = 0.0;
	
	public DefaultNode() {
		super();
		type = "hidden";
		label = "";
		affinity = "grey";
	}
	
	public DefaultNode(Marking marking, String affinity) {
		super(marking);
		type = "hidden";
		label = "";
		this.affinity = affinity;
		
		randomize();
	}
	
	// clone
	DefaultNode(Node node) {
		super(node.getMarking());
		type = node.getType();
		label = node.getLabel();
		activation = node.getActivation();
		affinity = node.getAffinity();
		bias = node.getBias();
	}
	
	// root of a series needs this
	DefaultNode(Marking marking, String name, String type, String affinity) {
		super(marking);
		this.type = type;
		this.label = name;
		this.affinity = affinity;
		randomize();
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
	
	public void setLabel(String label) {
		this.label = label;
	}
	
	public final String getAffinity() {
		return affinity;
	}
	
	public final void setAffinity(String affinity) {
		this.affinity = affinity;
	}
	
	public final double getBias() {
		return bias;
	}
	
	public final void setBias(double bias) {
		this.bias = bias;
	}
	
	public Gene clone() {
		return new DefaultNode(this);
	}
	
	public void randomize() {
		if(type.equals("in"))
			activation = FunctionParser.IDENTITY;
		else
			activation = ParameterTableInstance.get().getRandomItemFromSet("activations");
		
		//bias = client.utilities.Random.instance().nextGaussian();
		bias = 0.0;
	}
	
	public String getElementName() {
		return "node";
	}
	
	public void load(Element xmlElement) {
		super.load(xmlElement);
		type = xmlElement.getAttribute("type");
		activation = XML.parseElementText(xmlElement, "activation");
		
		// note: dtd will enforce the attributes exists when used
		if(xmlElement.hasAttribute("label"))
			label = xmlElement.getAttribute("label");

		if(xmlElement.hasAttribute("affinity"))
			affinity = xmlElement.getAttribute("affinity");
		
		if(xmlElement.hasAttribute("bias"))
			bias = Double.parseDouble(xmlElement.getAttribute("bias"));
	}
	
	public void store(Element xmlElement, Document xmlDocument) {
		super.store(xmlElement, xmlDocument);
		xmlElement.setAttribute("type", type);
		if(hasLabel())
			xmlElement.setAttribute("label", label);
		
		if(affinity.length() > 0)
			xmlElement.setAttribute("affinity", affinity);
		
		xmlElement.setAttribute("bias", Double.toString(bias));
		
		xmlElement.appendChild(XML.createElementText("activation", activation, xmlDocument));
	}
}
