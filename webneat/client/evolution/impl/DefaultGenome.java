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

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import java.util.*;

import client.evolution.*;
import client.utilities.*;

class DefaultGenome implements Genome {
	private ArrayList <Node> nodes;
	private ArrayList <Link> links;
	private long age;
	private static final GeneFinder geneFinder = new GeneFinder();
	private LinkedList <Genome> parents;
	private LinkedList <Identifier> parentIdentifiers;
	private Identifier identifier;
	private int dominantPhenotype = 0;
	
	public DefaultGenome() {
		this(GeneticFactoryInstance.get().createInvalidIdentifier());
	}
	
	private DefaultGenome(Identifier id) {
		nodes = new ArrayList <Node> ();
		links = new ArrayList <Link> ();
		parents = new LinkedList <Genome> ();
		parentIdentifiers = null;
		age = 0;
		identifier = id;
	}
	
	DefaultGenome(Identifier id, Genome g) {
		this(id);
		
		age = g.getAge() + 1;
		dominantPhenotype = g.getDominantPhenotype();
		
		for(Node node : g.getNodes())
			nodes.add(GeneticFactoryInstance.get().copyNode(node));
		
		for(Link link : g.getLinks())
			links.add(GeneticFactoryInstance.get().copyLink(link));	
	}
	
	DefaultGenome(Identifier id, String [] in, String [] out, int hiddenCount) {
		this(id);
		
		Collection <Node> inputs = new ArrayList <Node> ();
		Collection <Node> outputs = new ArrayList <Node> ();
		Collection <Node> hidden = new ArrayList <Node> ();
		
		// TODO better affinity
		
		for(String input : in)
			inputs.add(GeneticFactoryInstance.get().createNode(input, "in", "grey"));
		nodes.addAll(inputs);
		
		for(String output : out)
			outputs.add(GeneticFactoryInstance.get().createNode(output, "out", output.equals("ink") || output.equals("brightness") ? "grey" : "color"));
		nodes.addAll(outputs);
		
		for(int i = 0; i < hiddenCount; i++)
			hidden.add(GeneticFactoryInstance.get().createNode("", "hidden", "grey"));
		nodes.addAll(hidden);
		
		Collections.sort(nodes);
		
		if(hiddenCount == 0)
			for(Node s : inputs)
				for(Node t : outputs)
					links.add(GeneticFactoryInstance.get().createLink(s, t));
		else {
			for(Node s : inputs)
				for(Node t : hidden)
					links.add(GeneticFactoryInstance.get().createLink(s, t));
			
			for(Node s : hidden)
				for(Node t : outputs)
					links.add(GeneticFactoryInstance.get().createLink(s, t));
		}
		
		Collections.sort(links);

		randomize();
	}
	
	public int getDominantPhenotype() {
		return dominantPhenotype;
	}
	
	public void setDominantPhenotype(int index) {
		dominantPhenotype = index;
	}
	
	public Collection <Gene> getGenes() {
		java.util.TreeSet <Gene> s = new java.util.TreeSet <Gene> ();
		s.addAll(nodes);
		s.addAll(links);
		return s;
	}
	
	public Collection <Node> getNodes() {
		return nodes;
	}
	
	public Collection <Link> getLinks() {
		return links;
	}
	
	public int countNodes() {
		return nodes.size();
	}
	
	public int countLinks() {
		return links.size();
	}
	
	public long getAge() {
		return age;
	}
	
	public void overrideAge() {
		age -= 1;
	}
	
	public String getElementName() {
		return "genome";
	}
	
	public void load(Element xmlElement) {
		if(xmlElement.hasAttribute("phenotype"))
			dominantPhenotype = xmlElement.getAttribute("phenotype").equals("color") ? 1 : 0;
		
		identifier.load((Element) xmlElement.getElementsByTagName("identifier").item(0));
		loadNodes((Element) xmlElement.getElementsByTagName("nodes").item(0));
		loadLinks((Element) xmlElement.getElementsByTagName("links").item(0));
		loadParents((Element) xmlElement.getElementsByTagName("parents").item(0));
		age = Long.parseLong(xmlElement.getAttribute("age"));
	}
	
	public void store(Element xmlElement, Document xmlDocument) {
		Element e;
		
		xmlElement.setAttribute("phenotype", dominantPhenotype == 1 ? "color" : "structure");
		
		e = xmlDocument.createElement("identifier");
		identifier.store(e, xmlDocument);
		xmlElement.appendChild(e);
		
		e = xmlDocument.createElement("parents");
		storeParents(e, xmlDocument);
		xmlElement.appendChild(e);
		
		e = xmlDocument.createElement("nodes");
		storeNodes(e, xmlDocument);
		xmlElement.appendChild(e);
		
		e = xmlDocument.createElement("links");
		storeLinks(e, xmlDocument);
		xmlElement.appendChild(e);
		
		xmlElement.setAttribute("age", Long.toString(age));
	}
	
	private void loadNodes(Element xmlElement) {
		nodes.clear();
		
		NodeList list = xmlElement.getElementsByTagName("node");
		for(int i = 0; i < list.getLength(); i++) {
			Node node = GeneticFactoryInstance.get().createInvalidNode();
			node.load((Element) list.item(i));
			GeneticFactoryInstance.get().reserveMarking(node.getMarking());
			nodes.add(node);
		}
		
		Collections.sort(nodes);
	}
	
	private void loadLinks(Element xmlElement) {
		links.clear();
		
		NodeList list = xmlElement.getElementsByTagName("link");
		for(int i = 0; i < list.getLength(); i++) {
			Link link = GeneticFactoryInstance.get().createInvalidLink();
			link.load((Element) list.item(i));
			GeneticFactoryInstance.get().reserveMarking(link.getMarking());
			links.add(link);
		}
		
		Collections.sort(links);
	}

	private void loadParents(Element xmlElement) {
		parentIdentifiers = new LinkedList <Identifier> ();
		
		NodeList list = xmlElement.getElementsByTagName("identifier");
		for(int i = 0; i < list.getLength(); i++) {
			Identifier id = GeneticFactoryInstance.get().createInvalidIdentifier();
			id.load((Element) list.item(i));
			parentIdentifiers.add(id);
		}
	}
	
	private void storeNodes(Element xmlElement, Document xmlDocument) {
		for(Node node : nodes)
			XML.storeElement(node, xmlElement, xmlDocument);

		xmlElement.setAttribute("count", Integer.toString(nodes.size()));
	}
	
	private void storeLinks(Element xmlElement, Document xmlDocument) {
		for(Link link : links)
			XML.storeElement(link, xmlElement, xmlDocument);
		
		xmlElement.setAttribute("count", Integer.toString(links.size()));
	}
	
	private void storeParents(Element xmlElement, Document xmlDocument) {
		for(Genome parent : parents)
			XML.storeElement(parent.getIdentifier(), xmlElement, xmlDocument);
		
		xmlElement.setAttribute("count", Integer.toString(parents.size()));
	}
	
	public void randomize() {
		for(Gene node : nodes)
			node.randomize();
		for(Gene link : links)
			link.randomize();
	}

	public void addLink(Link link) {
		links.add(link);
		if(links.size() >= 2 && link.compareTo(links.get(links.size() - 2)) < 0)
			Collections.sort(links);
	}
	
	public void addNode(Node node) {
		nodes.add(node);
		if(nodes.size() >= 2 && node.compareTo(nodes.get(nodes.size() - 2)) < 0)
			Collections.sort(nodes);
	}
	
	public boolean hasLinkConnecting(Node source, Node destination) {
		for(Link link : getLinks())
			if(link.connects(source.getMarking(), destination.getMarking()))
				return true;
		return false;
	}
	
	public Node getNode(Marking marking) {
		int index = Collections.binarySearch(nodes, marking, geneFinder);
		return index >= 0 ? nodes.get(index) : null;
	}
	
	public Link getLink(Marking marking) {
		int index = Collections.binarySearch(links, marking, geneFinder);
		return index >= 0 ? links.get(index) : null;
	}
	
	public void addParent(Genome parent) {
		parents.add(parent);
	}
	
	public Collection <Genome> getParents() {
		return parents;
	}

	public Collection <Identifier> getParentIdentifiers() {
		return parentIdentifiers;
	}
	
	public Identifier getIdentifier() {
		return identifier;
	}
	
	public boolean isValid() {
		return identifier.isValid();
	}
	
	public int compareTo(Genome other) {
		return identifier.compareTo(other.getIdentifier());
	}
}
