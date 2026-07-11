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

package client.evolution.generators;

import java.util.*;

import client.evolution.*;

public class InkToHSB extends AbstractMutator {
	private int hiddenColorNodes;
	
	public InkToHSB() {
		hiddenColorNodes = Integer.parseInt( client.ParameterTableInstance.get().getParameter("evolution", "hidden color nodes"));
	}
	
	public void mutate(Genome g) {
		g.overrideAge();
		
		Node brightness = null;
		
		for(Node n : g.getNodes())
			if(n.getLabel().equals("ink")) {
				brightness = n;
				break;
			}
		
		// already done!
		if(brightness == null)
			return;
		
		for(Node n : g.getNodes())
			n.setAffinity("grey");

		Node hue = GeneticFactoryInstance.get().createNode("hue", "out", "color");
		Node saturation = GeneticFactoryInstance.get().createNode("saturation", "out", "color");
		brightness.setLabel("brightness");
		
		hue.setActivation("sin(x)");
		saturation.setActivation("sigmoid(x)");
		
		g.addNode(hue);
		g.addNode(saturation);
		
		g.getLinks().add(GeneticFactoryInstance.get().createLink(brightness, hue));
		g.getLinks().add(GeneticFactoryInstance.get().createLink(brightness, saturation));
		
		addColorNetwork(g);
	}
	
	private void addColorNetwork(Genome g) {
		Collection <Node> inputs = new ArrayList <Node> ();
		Collection <Node> outputs = new ArrayList <Node> ();
		Collection <Node> hidden = new ArrayList <Node> ();
		
		for(Node n : g.getNodes())
			if(n.getType().equals("in") || n.getLabel().equals("brightness"))
				inputs.add(n);

		for(Node n : g.getNodes())
			if(n.getType().equals("out") && !n.getLabel().equals("brightness"))
				outputs.add(n);
	
		
		for(int i = 0; i < hiddenColorNodes; i++)
			hidden.add(GeneticFactoryInstance.get().createNode("", "hidden", "color"));
		
		g.getNodes().addAll(hidden);
		Collections.sort((List <Node>) g.getNodes());
		
		// add links
		Collection <Link> links = new ArrayList <Link> ();
		
		if(hiddenColorNodes == 0)
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
		
		// force color links to be have weight 0 initially to stop 
		// rainbowing effects in colored network
		for(Link link : links)
			if(g.getNode(link.getSourceMarking()).getAffinity().equals("color"))
				link.setWeight(0.0);
		
		g.getLinks().addAll(links);
		Collections.sort((List <Link>) g.getLinks());
	}
}
