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
	public void mutate(Genome g) {
		Node ink = null;
		
		for(Node n : g.getNodes())
			if(n.getLabel().equals("ink")) {
				ink = n;
				break;
			}

		Node hue = GeneticFactoryInstance.get().createNode("hue", "out");
		Node saturation = GeneticFactoryInstance.get().createNode("saturation", "out");
		Node brightness = GeneticFactoryInstance.get().createNode("brightness", "out");
		
		hue.randomize();
		saturation.setActivation("sigmoid(x)");//TODO better impl
		brightness.setActivation(ink.getActivation());
		
		g.addNode(hue);
		g.addNode(saturation);
		g.addNode(brightness);
		
		List <Link> remove = new LinkedList <Link> ();
		List <Link> add = new LinkedList <Link> ();
		
		for(Link link : g.getLinks())
			if(ink.matches(link.getDestinationMarking())) {
				Link temp = GeneticFactoryInstance.get().createLink(g.getNode(link.getSourceMarking()), brightness);
				temp.setWeight(link.getWeight());
				add.add(temp);
				remove.add(link);
			}
		
		g.getLinks().removeAll(remove);
		g.getLinks().addAll(add);
		
		addRandomLinks(g, hue, 5);
		addRandomLinks(g, saturation, 5);

		Collections.sort((List <Node>) g.getNodes());
		Collections.sort((List <Link>) g.getLinks());
	}
	
	private void addRandomLinks(Genome g, Node dest, int howMany) {
		ArrayList <Node> nodes = new ArrayList <Node> ();
		nodes.addAll(g.getNodes());
		for(Node n : g.getNodes())
			if(n.getType().equals("out"))
				nodes.remove(n);
		
		for(int i = 0; i < howMany && nodes.size() > 0; i++) {
			Node src = nodes.get(client.utilities.Random.instance().nextInt(nodes.size()));
			Link link = GeneticFactoryInstance.get().createLink(src, dest);
			link.setWeight(0.0);
			g.getLinks().add(link);
		}
	}
}
