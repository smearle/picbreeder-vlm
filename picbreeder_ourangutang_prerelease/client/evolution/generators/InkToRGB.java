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

public class InkToRGB extends AbstractMutator {
	public void mutate(Genome g) {
		Node ink = null;
		
		for(Node n : g.getNodes())
			if(n.getLabel().equals("ink")) {
				ink = n;
				break;
			}

		Node red = GeneticFactoryInstance.get().createNode("red", "out");
		Node green = GeneticFactoryInstance.get().createNode("green", "out");
		Node blue = GeneticFactoryInstance.get().createNode("blue", "out");
		
		red.setActivation(ink.getActivation());
		green.setActivation(ink.getActivation());
		blue.setActivation(ink.getActivation());
		
		g.addNode(red);
		g.addNode(green);
		g.addNode(blue);
		
		List <Link> remove = new LinkedList <Link> ();
		List <Link> add = new LinkedList <Link> ();
		
		for(Link link : g.getLinks())
			if(ink.matches(link.getDestinationMarking())) {
				Link temp = GeneticFactoryInstance.get().createLink(g.getNode(link.getSourceMarking()), red);
				temp.setWeight(link.getWeight());
				add.add(temp);

				temp = GeneticFactoryInstance.get().createLink(g.getNode(link.getSourceMarking()), green);
				temp.setWeight(link.getWeight());
				add.add(temp);

				temp = GeneticFactoryInstance.get().createLink(g.getNode(link.getSourceMarking()), blue);
				temp.setWeight(link.getWeight());
				add.add(temp);
				
				remove.add(link);
			}
		
		g.getLinks().removeAll(remove);
		g.getLinks().addAll(add);

		Collections.sort((List <Node>) g.getNodes());
		Collections.sort((List <Link>) g.getLinks());
	}
}
