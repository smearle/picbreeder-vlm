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

import client.evolution.*;
import client.utilities.Random;
import java.util.ArrayList;

/**
 * Basic implementation.  Does not check for recurrencies or anything.
 * The only gaurantees are that duplicate links are not added and inputs
 * cannot be the destination.
 * 
 */
public class AddLinks extends AbstractMutator {
	public void mutate(Genome offspring) {
		ArrayList <Node> nodes = new ArrayList <Node> ();
		nodes.addAll(offspring.getNodes());
		
		for(Node n : nodes)
			if(n.getType().equals("out"))
				nodes.remove(n);
		
		Node source = nodes.get(Random.instance().nextInt(nodes.size()));
		nodes.clear();
		nodes.addAll(offspring.getNodes());
		nodes.remove(source);
		
		for(Node n : offspring.getNodes())
			if(n.getType().equals("in"))
				nodes.remove(n);
		
		if(nodes.size() > 0) {
			Node destination = nodes.get(Random.instance().nextInt(nodes.size()));
		
			if(!offspring.hasLinkConnecting(source, destination))
				offspring.addLink(GeneticFactoryInstance.get().createLink(source, destination));
		}
	}
}
