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
import java.util.List;

/**
 * 
 * 
 */
public class AddNodes extends AbstractMutator {	
	/**
	 * Randomly selects a link and splits it.  The original link is preserved.
	 * 
	 * @param offspring The genome to mutate
	 */
	public void mutate(Genome offspring) {
		List <Link> links = (List <Link>) offspring.getLinks();
		Link link = links.get(Random.instance().nextInt(offspring.countLinks()));
		
		Node source = offspring.getNode(link.getSourceMarking());
		Node destination = offspring.getNode(link.getDestinationMarking());
		
		String affinity = "";
		if(source.getType().equals("out"))
			affinity = destination.getAffinity();
		else
			affinity = Random.instance().nextBoolean() ? source.getAffinity() : destination.getAffinity();
		
		Node node = GeneticFactoryInstance.get().createNode(link, affinity);
		Link toNode = GeneticFactoryInstance.get().createLink(source, node);
		Link fromNode = GeneticFactoryInstance.get().createLink(node, destination);
		
		offspring.addNode(node);
		offspring.addLink(toNode); // note, order on link insertion matters!
		offspring.addLink(fromNode); // insert in increasing order of markings
	}
 }
