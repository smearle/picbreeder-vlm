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
public class AddNodeWithinAffinity extends AbstractMutator {
	private final String affinity;
	
	public AddNodeWithinAffinity(String affinity) {
		this.affinity = affinity;
	}
	
	/**
	 * Randomly selects a link and splits it.  The original link is preserved.
	 * 
	 * @param offspring The genome to mutate
	 */
	public void mutate(Genome offspring) {
		List <Link> links = new java.util.ArrayList <Link> ();
		
		for(Link link : offspring.getLinks())
			if(offspring.getNode(link.getDestinationMarking()).getAffinity().equals(affinity))
				links.add(link);
		
		// no links can occur when a grey network becomes a color network
		// and 0 hidden neurons are added to the color affinity
		if(links.size() == 0)
			return;
		
		Link link = links.get(Random.instance().nextInt(links.size()));
		
		Node source = offspring.getNode(link.getSourceMarking());
		Node destination = offspring.getNode(link.getDestinationMarking());
		
		Node node = GeneticFactoryInstance.get().createNode(link, affinity);
		Link toNode = GeneticFactoryInstance.get().createLink(source, node);
		Link fromNode = GeneticFactoryInstance.get().createLink(node, destination);
		
		offspring.addNode(node);
		offspring.addLink(toNode); // note, order on link insertion matters!
		offspring.addLink(fromNode); // insert in increasing order of markings
	}
 }
