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
import client.utilities.Random;

public class AddAcyclicLink extends AbstractMutator {
	/**
	 * Mutate the offspring by adding a link.  The new link will not cause
	 * a recurrence in the offspring.  This implementation attempts to
	 * accomplish this using random starting point.  If that choice results
	 * in no possible links, the method will not add a new link. 
	 */
	public void mutate(Genome offspring) {
		// create a copy of the node set
		ArrayList <Node> nodes = new ArrayList <Node> ();
		nodes.addAll(offspring.getNodes());
		
		// arbitrarily pick a source node
		// do note allow the outputs as sources
		LinkedList <Node> outputs = new LinkedList <Node> ();
		for(Node n : nodes)
			if(n.getType().equals("out"))
				outputs.add(n);
		
		nodes.removeAll(outputs);

		if(nodes.size() == 0)
			return;
		
		Node source = nodes.get(Random.instance().nextInt(nodes.size()));
		nodes.addAll(outputs);
	
		// remove everything that precedes the source
		depthFirstTraversal(offspring, source, nodes);

		// don't flow into an input, that makes no sense
		LinkedList <Node> inputs = new LinkedList <Node> ();
		for(Node n : nodes)
			if(n.getType().equals("in"))
				inputs.add(n);
		
		nodes.removeAll(inputs);
		
		// remove the neighbors so we don't try duplicating a connection
		LinkedList <Node> neighbors = new LinkedList <Node> ();
		for(Node n : nodes)
			if(offspring.hasLinkConnecting(source, n))
				neighbors.add(n);

		nodes.removeAll(neighbors);

		// if we still have choices, add a link!
		if(nodes.size() > 0) {
			Node destination = nodes.get(Random.instance().nextInt(nodes.size()));
			offspring.addLink(GeneticFactoryInstance.get().createLink(source, destination));
		}
	}
	
	/**
	 * Traverses against the activation direction to find all nodes leading 
	 * from the inputs to the node at the specified position. This method will
	 * will remove any node from <code>remainingNodes</code> that
	 * will be activated prior to the node named <code>position</code>.  If
	 * <code>remainingNodes.contains(position) == false</code>, this method
	 * will immediately return.
	 * <p>
	 * This is a modification of a depth-first search.  It's run time is
	 * slightly worse since we do not have access to the the neighbors of
	 * the position node. This will run in O(N*M), where N is the number of 
	 * nodes and M is the number of links.
	 * 
	 * @param genome The genome to search
	 * @param position The node that is being considered
	 * @param remainingNodes The set of remaining nodes 
	 * 	<b>(do NOT pass genome.getNodes())</b>
	 */
	private void depthFirstTraversal(Genome genome, Node position, Collection <Node> remainingNodes) {
		if(!remainingNodes.contains(position))
			return;
		
		remainingNodes.remove(position);
		
		for(Link link : genome.getLinks())
			if(position.matches(link.getDestinationMarking()))
				depthFirstTraversal(genome, genome.getNode(link.getSourceMarking()), remainingNodes);
	}
}
