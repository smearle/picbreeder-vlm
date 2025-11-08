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
import java.util.*;

/**
 * SimplifyGenome does just that... it simplifies a genome. The mutation operation
 * essentially destroys all genes that do not contribute to the phenotype. This is
 * potentially disastrous. Unnescessary genes may be useful (as in nature) to
 * gaurantee that a mutation does not completely destroy an individual.
 * However, this code may be used to clean house on genomes every now and then,
 * requiring less storage and computation time. It's a trade off...
 * 
 * @author Nick Beato
 *
 */
public class SimplifyGenome extends AbstractMutator implements Comparator <Link> {
	private final double TOLERANCE;
	
	/**
	 * Constructs a generator that will prune genes off of a genome that
	 * do not affect the phenotype. This is potentially very destructive,
	 * since it means that all mutations will occur on meaningful genes!
	 */
	public SimplifyGenome() {
		// TODO parameterize this
		TOLERANCE = 1e-3;
	}
	
	public void mutate(Genome g) {
		LinkedList <Link> links = new LinkedList <Link> ();
		links.addAll(g.getLinks());
		
		// sort the links by weight (absolute value)
		// so the weights closer to zero occur earlier
		Collections.sort(links, this);
		
		for(Link link : links) 
			if(Math.abs(link.getWeight()) > TOLERANCE) // done!
				return;
			else if(g.getLinks().contains(link)) // might have been removed already
				pruneGenome(g, link);
		
	}
	
	private void pruneGenome(Genome g, Link removedLink) {
		// remove the section of the graph that is no longer reachable
		// by performing 2 BFS's. one starts from the inputs and
		// goes to the outputs. the other starts from the outputs
		// and goes to the inputs.
		// the link is only removed if all inputs reach some
		// output (and vice versa)
		
		// work with a local copy of the links and nodes
		Collection <Link> links = new LinkedList <Link> ();
		links.addAll(g.getLinks());
		links.remove(removedLink);

		Collection <Node> nodes = g.getNodes();
		
		// find the inputs and outputs
		Collection <Node> inputs = new LinkedList <Node> ();
		for(Node n : nodes)
			if(n.getType().equals("in"))
				inputs.add(n);
		
		Collection <Node> outputs = new LinkedList <Node> ();
		for(Node n : nodes)
			if(n.getType().equals("out"))
				outputs.add(n);
		
		// neighbors of a node (u is a neighbor of v if uv is a link)
		// construct the neighbors to lower the algorithm to O(N^2) average
		Map <Node, Collection <Link> > neighborsAgainst = new TreeMap <Node, Collection <Link> > ();
		Map <Node, Collection <Link> > neighborsWith = new TreeMap <Node, Collection <Link> > ();
		
		for(Node n : nodes) {
			neighborsAgainst.put(n, new LinkedList <Link> ());
			neighborsWith.put(n, new LinkedList <Link> ());
		}
		
		for(Link link : links) {
			neighborsAgainst.get(g.getNode(link.getDestinationMarking())).add(link);
			neighborsWith.get(g.getNode(link.getSourceMarking())).add(link);
		}
		
		// run a bfs from the outputs to the inputs. keep track of all
		// nodes and links traversed
		Collection <Link> retainLinksAgainst = new LinkedList <Link> ();
		Collection <Node> retainNodesAgainst = new LinkedList <Node> ();
		
		breadthFirstTraversal(g, outputs, neighborsAgainst, retainNodesAgainst, retainLinksAgainst);
		
		// gaurantee that all of the inputs have been reached
		// if not, we disconnected an input and should not
		// perform pruning, as it is easier to mutate the link
		// then try to reconnect it
		for(Node n : inputs)
			if(!retainNodesAgainst.contains(n))
				return;

		// runs a bfs from the inputs to the outsputs. keep track of all
		// nodes and links traversed
		Collection <Link> retainLinksWith = new LinkedList <Link> ();
		Collection <Node> retainNodesWith = new LinkedList <Node> ();
		
		breadthFirstTraversal(g, inputs, neighborsWith, retainNodesWith, retainLinksWith);
		
		// gaurantee that all of the outputs have been reached
		// if not, we disconnected an output and should not
		// perform pruning, as it is easier to mutate the link
		// then try to reconnect it
		for(Node n : outputs)
			if(!retainNodesWith.contains(n))
				return;
		
		// if we get here, we can remove all nodes and links that
		// were not reached by BOTH bfs operations
		g.getNodes().retainAll(retainNodesWith);
		g.getNodes().retainAll(retainNodesAgainst);
		g.getLinks().retainAll(retainLinksWith);
		g.getLinks().retainAll(retainLinksAgainst);
	}
	
	private void breadthFirstTraversal(Genome g, Collection <Node> initial, Map <Node, Collection <Link> > neighbors, Collection <Node> visitedNodes, Collection <Link> visitedLinks) {
		// BFS code, add the things that are traversed to the retain sets
		Queue <Node> q = new LinkedList <Node> ();
		q.addAll(initial);
		
		while(q.size() > 0) {
			Node n = q.poll();
			visitedNodes.add(n);
			
			for(Link link : neighbors.get(n)) {
				visitedLinks.add(link);
				
				Marking marking = link.getSourceMarking();
				if(n.matches(marking))
					marking = link.getDestinationMarking();
				
				Node other = g.getNode(marking);
				if(!visitedNodes.contains(other))
					q.offer(other);
			}
		}
	}
	
	/**
	 * Compares links a and b so that links may be
	 * sorted in order of increasing absolute value.
	 * 
	 * @param a One link
	 * @param b The other link
	 * @return -1 if a < b, 0 if a == b, and 1 if a > b
	 */
	public int compare(Link a, Link b) {
		double diff = Math.abs(a.getWeight()) - Math.abs(b.getWeight()); 
		if(diff < 0)
			return -1;
		else if(diff > 0)
			return 1;
		else
			return 0;
	}
}
