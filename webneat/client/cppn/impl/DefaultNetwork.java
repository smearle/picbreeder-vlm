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

package client.cppn.impl;

import client.cppn.*;
import client.evolution.*;

import java.util.*;

/**
 * 
 * @author Adam Campbell
 */

final class DefaultNetwork extends AbstractNetwork {
	/**
	 * Represents the height of this network.  Height is used to determine 
	 * how many times the CPPN should be activated for each pixel.
	 */	
	private final int height;
	
	/**
	 * Constructs a network out of the given genome.  The neurons and connections
	 * are obtained from the genome.
	 *
	 * @param genome Genome from which to create this network.
	 */
	DefaultNetwork(Genome genome){
		super(genome);
		height = computeHeight();
	}
	
	protected void activate() {
		for(int i = 0; i < height; i++)
			activateOnce();
	}

	/**
	 * Computes the height of this network.  The height is used as an indicator of how many times the network
	 * should be activated in order to push the input values to the output values.
	 * <p>
	 * This method will perform a topological sort to calculate the length of the
	 * critical path.  If the topological sort succeeds, then the network is a DAG
	 * (Directed Acyclic Graph).  in this situation, the longest path is known.  If the
	 * network is not a DAG, there are recurrent connections.  In this scenario, we 
	 * simply return the number of neurons in the network.
	 * <p>
	 * This algorithm runs in O(N^2) where N is the number of neurons.
	 */
	private int computeHeight(){
		// use a topological sort to determine the critical path (longest path).
		// if a topological sort cannot be found, there are recursive links
		// and we can return the neurons.length
		
		// this algorithm will reference nodes as indexes into parallel arrays
		
		// stores the incoming degree to a node
		int [] inDegrees = new int[neurons.length];
		
		// stores where we can get to from a neuron
		LinkedList <Integer> [] neighbors = new LinkedList[neurons.length];
		for(int i = 0; i < neurons.length; i++)
			neighbors[i] = new LinkedList <Integer> ();
		
		// maps a neuron to it's index in the arrays
		java.util.HashMap <Neuron, Integer> map = new java.util.HashMap <Neuron, Integer> ();
		
		int i = 0;
		for(Neuron n : neurons)
			map.put(n, i++);

		java.util.Arrays.fill(inDegrees, 0);
		for(Connection c : connections) {
			int dest = map.get(c.getDestination());
			int src = map.get(c.getSource());
			
			// self loop
			if(dest == src)
				return neurons.length;
			
			neighbors[src].add(dest);
			inDegrees[dest]++;
		}
		
		Queue <Integer> q = new LinkedList <Integer> ();
		Queue <Integer> next = new LinkedList <Integer> ();
		
		for(i = 0; i < neurons.length; i++)
			if(inDegrees[i] == 0)
				q.offer(i);
		
		int found = 0;
		int depth = 0;
		while(q.size() > 0) {
			depth++;
			found += q.size();
			
			while(q.size() > 0)
				for(int dest : neighbors[q.poll()])
					if(--inDegrees[dest] == 0)
						next.offer(dest);
			
			// swap
			Queue <Integer> t = q;
			q = next;
			next = t;
			
			// clear garbage from current run
			next.clear();
		}
		
		// it's a DAG and the depth is valid
		if(found == neurons.length)
			return depth;
		// it's not a DAG, return the number of neurons
		else
			return neurons.length;
	}
	
	/**
	 * Activates the network exactly once. Neurons output their values, and connections transmit the weighted values
	 * from their source neurons.
	 *
	 */
	private void activateOnce(){
		for(Connection connection: connections){
			connection.transmit();
		}
		for(Connection connection: connections){
			((DefaultConnection)connection).setDestinationActive();
		}
		for(Neuron neuron: neurons){
			neuron.activate();	
		}
	}

	/**
	 * Sets the input values to all neurons to zero.</p>
	 *
	 */
	protected void clearNetwork(){
		for(Neuron neuron: neurons){
			neuron.clearInput();	
		}
		for(Connection connection: connections){
			// TODO turn off active flag
			((DefaultConnection)connection).clear();
		}
	}
}


