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

import java.util.LinkedList;
import java.util.Queue;

import client.cppn.*;
import client.evolution.*;

/**
 * An acyclic CPPN can be constructed and activated in linear time
 * with respect to edges.  This is considerably faster than a
 * "normal" (recursive) CPPN.
 * <p>
 * During construction, the connections are sorted according to
 * a topological sort.  If the topological sort succeeds, the
 * network is instantiated.  Otherwise, the constructor will
 * throw an exception, indicating that the genome has recursive
 * links.
 * 
 * @author Nick Beato
 *
 */
final class AcyclicCPPN extends AbstractNetwork {
	AcyclicCPPN(Genome genome) throws CreationFailedException {
		super(genome);
		topologicalSort();
	}
	
	public void activate() {
		// reset the input nodes to inactive
		for(Neuron n : neurons)
			((DefaultNeuron) n).setActive(false);
		
		// run activation
		// use the active flag to determine whether the
		// activation function already fired
		for(Connection c : connections) {
			Neuron src = c.getSource();
			
			if(!src.isActive()) {
				((DefaultNeuron)src).setActive(true);
				src.activate();
			}
			
			c.transmit();
		}
		
		// because only source nodes were activated,
		// the output neurons are not ready.
		// activate the output layer
		for(Neuron n : neurons)
			if(!n.isActive()) {
				((DefaultNeuron)n).setActive(true);
				n.activate();
			}
	}
	
	public void clearNetwork() {
		for(Neuron n : neurons)
			n.clearInput();
	}

	/**
	 * Sorts the connections in a topological order so that activation
	 * can simply iterate over the edges, dropping the activation time
	 * to linear in the number of edges.
	 * <p>
	 * This algorithm runs in O(N^2) where N is the number of neurons.
	 */
	private void topologicalSort() throws CreationFailedException {
		// this algorithm will reference nodes as indexes into parallel arrays
		
		// stores the incoming degree to a node
		int [] inDegrees = new int[neurons.length];
		
		// stores where we can get to from a neuron
		// java note: to create an array of a generic type, you have to drop
		// the generic syntax. type-checking still occurs everywhere else
		LinkedList <Connection> [] neighbors = new LinkedList[neurons.length];
		for(int i = 0; i < neurons.length; i++)
			neighbors[i] = new LinkedList <Connection> ();
		
		// maps a neuron to it's index in the arrays
		java.util.HashMap <Neuron, Integer> map = new java.util.HashMap <Neuron, Integer> ();
		
		// create the said mapping
		int i = 0;
		for(Neuron n : neurons)
			map.put(n, i++);

		// calculate the incoming degree to each neuron
		// and also build the neighborhood
		java.util.Arrays.fill(inDegrees, 0);
		for(Connection c : connections) {
			int dest = map.get(c.getDestination());
			int src = map.get(c.getSource());
			
			// self loop
			if(dest == src)
				throw new CreationFailedException();
			
			neighbors[src].add(c);
			inDegrees[dest]++;
		}
		
		// use a queue to process the the nuerons with 0
		// incoming degree
		Queue <Integer> q = new LinkedList <Integer> ();
		
		// find the input layer (initial nodes)
		for(i = 0; i < neurons.length; i++)
			if(inDegrees[i] == 0)
				q.offer(i);
		
		// the output (topologically sorted edges)
		Connection []sorted = new Connection[connections.length];

		i = 0;
		// found indicates how many nodes were reached.
		// this will equal the number of neurons on success
		int found = q.size();
		while(q.size() > 0) {
			// add all edges coming from the next neuron
			// in order
			for(Connection c : neighbors[q.poll()]) {
				sorted[i++] = c;
				int dest = map.get(c.getDestination());
				
				// decrement the incoming degree of the target
				if(--inDegrees[dest] == 0) {
					found++;
					q.offer(dest);
				}
			}
		}

		// it's not a DAG, return the number of neurons
		if(found != neurons.length)
			throw new CreationFailedException();
		
		System.arraycopy(sorted, 0, connections, 0, connections.length);
	}
}
