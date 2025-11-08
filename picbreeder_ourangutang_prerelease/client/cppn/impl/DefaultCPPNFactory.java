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
 * Creates all objects in the CPPN package. This is used so that
 * the implmentation may change without rewriting any code in the package.
 * 
 */
public class DefaultCPPNFactory implements CPPNFactory {
	/**
	 * This function is not yet implemented and should not be called.  Eventually, we may
	 * add the ability to create a default, simple network.
	 * 
	 * @return Network New network.
	 */
	public Network createNetwork() {		
		return null;
	} 

	public Network createNetwork(Genome genome) {
		try {
			return new AcyclicCPPN(genome);
		}
		catch(CreationFailedException e) {
			return new DefaultNetwork(genome);
		}
	}
	
	public Neuron createNeuron(Node node) {		
		return new DefaultNeuron(node);
	} 

	public Connection createConnection(Neuron from, Neuron to, Link link) {		
		return new DefaultConnection(from, to, link);
	} 

	public InputLayer createInputLayer(Collection<Neuron> neurons) {		
		return new DefaultInputLayer(neurons);
	} 

	public OutputLayer createOutputLayer(Collection<Neuron> neurons) {		
		//return new DefaultOutputLayer(neurons);
		return new HSBOutputLayer(neurons);
		//return new RGBOutputLayer(neurons);
	}
}
