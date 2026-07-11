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

import java.util.*;

import client.cppn.Neuron;
import client.cppn.OutputLayer;

/**
 * 
 * @author Adam Campbell
 */
final class DefaultOutputLayer implements OutputLayer {

	/**
	 * <p>Represents the neuron containing the output value.  It will represent a gray scale value.</p>
	 *
	 */
	DefaultNeuron output;
	
	/**
	 * Represents the actual image output values.
	 */
	double [] values;

	/**
	 * Constructs an output layer out of the list of neurons given. 
	 * The output represents a grey scale value.
	 *
	 * @param neurons List of nodes from which this output layer
	 * 	will obtain the correct output node.
	 */
	public DefaultOutputLayer(Collection<Neuron> neurons){
		output = find(neurons, "ink");
		values = new double[3];
		Arrays.fill(values, 1.0);
	}

	/**
	 * Returns an output neuron with a specific label.  If no neuron in the
	 * collection matches the criteria, then null is returned.
	 *
	 * @param nodes Collection of neurons to be searched.
	 * @param label Label of the sought after neuron.
	 * @return Output neuron with given label.  If none is found, then null
	 * 	is returned.
	 */
	private static DefaultNeuron find(Collection<Neuron> neurons, String label){
		for(Neuron neuron: neurons){
			if(neuron.getNode().getType().equals("out") && neuron.getNode().getLabel().equals(label)){
				return (DefaultNeuron)neuron;
			}
		}		
		return null;
	}
	
	public double[] readOutput(){
		final double out = Math.min(Math.abs(output.getOutput()), 1.0);
		Arrays.fill(values, out);
		return values;
	}
	
	/**
	 *
	 */
	boolean isActive(){
		// TODO FIX ME
		return true;//return output.isActive();
	}
}


