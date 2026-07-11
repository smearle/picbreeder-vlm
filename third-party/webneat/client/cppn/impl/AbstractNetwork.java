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

import client.cppn.*;
import client.evolution.*;

abstract class AbstractNetwork implements Network {
	/**
	 * List of neurons for this network. Used in the activation functions.
	 *
	 */
	protected final Neuron[] neurons;
	
	/**
	 * List of neurons for this network. Used in the activation functions.
	 *
	 */
	protected final Connection[] connections;
	
	/**
	 * Represents the input layer for this network.
	 *
	 */
	private final InputLayer inputLayer;
	
	/**
	 * Represents the output layer of the network.
	 *
	 */
	private final OutputLayer[] outputLayers;
	
	/**
	 * Represents the genome that created this network.
	 */
	private final Genome genome;
	
	
	AbstractNetwork(Genome genome) {
		this.genome = genome;
		
		neurons = new Neuron[genome.countNodes()];
		connections = new Connection[genome.countLinks()];

		LinkedList <Neuron> tempNeurons = new LinkedList <Neuron> ();
		
		int i = 0;
		for(Node node : genome.getNodes()) {
			neurons[i] = CPPNFactoryInstance.get().createNeuron(node);
			tempNeurons.add(neurons[i]);
			i++;
		}
		
		Neuron source=null, destination=null;
		i = 0;
		for(Link link: genome.getLinks()) {
			for(Neuron neuron: neurons) {
				if(neuron.getNode().matches(link.getSourceMarking()))
					source = neuron;
				
				if(neuron.getNode().matches(link.getDestinationMarking()))
					destination = neuron;
			}
			
			connections[i++] = CPPNFactoryInstance.get().createConnection(source, destination, link);
		}
		
		inputLayer = CPPNFactoryInstance.get().createInputLayer(tempNeurons);
		
		outputLayers = new OutputLayer[2];
		outputLayers[0] = CPPNFactoryInstance.get().createOutputLayer(tempNeurons, 0);
		outputLayers[1] = CPPNFactoryInstance.get().createOutputLayer(tempNeurons, 1);
	}
	
	public final Genome getGenome() {
		return genome;
	}
	
	public double []evaluate(double x, double y) {
		clearNetwork();
		inputLayer.writeInput(x, y);
		activate();
		return outputLayers[1].readOutput();
	}
	
	public void evaluateAt(double x, double y) {
		clearNetwork();
		inputLayer.writeInput(x, y);
		activate();
	}
	
	public double []readOutput(int i) {
		return outputLayers[i].readOutput();
	}
	
	protected abstract void activate();
	protected abstract void clearNetwork();
}
