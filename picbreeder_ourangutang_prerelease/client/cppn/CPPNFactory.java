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

package client.cppn;

import client.evolution.*;
import java.util.*;

/**
 * The CPPNFactory is responsible for creating all cppn objects. This
 * hides the CPPN implementation from the GUI.
 * 
 * @author Adam Campbell
 * @author Nick Beato
 */
public interface CPPNFactory {
	/**
	 * Creates a default network with the given genome.
	 * 
	 * @param genome Genome from which the network should be created.
	 * @return New network.
	 */
	public Network createNetwork(Genome genome);

	/**
	 * Creates a default neuron out of the given node.
	 * 
	 * @param node Node that the neuron is created out of.
	 * @return New neuron.
	 */
	public Neuron createNeuron(Node node);

	/**
	 * Creates a new connection.
	 * 
	 * @param source The source neuron.
	 * @param destination The destination neuron.
	 * @param link Contains the weight for this connection.
	 * @return New connection.
	 */
	public Connection createConnection(Neuron source, Neuron destination, Link link);

	/**
	 * Creates new input layer with the given collection of neurons.
	 * 
	 * @param neurons List of neurons from which the input neuron layer will be created.
	 * @return New input layer.
	 */
	public InputLayer createInputLayer(Collection <Neuron> neurons);

	/**
	 * <p>Creates new output layer with the given collection of neurons.</p>
	 * 
	 * 
	 * @param neurons List of neurons from which the output neuron layer will be created.
	 * @return New output layer.
	 */
	public OutputLayer createOutputLayer(Collection <Neuron> neurons);
 }
