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

import client.evolution.Node;

/**
 * A neuron represents a neuron in the CPPN.  See neural network
 * literatue if this doesn't make sense.
 * 
 * @author Adam Campbell
 */

public interface Neuron {
	/**
	 * Activates this neuron by setting its output value to the value of its
	 * activation function applied to the sum of its inputs.
	 */
	public void activate();

	/**
	 * Returns boolean indicating whether or not this neuron is active. A neuron
	 * becomes active when its setInput or addInput functions are called.
	 *
	 * @return Indication of whether or not this neuron is active.
	 */
	public boolean isActive();

	/**
	 * Sets the input of this neuron to x.  This function should only be called
	 * in the InputLayer.
	 *
	 * @param x Value the input is set to.
	 */
	public void setInput(double x);
	
	/**
	 * Adds a value to this neuron's input.  Also, because this neuron is receiving input, it must be
	 * true that one of its incoming connections is active, so addInput also sets active to true for this
	 * network.
	 * 
	 * @param x Value to be added to the input.
	 */
	public void addInput(double x);
	
	/**
	 * Sets this neuron's input to zero.
	 */
	public void clearInput();
	
	/**
	 * Returns the output value of this neuron.
	 * 
	 * @return Output of this neuron.
	 */
	public double getOutput();

	/**
	 * Returns the gene that created this neuron.
	 *
	 * @return Node gene that created this neuron.
	 */
	public Node getNode();
}


