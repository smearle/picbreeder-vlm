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

import client.evolution.Link;

/**
 * A Connection takes an input value from a neuron, modifies 
 * the value by multiplying it by its weight, and then sends this
 * value its output neuron.
 * 
 * @author Adam Campbell
 */

public interface Connection {
	/**
	 * If the input neuron is active, its value is weighted by this connection and then sent
	 * to the output neuron.
	 */
	public void transmit();
	
	/**
	 * Retrieves the gene that this connection represents.
	 * 
	 * @return The link gene
	 */
	public Link getLink();
	
	/**
	 * Gets the source of this connection.
	 * 
	 * @return The source
	 */
	public Neuron getSource();

	/**
	 * Gets the destination of this connection.
	 * 
	 * @return The source
	 */
	public Neuron getDestination();

	/**
	 * Returns boolean indicating whether or not this connection is active. A connection
	 * becomes active when its input neuron becomes active.
	 *
	 * @return Indication of whether or not this connection is active.
	 */
	public boolean isActive();
}

