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

import client.cppn.Connection;
import client.cppn.Neuron;
import client.evolution.Link;

/**
 * A connection takes an input value from a neuron, modifies 
 * the value by multiplying it by its weight, and then sends this
 * value its output neuron.
 * 
 * @author Adam Campbell
 */
final class DefaultConnection implements Connection {
	/**
	 * Represents the neuron that this connection gets its value from.
	 *
	 */
	private final Neuron source;
	/**
	 * Represents the neuron that this connection sends its output value to.
	 *
	 */
	private final Neuron destination;
	/**
	 * Represents the weight of this connection.  This value should be in the range [-MAX_WEIGHT,MAX_WIEGHT].
	 *
	 */
	private double weight;
	/**
	 * Represents whether or not this connection is active (ie. its input neuron is active.
	 *
	 */
	private boolean active;
	
	/**
	 * The gene that represents this connection.
	 */
	private final Link link;
	
	/**
	 * Constructs a new connection given its source neuron, destination neuron, and 
	 * the weight obtained through the link parameter.
	 *
	 * @param s Neuron that this connection will get its input value from.
	 * @param d Neuron that this connection will send its output value to.
	 * @param link The weight for this connection is obtained from the link.
	 */
	public DefaultConnection(Neuron s, Neuron d, Link link){
		source = s;
		destination = d;
		weight = link.getWeight(); // clamping in genome, cppn should be ok
		active = false;
		this.link = link;
	}
	
	public void transmit(){
		if(source.isActive()){
			active = true;
			destination.addInput(weight * source.getOutput());
		}
	}
	
	public void setDestinationActive(){
		if(active) ((DefaultNeuron)destination).setActive(true);
	}
	
	public Link getLink() {
		return link;
	}
	
	public boolean isActive(){
		return active;
	}
	
	public void clear(){
		active = false;
	}
	
	public Neuron getSource() {
		return source;
	}
	
	public Neuron getDestination() {
		return destination;
	}
}


