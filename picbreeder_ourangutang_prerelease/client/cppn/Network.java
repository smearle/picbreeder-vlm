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

import client.evolution.Genome;

/**
 * The Network class represents all parts of a CPPN. This is the analogue
 * of a neural network.
 * 
 * @author Adam Campbell
 */

public interface Network {
	/**
	 * Computes the value of the network with the given inputs.
	 * 
	 * @param x The x input
	 * @param y The y input
	 * @return The output
	 */
	public double []evaluate(double x, double y);
	
	/**
	 * Retreives the genome that created this CPPN.
	 * 
	 * @return The genome that constructed this CPPN
	 */
	public Genome getGenome();
}
