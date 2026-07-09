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
 * <p>
 * Update for color: The CPPN will support grey/color outputs. This
 * changes our assumption about the input/output layer model since we now
 * support more then one output. Codewise, this means the client must
 * implement the interpretation of the output.
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
	@Deprecated
	public double []evaluate(double x, double y);
	
	/**
	 * Computes the value of the network with the given input. You
	 * can read the results from the output layer.
	 * 
	 * @param x The x input
	 * @param y The y input
	 */
	public void evaluateAt(double x, double y);

	/**
	 * Gets the output values associated with the specified <code>index</code>.
	 * Multiple output layers may exist to facilitate viewing different
	 * subnets of the main network.
	 * 
	 * @param index The output layer index
	 * @return The output values
	 */
	public double []readOutput(int index);
	
	/**
	 * Retreives the genome that created this CPPN.
	 * 
	 * @return The genome that constructed this CPPN
	 */
	public Genome getGenome();
}
