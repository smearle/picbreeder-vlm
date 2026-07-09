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

/**
 * The InputLayer abstracts the input layer of the neural network.  This layer
 * encapsulates the nueral network's input neurons into two distinct invariants:
 * The x and y values (normalized from -1 to 1).  All other inputs are constant
 * or derivable.
 * <p>
 * For example, The bias is set via the parameter table and is constant in a series.
 * Also, the "distance from center" can be calculate from the variables x and y.
 * <p>
 * We decided at design time that no other per-pixel variables would exist for webneat.
 * If a per-image or per generation variable exists, you may add functions within the
 * implementation to set class members.  The x and y values are the only variables
 * that change per pixel.
 * 
 * @author Adam Campbell
 */
public interface InputLayer {
	/**
	 * Sets the values of the input neurons.
	 * 
	 * @param x Current value in the x direction.
	 * @param y Current value in the y direction.
	 */
	public void writeInput(double x, double y);
}
