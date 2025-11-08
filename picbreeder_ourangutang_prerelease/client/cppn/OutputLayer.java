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
 * The OutputLayer abstracts the output neurons from the image drawing
 * process.  Since the only values that a renderer cares about are the RGB
 * values, the OutputLayer is responsible for interpreting the CPPN's
 * output neurons and transcribing them to RGB values, in the range of
 * [0.0, 1.0].
 * 
 * @author Adam Campbell
 */
public interface OutputLayer {
	/**
	 * Returns an array containing the RGB interpretation of the output layer.
	 * This output is in the range [0.0, 1.0].
	 * 
	 * @return Array with one element representing the ink level
	 * 	output by the network using this output layer.
	 */
	public double[] readOutput();
}
