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

package client.renderers;

import client.cppn.Network;
import client.evolution.Individual;

/**
 * A RenderingAlgorithm contains the code nescessary to render an image
 * from the CPPN and Phenotype.  This implementation class should contain
 * the information such as pixel scheduling (interlaced, raster lines,
 * etc.) and swing notifications.
 * 
 * @author Nick
 */

public interface RenderingAlgorithm {
	/**
	 * Renders the image given the specified network.
	 * 
	 * @param image The result image
	 */
	public void render(Individual image);
	
	/**
	 * Gets the quality level of this rendering algorithm.
	 * Higher quality is considered better.
	 * 
	 * @return The quality
	 */
	public int quality();
}
