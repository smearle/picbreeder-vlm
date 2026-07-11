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

package client.renderers.algorithms;

import client.ParameterTableInstance;
import client.renderers.RenderingAlgorithm;

/**
 * The AbstractRenderingAlgorithm implements the code to maintain
 * quality across different implementations. Essentially, it considers
 * the number of times a pixel is sampled to be directly proportional
 * to its quality.
 * 
 * @author Nick
 *
 */
abstract class AbstractRenderingAlgorithm implements RenderingAlgorithm {
	private final int quality;
	
	/**
	 * Constructs an algorithm given that each pixel is sampled exactly
	 * <code>samplesPerPixel</code> (on average).
	 * 
	 * @param samplesPerPixel The number of samples for each pixel
	 */
	protected AbstractRenderingAlgorithm(double samplesPerPixel) {
		final int WIDTH = ParameterTableInstance.get().getInteger("display", "width");
		final int HEIGHT = ParameterTableInstance.get().getInteger("display", "height");
		
		quality = (int) (WIDTH * HEIGHT * samplesPerPixel + 1e-9);
	}
	
	public final int quality() {
		return quality;
	}
}
