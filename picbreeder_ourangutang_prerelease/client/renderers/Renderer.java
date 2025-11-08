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

import java.util.Collection;
import client.evolution.Individual;

/**
 * A Renderer is responsible for rendering a set of individuals.  The
 * renderer schedules them and applies the RenderingAlgorithm to each
 * one of them if the algorithm produces a better quality image.
 * 
 * @author Nick
 */
public interface Renderer {
	/**
	 * Schedules a set of individuals to be rendered by this
	 * renderer.
	 * 
	 * @param objects The individuals
	 */
	public void render(Collection <Individual> objects);
}
