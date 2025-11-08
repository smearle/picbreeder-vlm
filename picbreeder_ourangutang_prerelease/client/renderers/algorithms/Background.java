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

import client.cppn.Network;
import client.evolution.Phenotype;

/**
 * The Background algorithm renders the entire image in the background
 * and notifies the image only once it has completed rendering it.
 * 
 * @author Nick
 *
 */

public class Background extends AbstractRenderingAlgorithm {
	public Background() {
		super(1.0);
	}
	
	public void render(Network net, Phenotype image) {
		final int h = image.getHeight();
		final int w = image.getWidth();
		
		double fx, fy;
		double []output;
		
		for(int y = 0; y < h; y++) {
			fy = image.computeInputY(y);
			
			for(int x = 0; x < w; x++) {
				fx = image.computeInputX(x);
				output = net.evaluate(fx, fy);
				image.setValue(x, y, output);
			}
		}

		image.setQuality(quality());
		image.notifyCompleted();
	}
}
