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
 * The LOD algorithm first renders a blurry version of the image,
 * then it renders a higher quality version of the image. This is the
 * same as rendering the image with thw LowQuality algorithm followed
 * by the Background algorithm.
 * 
 * @author Nick
 *
 */
public final class LOD extends AbstractRenderingAlgorithm {
	private final int stride;
	
	public LOD() {
		super(1.0);
		stride = client.ParameterTableInstance.get().getInteger("display", "low resolution stride");
	}
	
	public void render(Network net, Phenotype image) {
		final int h = image.getHeight();
		final int w = image.getWidth();
		
		double fx, fy;
		double []output;
		double []copy = new double[4];
		
		for(int y = 0; y < h; y += stride) {
			fy = image.computeInputY(y);
			
			for(int x = 0; x < w; x += stride) {
				fx = image.computeInputX(x);
				output = net.evaluate(fx, fy);
				
				for(int i = 0; i < stride; i++)
					for(int j = 0; j < stride; j++) {
						System.arraycopy(output, 0, copy, 0, 3);
						image.setValue(x+i, y+j, copy);
					}
			}
		}
		
		image.notifyUpdated();

		for(int y = 0; y < h; y++) {
			fy = image.computeInputY(y);
			
			for(int x = 0; x < w; x++) {
				if(x % stride == 0 && y % stride == 0)
					continue;
				
				fx = image.computeInputX(x);
				output = net.evaluate(fx, fy);
				image.setValue(x, y, output);
			}
		}
		
		image.setQuality(quality());
		image.notifyCompleted();
	}
}
