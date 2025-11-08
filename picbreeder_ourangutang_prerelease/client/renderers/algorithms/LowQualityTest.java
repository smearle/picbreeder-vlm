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
 * The LowQuality algorithm renders the image at lower resolution
 * to decrease the render time.
 * 
 * @author Nick
 *
 */

public class LowQualityTest extends AbstractRenderingAlgorithm {
	private final int stride;
	
	public LowQualityTest() {
		this(client.ParameterTableInstance.get().getInteger("display", "low resolution stride"));
	}
	
	private LowQualityTest(int stride) {
		super(1.0 / (stride * stride));
		this.stride = stride;
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
		
		image.setQuality(quality());
		image.notifyCompleted();
	}
}
