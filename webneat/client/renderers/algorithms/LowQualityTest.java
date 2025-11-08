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

import client.cppn.CPPNFactoryInstance;
import client.cppn.Network;
import client.evolution.Phenotype;
import client.evolution.Individual;

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
	
	public void render(Individual ind) {
		Phenotype [] ps = new Phenotype[ind.countPhenotypes()];
		for(int i = 0; i < ps.length; i++)
			ps[i] = ind.getPhenotype(i);
		
		final int h = ps[0].getHeight();
		final int w = ps[0].getWidth();

		final Network net = CPPNFactoryInstance.get().createNetwork(ind.getGenome());
		
		double fx, fy;
		double []output;
		double []copy = new double[4];
		
		for(int y = 0; y < h; y += stride) {
			fy = ps[0].computeInputY(y);
			
			for(int x = 0; x < w; x += stride) {
				fx = ps[0].computeInputX(x);
				net.evaluateAt(fx, fy);
				
				for(int a = 0; a < ps.length; a++) {
					output = net.readOutput(a);
					
					for(int i = 0; i < stride; i++)
						for(int j = 0; j < stride; j++) {
							System.arraycopy(output, 0, copy, 0, 3);
							ps[a].setValue(x+i, y+j, copy);
						}
				}
			}
		}

		
		ind.setQuality(quality());
		ind.notifyCompleted();
	}
}
