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
import client.cppn.CPPNFactoryInstance;
import client.evolution.Individual;
import client.evolution.Phenotype;

/**
 * RenderingAlgorithm that renders each row, notifying the phenotype as it
 * proceeds.
 * 
 * @author Nick
 */

public final class RowBased extends AbstractRenderingAlgorithm {
	public RowBased() {
		super(1.0);
	}

	public void render(Individual ind) {
		Phenotype [] ps = new Phenotype[ind.countPhenotypes()];
		for(int i = 0; i < ps.length; i++)
			ps[i] = ind.getPhenotype(i);
		
		final int h = ps[0].getHeight();
		final int w = ps[0].getWidth();
		
		final Network net = CPPNFactoryInstance.get().createNetwork(ind.getGenome());
		
		double fx, fy;
		double [] copy = new double[4];
		double [] outputs;
		
		for(int y = 0; y < h; y++) {
			ind.notifyUpdated();
			
			fy = ps[0].computeInputY(y);
			
			for(int x = 0; x < w; x++) {
				fx = ps[0].computeInputX(x);
				net.evaluateAt(fx, fy);
				
				for(int i = 0; i < ps.length; i++) {
					outputs = net.readOutput(i);
					System.arraycopy(outputs, 0, copy, 0, 3);
					ps[i].setValue(x, y, copy);
				}
			}
		}

		ind.setQuality(quality());
		ind.notifyCompleted();
	}
}
