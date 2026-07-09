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

import client.TaskListener;
import client.cppn.Network;
import client.cppn.CPPNFactoryInstance;
import client.evolution.Individual;
import client.evolution.Phenotype;

/**
 * RenderingAlgorithm that renders the entire image in the background using multiple threads.
 * This is useful for large images.
 * 
 * @author Nick
 */

public final class Parallel extends AbstractRenderingAlgorithm implements TaskListener {
	private final int numThreads;
	private int activeThreads;
	
	public Parallel(int numThreads) {
		super(1.0);
		
		this.numThreads = numThreads;
		activeThreads = 0;
	}
	
	public void notifyTaskStarting() {
		synchronized(this) {
			activeThreads++;
		}
	}
	
	public void notifyTaskFinished() {
		synchronized(this) {
			activeThreads--;
			if(activeThreads == 0)
				notify();
		}
	}
	
	public void render(Individual ind) {
		final int h = ind.getPhenotype(0).getHeight();
		final int w = ind.getPhenotype(0).getWidth();
		
		int increment = h / numThreads;
		
		for(int y = 0; y < h; y += increment) {
			new RenderThread(ind, 0, y, w, Math.min(y + increment, h)).start();
		}

		synchronized(this) {
			if(activeThreads > 0) {
				try {
					wait();
				}
				catch(java.lang.InterruptedException e) {
					e.printStackTrace();
				}
			}
			
			ind.setQuality(quality());
			ind.notifyCompleted();
		}
	}
	
	private void runOnce(Individual individual, int xMin, int yMin, int xMax, int yMax) {
		Phenotype [] ps = new Phenotype[individual.countPhenotypes()];
		
		for(int i = 0; i < ps.length; i++)
			ps[i] = individual.getPhenotype(i);
		
		final Network net = CPPNFactoryInstance.get().createNetwork(individual.getGenome());
		
		double fx, fy;
		double [] copy = new double[4];
		double [] outputs;
		
		for(int y = yMin; y < yMax; y++) {
			fy = ps[0].computeInputY(y);
			
			for(int x = xMin; x < xMax; x++) {
				fx = ps[0].computeInputX(x);
				net.evaluateAt(fx, fy);
				
				for(int i = 0; i < ps.length; i++) {
					outputs = net.readOutput(i);
					System.arraycopy(outputs, 0, copy, 0, 3);
					ps[i].setValue(x, y, copy);
				}
			}
		}
	}
	
	private class RenderThread extends Thread {
		private final int xMin, yMin, xMax, yMax;
		private final Individual ind;
		
		RenderThread(Individual ind, int xMin, int yMin, int xMax, int yMax) {
			super();
			this.xMin = xMin;
			this.yMin = yMin;
			this.xMax = xMax;
			this.yMax = yMax;
			this.ind = ind;
			notifyTaskStarting();
		}
		
		public void run() {
			runOnce(ind, xMin, yMin, xMax, yMax);
			notifyTaskFinished();
		}
	}
}
