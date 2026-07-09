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
import client.TaskListener;
import client.evolution.Individual;

abstract class AbstractRenderer implements Renderer {
	private final TaskListener listener;
	protected Collection <Individual> individuals;
	
	protected AbstractRenderer(TaskListener owner) {
		listener = owner;
	}
	
	public void render(Collection <Individual> individuals) {
		this.individuals = individuals;
		
		spawnThread();
	}
	
	protected final boolean consider(Individual individual) {
		// TODO fix phenotype quality
		return !individual.isRendered() || RenderingAlgorithmInstance.get().quality() > individual.getQuality();
	}
	
	protected abstract void runOnce();
	
	private void spawnThread() {
		new RenderThread().start();
	}
	
	private class RenderThread extends Thread {
		RenderThread() {
			super();
			listener.notifyTaskStarting();
		}
		
		public void run() {
			try {
				// sleep so swing can update everything nicely
				// without stealing the system from it
				Thread.sleep(50);
			}
			catch(Exception e) {
				e.printStackTrace();
			}
			
			runOnce();
			listener.notifyTaskFinished();
		}
	}
}
