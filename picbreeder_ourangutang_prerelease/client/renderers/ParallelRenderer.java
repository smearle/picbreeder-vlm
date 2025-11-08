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

import java.util.LinkedList;
import client.evolution.*;
import client.*;

/**
 * A ParallelRenderer renders the individuals on different threads (one
 * thread for each individual). When all threads have completed, it notifies
 * a task listener that all thread are complete.
 * <p>
 * This implementation will spawn at most <code>objects.size() + 1</code>
 * threads during rendering. In certain circumstances, less threads may be
 * spawned (when certain individuals do not need to render).
 * 
 * @author Nick
 */
public class ParallelRenderer extends AbstractRenderer implements TaskListener {
	private int counter;
	protected final Renderer []subrenderer;

	/**
	 * Creates a renderer that notifies the owner before
	 * rendering is started and after rendering has completed.
	 * 
	 * @param owner The owning object
	 */
	public ParallelRenderer(TaskListener owner) {
		super(owner);
		
		final int numberOfImages = ParameterTableInstance.get().getInteger("evolution", "population size");
		
		subrenderer = new SingleRenderer[numberOfImages];
		for(int i = 0; i < numberOfImages; i++)
			subrenderer[i] = new SingleRenderer(this);
		counter = 0;
	}
	
	public String toString() {
		return "Simultaneous Renderer";
	}
	
	public void notifyTaskStarting() {
	}
	
	public void notifyTaskFinished() {
		synchronized(this) {
			counter--;
			if(counter == 0)
				notify();
		}
	}
	
	protected void runOnce() {
		LinkedList <Individual> toRender = new LinkedList <Individual> ();
		
		for(Individual ind : individuals)
			if(consider(ind))
				toRender.add(ind);
		
		counter = toRender.size();
		
		if(counter > 0) {
			int which = 0;
			for(Individual ind : toRender) {
				LinkedList <Individual> t = new LinkedList <Individual> ();
				t.add(ind);
				subrenderer[which++].render(t);
			}

			synchronized(this) {
				if(counter > 0) {
					try {
						wait();
					}
					catch(Exception e) {
						e.printStackTrace();
					}
				}
			}
		}
	}
}
