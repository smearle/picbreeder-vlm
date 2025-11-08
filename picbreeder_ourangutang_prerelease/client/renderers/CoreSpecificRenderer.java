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

import java.util.*;
import client.evolution.*;
import client.*;

/**
 * A CoreSpecificRenderer renders a set of individuals by detecting
 * the number of virtual processors that a system has and spawning
 * at most that many threads (plus one for management). The individuals
 * are delegated to the threads for rendering.
 *  
 * @author Nick
 *
 */
public class CoreSpecificRenderer extends AbstractRenderer implements TaskListener {
	private int counter;
	private static final int NUMBER_OF_PROCESSORS;
	private static final String NAME;
	
	protected final Renderer []subrenderer;
	
	static {
		NUMBER_OF_PROCESSORS = Runtime.getRuntime().availableProcessors();
		
		if(NUMBER_OF_PROCESSORS == 2)
			NAME = "Dual Core Renderer";
		else if(NUMBER_OF_PROCESSORS == 4)
			NAME = "Quad Core Renderer";
		else
			NAME = Integer.toString(NUMBER_OF_PROCESSORS) + " CPU Renderer";
	}
	
	/**
	 * Checks if this CPU has more than one processor or core.
	 * 
	 * @return <code>true</code> if it does, <code>false</code> if not.
	 */
	public static boolean isSupported() {
		return NUMBER_OF_PROCESSORS > 1;
	}

	/**
	 * Creates a renderer that notifies the owner before
	 * rendering is started and after rendering has completed.
	 * 
	 * @param owner The owning object
	 */
	public CoreSpecificRenderer(TaskListener owner) {
		super(owner);
		counter = 0;
		
		subrenderer = new SingleRenderer[NUMBER_OF_PROCESSORS];
		for(int i = 0; i < NUMBER_OF_PROCESSORS; i++)
			subrenderer[i] = new SingleRenderer(this);
	}
	
	public String toString() {
		return NAME;
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
		// figure out which images are not ready
		LinkedList <Individual> toRender = new LinkedList <Individual> ();
		
		for(Individual ind : individuals)
			if(consider(ind))
				toRender.add(ind);
		
		counter = toRender.size();
		
		// maybe everything is rendered?
		if(counter > 0) {
			
			// split the work evenly
			int processor = 0;
			Vector <LinkedList <Individual> > work = new Vector <LinkedList <Individual> >();
			
			counter = Math.min(counter, NUMBER_OF_PROCESSORS);
			for(int i = 0; i < counter; i++)
				work.add(new LinkedList <Individual> ());
			
			for(Individual ind : toRender)
				work.get(processor++ % NUMBER_OF_PROCESSORS).add(ind);
			
			// tell the subrenderers to render their portions
			processor = 0;
			for(LinkedList <Individual> ind : work)
				subrenderer[processor++].render(ind);

			// wait for the threads to finish.
			// there's a small chance that all of them already
			// finished, so make sure not to block unless you have to
			synchronized(this) {
				if(counter > 0) {
					try {
						wait();
					}
					catch(Exception e) {
						// why'd it get interrupted?
						e.printStackTrace();
					}
				}
			}
		}
	}
}
