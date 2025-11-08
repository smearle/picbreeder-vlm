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

import client.TaskListener;
import client.evolution.Individual;

/**
 * A SingleRenderer will render all individuals on a single thread in
 * the background. It will notify the caller via a TaskListener when
 * rendering has completed.
 * 
 * @author Nick
 */
public class SingleRenderer extends AbstractRenderer {
	/**
	 * Creates a renderer that notifies the owner before
	 * rendering is started and after rendering has completed.
	 * 
	 * @param owner The owning object
	 */
	public SingleRenderer(TaskListener owner) {
		super(owner);
	}

	public String toString() {
		return "Sequential Renderer";
	}
	
	protected void runOnce() {
		for(Individual ind : individuals)
			if(consider(ind))
				renderImplementation(ind);
	}
	
	protected void renderImplementation(Individual individual) {
		try {
			test.IndividualTest ind = (test.IndividualTest) individual;
			RenderingAlgorithmInstance.get().render(ind.getNetwork(), ind.getPhenotype());
			ind.setRendered(true);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
}
