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


package client.evolution.generators;

import client.*;
import client.evolution.*;
import client.utilities.*;

/**
 * 
 * 
 */
public class MutateActivationWithinAffinity extends AbstractMutator {
	private final double MUTATION_RATE;
	private final String affinity;
	
	public MutateActivationWithinAffinity(String affinity) {
		MUTATION_RATE = ParameterTableInstance.get().getDouble("evolution", "activation mutation rate");
		this.affinity = affinity;
	}
	
	public void mutate(Genome offspring) {
		for(Node node : offspring.getNodes())
			if(inAffinity(node) && Random.instance().nextBoolean(MUTATION_RATE))
				node.setActivation(ParameterTableInstance.get().getRandomItemFromSet("activations"));
	}
	
	private boolean inAffinity(Node n) {
		return n.getAffinity().equals(affinity);
	}
}
