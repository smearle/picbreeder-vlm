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
public class MutateLinks extends AbstractMutator {
	private final double MAX_WEIGHT;
	private final double MUTATION_RATE;
	
	public MutateLinks() {
		MAX_WEIGHT = ParameterTableInstance.get().getDouble("evolution", "max link weight");
		MUTATION_RATE = ParameterTableInstance.get().getDouble("evolution", "weight mutation rate");
	}
	
	public void mutate(Genome offspring) {
		for(Link link : offspring.getLinks())
			if(Random.instance().nextBoolean(MUTATION_RATE))
				link.setWeight(mutateWeight(link.getWeight()));
	}
	
	private double mutateWeight(double w) {
		if(Random.instance().nextBoolean())
			return clamp(w + Random.instance().nextGaussian());
		else
			return clamp(w - Random.instance().nextGaussian());
	}
	
	private double clamp(double w) {
		return Math.min(MAX_WEIGHT, Math.max(-MAX_WEIGHT, w));
	}
}
