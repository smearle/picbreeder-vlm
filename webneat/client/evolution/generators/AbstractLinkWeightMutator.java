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

import client.ParameterTableInstance;
import client.evolution.Genome;
import client.evolution.Link;
import client.evolution.generators.AbstractMutator;
import client.utilities.Random;

public abstract class AbstractLinkWeightMutator extends AbstractMutator {
	private final double MAX_WEIGHT;
	private final double MUTATION_RATE;
	private final double MIN_MUTATION_POWER;
	private final double MAX_MUTATION_POWER;
	private static double interpolationParameter = 0.0;
	
	protected AbstractLinkWeightMutator() {
		MAX_WEIGHT = ParameterTableInstance.get().getDouble("evolution", "max link weight");
		MUTATION_RATE = ParameterTableInstance.get().getDouble("evolution", "weight mutation rate");
		MIN_MUTATION_POWER = ParameterTableInstance.get().getDouble("evolution", "min weight mutation power");
		MAX_MUTATION_POWER = ParameterTableInstance.get().getDouble("evolution", "max weight mutation power");
	}
	
	public void mutate(Genome offspring) {
		for(Link link : offspring.getLinks())
			if(isValidLink(offspring, link) && Random.instance().nextBoolean(MUTATION_RATE))
				link.setWeight(mutateWeight(link.getWeight()));
	}
	
	protected abstract boolean isValidLink(Genome g, Link link);
	
	public static void setInterpolationParameter(double d) {
		interpolationParameter = d;
	}
	
	protected double computePower() {
		return interpolationParameter * MAX_MUTATION_POWER + (1.0 - interpolationParameter) * MIN_MUTATION_POWER;
	}
	
	protected double mutateWeight(double w) {
		// next gaussian is signed! oops
		double value = Random.instance().nextGaussian();
		return clamp(w + value * computePower());
	}
	
	protected double clamp(double w) {
		return Math.min(MAX_WEIGHT, Math.max(-MAX_WEIGHT, w));
	}
}
